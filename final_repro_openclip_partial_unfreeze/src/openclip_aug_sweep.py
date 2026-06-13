from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import transforms
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import CLASSES, ImageRecord, load_rgb_image, load_test_records, load_train_records
from src.dataset import validate_images


@dataclass(frozen=True)
class AugConfig:
    train_views: int
    tta_views: int
    c: float
    class4_weight: float = 1.0
    class4_train_views: int = 0


def effective_train_views(label: str, config: AugConfig) -> int:
    if label == "Class_4" and config.class4_train_views > 0:
        return config.class4_train_views
    return config.train_views


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_aug_transform(
    preprocess,
    crop_scale_min: float,
    jitter: float,
    interpolation: str = "bicubic",
) -> transforms.Compose:
    normalize = preprocess.transforms[-1]
    interpolation_mode = interpolation_from_name(interpolation)
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=interpolation_mode),
            transforms.RandomResizedCrop(
                224,
                scale=(crop_scale_min, 1.0),
                ratio=(0.9, 1.1),
                interpolation=interpolation_mode,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=jitter,
                contrast=jitter,
                saturation=jitter,
                hue=min(jitter / 5.0, 0.05),
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_eval_transform(preprocess, interpolation: str = "bicubic") -> transforms.Compose:
    """构造第 1 个确定性视图；用于测试 32x32 小图上采样插值对特征的影响。"""
    normalize = preprocess.transforms[-1]
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=interpolation_from_name(interpolation)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def interpolation_from_name(name: str) -> transforms.InterpolationMode:
    mapping = {
        "nearest": transforms.InterpolationMode.NEAREST,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "lanczos": transforms.InterpolationMode.LANCZOS,
    }
    if name not in mapping:
        raise ValueError(f"未知插值方式：{name}")
    return mapping[name]


def apply_color_mode(image: Image.Image, color_mode: str) -> Image.Image:
    """对病理小图做轻量颜色归一化，默认保持原 RGB。"""
    if color_mode == "rgb":
        return image
    if color_mode == "autocontrast":
        return ImageOps.autocontrast(image)
    if color_mode == "equalize":
        return ImageOps.equalize(image)
    if color_mode == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    if color_mode == "grayscale_autocontrast":
        return ImageOps.autocontrast(ImageOps.grayscale(image)).convert("RGB")
    raise ValueError(f"未知 color_mode：{color_mode}")


def deterministic_aug_image(image: Image.Image, transform, seed: int) -> torch.Tensor:
    random_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    tensor = transform(image)
    random.setstate(random_state)
    torch.random.set_rng_state(torch_state)
    return tensor


@torch.no_grad()
def encode_views(
    records: List[ImageRecord],
    model: nn.Module,
    preprocess,
    aug_transform,
    num_views: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    color_mode: str = "rgb",
    eval_transform=None,
) -> np.ndarray:
    all_views: list[np.ndarray] = []
    model.eval()

    for view_idx in range(num_views):
        view_features: list[np.ndarray] = []
        iterator = range(0, len(records), batch_size)
        for start in tqdm(iterator, desc=f"提取第 {view_idx + 1}/{num_views} 个视图", leave=False):
            batch_records = records[start : start + batch_size]
            tensors = []
            for local_idx, record in enumerate(batch_records):
                image = load_rgb_image(record.path)
                image = apply_color_mode(image, color_mode)
                if view_idx == 0:
                    tensors.append(eval_transform(image) if eval_transform is not None else preprocess(image))
                else:
                    image_seed = seed + view_idx * 100_003 + (start + local_idx) * 17
                    tensors.append(deterministic_aug_image(image, aug_transform, image_seed))
            batch = torch.stack(tensors).to(device)
            features = model.encode_image(batch)
            features = torch.nn.functional.normalize(features, dim=1)
            view_features.append(features.detach().cpu().numpy().astype(np.float32))
        all_views.append(np.vstack(view_features))
    return np.stack(all_views, axis=0)


def feature_cache_path(
    cache_dir: Path,
    model_name: str,
    pretrained: str,
    num_views: int,
    seed: int,
    records: List[ImageRecord],
) -> Path:
    latest_mtime = max(record.path.stat().st_mtime_ns for record in records)
    name = f"openclip_aug_{model_name}_{pretrained}_views{num_views}_seed{seed}_{len(records)}_{latest_mtime}.npy"
    return cache_dir / name.replace("/", "-")


def load_or_extract_views(
    records: List[ImageRecord],
    model_name: str,
    pretrained: str,
    num_views: int,
    cache_dir: str,
    batch_size: int,
    device: torch.device,
    seed: int,
    force: bool,
    crop_scale_min: float = 0.72,
    jitter: float = 0.15,
    color_mode: str = "rgb",
    interpolation: str = "bicubic",
) -> np.ndarray:
    import open_clip

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    aug_tag = f"scale{crop_scale_min:.2f}_jitter{jitter:.2f}_{color_mode}_{interpolation}".replace(".", "p")
    path = feature_cache_path(cache_root, model_name, pretrained, num_views, seed, records)
    path = path.with_name(path.stem + f"_{aug_tag}" + path.suffix)
    if path.exists() and not force:
        return np.load(path)

    weight_cache = cache_root / "openclip_weights"
    weight_cache.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
        cache_dir=str(weight_cache),
    )
    aug_transform = build_aug_transform(preprocess, crop_scale_min, jitter, interpolation)
    eval_transform = build_eval_transform(preprocess, interpolation)
    features = encode_views(records, model, preprocess, aug_transform, num_views, batch_size, device, seed, color_mode, eval_transform)
    if not np.isfinite(features).all():
        raise ValueError("增强视图特征中出现 NaN 或无穷大。")
    np.save(path, features)
    return features


def evaluate_config(
    features_by_view: np.ndarray,
    labels: np.ndarray,
    config: AugConfig,
    n_splits: int,
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out_of_fold = np.empty_like(labels, dtype=object)
    fold_ids = np.zeros(len(labels), dtype=np.int32)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        train_feature_rows = []
        train_label_rows = []
        for idx in train_idx:
            views = effective_train_views(str(labels[idx]), config)
            train_feature_rows.append(features_by_view[:views, idx, :])
            train_label_rows.extend([labels[idx]] * views)
        train_features = np.vstack(train_feature_rows)
        train_labels = np.asarray(train_label_rows)
        valid_features = features_by_view[: config.tta_views, valid_idx, :].mean(axis=0)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=config.c,
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
        sample_weight = np.ones(len(train_labels), dtype=np.float32)
        sample_weight[train_labels == "Class_4"] = config.class4_weight
        model.fit(train_features, train_labels, logisticregression__sample_weight=sample_weight)
        pred = model.predict(valid_features)
        out_of_fold[valid_idx] = pred
        fold_ids[valid_idx] = fold
        fold_rows.append(
            {
                "macro_f1": f1_score(labels[valid_idx], pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(labels[valid_idx], pred),
            }
        )

    recalls = recall_score(labels, out_of_fold, labels=CLASSES, average=None, zero_division=0)
    row = {
        **asdict(config),
        "macro_f1": f1_score(labels, out_of_fold, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(labels, out_of_fold),
        "fold_macro_f1_std": float(np.std([item["macro_f1"] for item in fold_rows])),
        "fold_balanced_accuracy_std": float(np.std([item["balanced_accuracy"] for item in fold_rows])),
    }
    for label, value in zip(CLASSES, recalls):
        row[f"recall_{label}"] = float(value)
    return row, out_of_fold, fold_ids


def train_final_model(features_by_view: np.ndarray, labels: np.ndarray, config: AugConfig, seed: int):
    train_feature_rows = []
    train_label_rows = []
    for idx, label in enumerate(labels):
        views = effective_train_views(str(label), config)
        train_feature_rows.append(features_by_view[:views, idx, :])
        train_label_rows.extend([label] * views)
    train_features = np.vstack(train_feature_rows)
    train_labels = np.asarray(train_label_rows)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.c,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )
    sample_weight = np.ones(len(train_labels), dtype=np.float32)
    sample_weight[train_labels == "Class_4"] = config.class4_weight
    model.fit(train_features, train_labels, logisticregression__sample_weight=sample_weight)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选 OpenCLIP + 数据增强/TTA 单模型方案。")
    parser.add_argument("--train_dir", default="train_few_shot", help="训练集根目录或类别目录根目录。")
    parser.add_argument("--output_dir", default="outputs/openclip_aug_sweep")
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--model_name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_views", type=int, default=5)
    parser.add_argument("--train_views", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--tta_views", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--cs", nargs="*", type=float, default=[0.03, 0.05, 0.07, 0.1, 0.15])
    parser.add_argument("--class4_weights", nargs="*", type=float, default=[1.0])
    parser.add_argument("--class4_train_views", nargs="*", type=int, default=[0])
    parser.add_argument("--crop_scale_min", type=float, default=0.72)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument(
        "--color_mode",
        default="rgb",
        choices=["rgb", "autocontrast", "equalize", "grayscale", "grayscale_autocontrast"],
    )
    parser.add_argument("--force_features", action="store_true", help="忽略增强特征缓存并重新提取。")
    return parser.parse_args()


def save_oof_predictions(
    records: List[ImageRecord],
    labels: np.ndarray,
    predictions: np.ndarray,
    fold_ids: np.ndarray,
    output_path: Path,
) -> None:
    rows = []
    for record, true_label, pred_label, fold in zip(records, labels, predictions, fold_ids):
        rows.append(
            {
                "filename": record.filename,
                "path": str(record.path),
                "true_label": true_label,
                "pred_label": pred_label,
                "fold": int(fold),
                "correct": bool(true_label == pred_label),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_misclassified_sheets(
    oof_path: Path,
    output_dir: Path,
    max_images_per_pair: int = 40,
    thumb: int = 72,
) -> None:
    data = pd.read_csv(oof_path)
    mistakes = data[data["true_label"] != data["pred_label"]].copy()
    if mistakes.empty:
        return

    pair_counts = (
        mistakes.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    pair_counts.to_csv(output_dir / "misclassified_pair_counts.csv", index=False)

    sheet_dir = output_dir / "misclassified_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for _, row in pair_counts.head(12).iterrows():
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        subset = mistakes[
            (mistakes["true_label"] == true_label) & (mistakes["pred_label"] == pred_label)
        ].head(max_images_per_pair)
        if subset.empty:
            continue

        cols = min(10, len(subset))
        rows = int(np.ceil(len(subset) / cols))
        gap = 6
        header = 26
        sheet = Image.new("RGB", (cols * thumb + (cols + 1) * gap, rows * thumb + (rows + 1) * gap + header), "white")
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(sheet)
        draw.text((gap, 6), f"{true_label} -> {pred_label}  count={int(row['count'])}", fill="black", font=ImageFont.load_default())
        for i, item in enumerate(subset.itertuples(index=False)):
            image = load_rgb_image(item.path).resize((thumb, thumb), Image.NEAREST)
            x = gap + (i % cols) * (thumb + gap)
            y = header + gap + (i // cols) * (thumb + gap)
            sheet.paste(image, (x, y))
        sheet.save(sheet_dir / f"{true_label}_to_{pred_label}.png")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_train_records(args.train_dir)
    validate_images(records)
    labels = np.asarray([record.label for record in records])

    needed_views = max(max(args.train_views), max(args.tta_views), max(args.class4_train_views), args.max_views)
    print(
        f"已加载 {len(records)} 张训练图片，开始 {args.model_name}/{args.pretrained} "
        f"增强/TTA 筛选，共提取 {needed_views} 个视图。"
    )
    features_by_view = load_or_extract_views(
        records,
        args.model_name,
        args.pretrained,
        needed_views,
        args.cache_dir,
        args.batch_size,
        device,
        args.seed,
        args.force_features,
        args.crop_scale_min,
        args.jitter,
        args.color_mode,
    )

    rows = []
    predictions = {}
    fold_map = {}
    for train_views in args.train_views:
        for tta_views in args.tta_views:
            for c_value in args.cs:
                for class4_weight in args.class4_weights:
                    for class4_train_views in args.class4_train_views:
                        config = AugConfig(
                            train_views=train_views,
                            tta_views=tta_views,
                            c=c_value,
                            class4_weight=class4_weight,
                            class4_train_views=class4_train_views,
                        )
                        key = (
                            f"train{train_views}_tta{tta_views}_c{c_value}"
                            f"_w4{class4_weight}_v4{class4_train_views}"
                        )
                        print(f"正在评估 {key}...")
                        row, pred, fold_ids = evaluate_config(
                            features_by_view, labels, config, args.n_splits, args.seed
                        )
                        row.update(
                            {
                                "model_name": args.model_name,
                                "pretrained": args.pretrained,
                                "crop_scale_min": args.crop_scale_min,
                                "jitter": args.jitter,
                                "color_mode": args.color_mode,
                            }
                        )
                        rows.append(row)
                        predictions[key] = pred
                        fold_map[key] = fold_ids
                        print(
                            f"macro_f1={row['macro_f1']:.4f} "
                            f"balanced_accuracy={row['balanced_accuracy']:.4f}"
                        )

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "openclip_aug_sweep.csv", index=False)
    best = results.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0].to_dict()
    with (out_dir / "best_openclip_aug_config.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    best_config = AugConfig(
        train_views=int(best["train_views"]),
        tta_views=int(best["tta_views"]),
        c=float(best["c"]),
        class4_weight=float(best.get("class4_weight", 1.0)),
        class4_train_views=int(best.get("class4_train_views", 0)),
    )
    best_key = (
        f"train{best_config.train_views}_tta{best_config.tta_views}_c{best_config.c}"
        f"_w4{best_config.class4_weight}_v4{best_config.class4_train_views}"
    )
    cm = confusion_matrix(labels, predictions[best_key], labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / "best_confusion_matrix.csv")
    oof_path = out_dir / "best_oof_predictions.csv"
    save_oof_predictions(records, labels, predictions[best_key], fold_map[best_key], oof_path)
    save_misclassified_sheets(oof_path, out_dir)

    final_model = train_final_model(features_by_view, labels, best_config, args.seed)
    joblib.dump(
        {
            "model": final_model,
            "config": {
                "feature": "openclip_aug",
                "model_name": args.model_name,
                "pretrained": args.pretrained,
                "train_views": best_config.train_views,
                "tta_views": best_config.tta_views,
                "c": best_config.c,
                "class4_weight": best_config.class4_weight,
                "class4_train_views": best_config.class4_train_views,
                "class_weight": "balanced",
                "seed": args.seed,
                "crop_scale_min": args.crop_scale_min,
                "jitter": args.jitter,
                "color_mode": args.color_mode,
            },
            "classes": CLASSES,
        },
        out_dir / "final_openclip_aug_model.joblib",
    )

    print("\nOpenCLIP 增强/TTA 当前最佳单模型方案：")
    print(
        f"{args.model_name}/{args.pretrained} train_views={best_config.train_views} "
        f"tta_views={best_config.tta_views} C={best_config.c} "
        f"class4_weight={best_config.class4_weight} "
        f"class4_train_views={best_config.class4_train_views} "
        f"macro_f1={best['macro_f1']:.4f} balanced_accuracy={best['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
