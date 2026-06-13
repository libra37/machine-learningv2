from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import CLASSES, IMAGE_EXTENSIONS, ImageRecord, load_rgb_image, load_test_records, load_train_records, validate_images
from src.openclip_aug_sweep import deterministic_aug_image
from src.openclip_finetune import (
    FinetuneConfig,
    FewShotDataset,
    OpenCLIPClassifier,
    build_eval_aug_transform,
    build_train_transform,
    classification_loss,
    count_trainable_parameters,
    device_from_arg,
    freeze_for_strategy,
    infer_feature_dim,
    seed_everything,
)


def resolve_test_dir(path: str | Path) -> Path:
    root = Path(path)
    if any(item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS for item in root.iterdir()):
        return root
    nested = root / root.name
    if nested.is_dir() and any(item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS for item in nested.iterdir()):
        return nested
    return root


def make_final_config(args: argparse.Namespace) -> FinetuneConfig:
    return FinetuneConfig(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=0,
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
        class0_weight=args.class0_weight,
        class1_weight=args.class1_weight,
        class2_weight=args.class2_weight,
        class3_weight=args.class3_weight,
        class4_weight=args.class4_weight,
        crop_scale_min=args.crop_scale_min,
        jitter=args.jitter,
        rotation_mode="none",
        rotation_degrees=0.0,
        unfreeze=args.unfreeze,
        init_sklearn=False,
        eval_tta_views=args.tta_views,
        head_type="linear",
    )


def build_model(config: FinetuneConfig, model_name: str, pretrained: str, cache_dir: str, device: torch.device):
    import open_clip

    weight_cache = Path(cache_dir) / "openclip_weights"
    weight_cache.mkdir(parents=True, exist_ok=True)
    base_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
        cache_dir=str(weight_cache),
    )
    freeze_for_strategy(base_model, config.unfreeze, config)
    feature_dim = infer_feature_dim(base_model, preprocess, device)
    model = OpenCLIPClassifier(base_model, feature_dim, len(CLASSES), config.head_type).to(device)
    return model, preprocess


def train_final_model(
    records: list[ImageRecord],
    config: FinetuneConfig,
    model_name: str,
    pretrained: str,
    cache_dir: str,
    device: torch.device,
    seed: int,
) -> tuple[OpenCLIPClassifier, object, list[dict[str, float]]]:
    seed_everything(seed)
    model, preprocess = build_model(config, model_name, pretrained, cache_dir, device)
    trainable_params, total_params = count_trainable_parameters(model)
    print(f"可训练参数：{trainable_params} / {total_params} ({trainable_params / max(total_params, 1):.2%})")

    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    train_transform = build_train_transform(
        preprocess,
        config.crop_scale_min,
        config.jitter,
        config.rotation_mode,
        config.rotation_degrees,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        FewShotDataset(records, label_to_idx, train_transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    class_weights = torch.ones(len(CLASSES), dtype=torch.float32, device=device)
    class_weights[label_to_idx["Class_0"]] = config.class0_weight
    class_weights[label_to_idx["Class_1"]] = config.class1_weight
    class_weights[label_to_idx["Class_2"]] = config.class2_weight
    class_weights[label_to_idx["Class_3"]] = config.class3_weight
    class_weights[label_to_idx["Class_4"]] = config.class4_weight

    backbone_params = [param for param in model.openclip_model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.lr_backbone},
            {"params": model.head.parameters(), "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = classification_loss(logits, targets.to(device), class_weights, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"epoch={epoch:03d} train_loss={mean_loss:.4f}")
    return model, preprocess, history


@torch.no_grad()
def predict_test_tta(
    model: OpenCLIPClassifier,
    records: list[ImageRecord],
    preprocess,
    config: FinetuneConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    aug_transform = build_eval_aug_transform(
        preprocess,
        config.crop_scale_min,
        config.jitter,
        config.rotation_mode,
        config.rotation_degrees,
    )
    all_logits = []
    for view_idx in range(config.eval_tta_views):
        view_logits = []
        for start in tqdm(range(0, len(records), batch_size), desc=f"预测 TTA view {view_idx + 1}/{config.eval_tta_views}", leave=False):
            tensors = []
            for offset, record in enumerate(records[start : start + batch_size]):
                image = load_rgb_image(record.path)
                if view_idx == 0:
                    tensors.append(preprocess(image))
                else:
                    image_seed = seed + view_idx * 100_003 + (start + offset) * 17
                    tensors.append(deterministic_aug_image(image, aug_transform, image_seed))
            logits = model(torch.stack(tensors).to(device))
            view_logits.append(logits.detach().cpu())
        all_logits.append(torch.cat(view_logits, dim=0))
    mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    probs = F.softmax(mean_logits, dim=1).numpy()
    pred_idx = probs.argmax(axis=1)
    return pred_idx, probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练当前最优部分解冻单模型，并生成测试集 submission.csv。")
    parser.add_argument("--train_dir", default="train_few_shot")
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/openclip_partial_unfreeze_final_submission")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--model_name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=29)
    parser.add_argument("--tta_views", type=int, default=3)
    parser.add_argument("--unfreeze", default="emb_first3_last6")
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.10)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--crop_scale_min", type=float, default=0.85)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--class0_weight", type=float, default=1.15)
    parser.add_argument("--class1_weight", type=float, default=1.0)
    parser.add_argument("--class2_weight", type=float, default=1.25)
    parser.add_argument("--class3_weight", type=float, default=1.0)
    parser.add_argument("--class4_weight", type=float, default=1.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    config = make_final_config(args)

    train_records = load_train_records(args.train_dir)
    validate_images(train_records)
    test_dir = resolve_test_dir(args.test_dir)
    test_records = load_test_records(test_dir)
    print(f"训练图片数：{len(train_records)}，测试图片数：{len(test_records)}，测试目录：{test_dir}")

    model, preprocess, history = train_final_model(
        train_records,
        config,
        args.model_name,
        args.pretrained,
        args.cache_dir,
        device,
        args.seed,
    )
    pred_idx, probs = predict_test_tta(model, test_records, preprocess, config, args.batch_size, device, args.seed)
    pred_labels = [CLASSES[int(index)] for index in pred_idx]

    submission = pd.DataFrame({"filename": [record.filename for record in test_records], "label": pred_labels})
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
    submission.to_csv(out_path, index=False)

    confidence = probs.max(axis=1)
    confidence_df = pd.DataFrame(
        {
            "filename": [record.filename for record in test_records],
            "label": pred_labels,
            "confidence": confidence,
            **{f"prob_{label}": probs[:, idx] for idx, label in enumerate(CLASSES)},
        }
    )
    confidence_df.to_csv(out_dir / "test_predictions_with_confidence.csv", index=False)
    pd.DataFrame({"label": pred_labels}).value_counts().rename("count").reset_index().to_csv(
        out_dir / "test_pred_label_distribution.csv",
        index=False,
    )
    pd.DataFrame(history).to_csv(out_dir / "final_train_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "classes": CLASSES,
        },
        out_dir / "final_openclip_partial_unfreeze_model.pt",
    )
    with (out_dir / "final_config.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "test_dir": str(test_dir), "submission": str(out_path)}, f, ensure_ascii=False, indent=2)

    print(f"submission 已生成：{out_path}")
    print("预测类别分布：")
    print(submission["label"].value_counts().sort_index().to_string())
    print(
        "置信度："
        f"mean={confidence.mean():.4f}, median={np.median(confidence):.4f}, "
        f"p90={np.quantile(confidence, 0.9):.4f}, max={confidence.max():.4f}"
    )


if __name__ == "__main__":
    main()
