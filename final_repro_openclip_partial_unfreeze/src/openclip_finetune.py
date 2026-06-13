from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import CLASSES, ImageRecord, load_rgb_image, load_train_records, validate_images
from src.openclip_aug_sweep import (
    deterministic_aug_image,
    load_or_extract_views,
    save_misclassified_sheets,
    save_oof_predictions,
)


@dataclass(frozen=True)
class FinetuneConfig:
    lr_backbone: float
    lr_head: float
    weight_decay: float
    epochs: int
    patience: int
    batch_size: int
    label_smoothing: float
    class0_weight: float
    class1_weight: float
    class2_weight: float
    class3_weight: float
    class4_weight: float
    crop_scale_min: float
    jitter: float
    rotation_mode: str
    rotation_degrees: float
    unfreeze: str
    init_sklearn: bool = False
    init_c: float = 0.08
    init_train_views: int = 3
    eval_tta_views: int = 3
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    head_type: str = "linear"
    adapter_dim: int = 24
    adapter_scale: float = 1.0
    adapter_dropout: float = 0.0
    lr_adapter: float = 1e-3
    pairwise_margin_weight: float = 0.0
    pairwise_margin: float = 0.0
    pairwise_class_set: str = "Class_2,Class_3,Class_4"


class FewShotDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        label_to_idx: dict[str, int],
        transform,
    ) -> None:
        self.records = records
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_rgb_image(record.path)
        label = self.label_to_idx[str(record.label)]
        return self.transform(image), label


class OpenCLIPClassifier(nn.Module):
    def __init__(
        self,
        openclip_model: nn.Module,
        feature_dim: int,
        num_classes: int,
        head_type: str = "linear",
        adapter_dim: int = 24,
        adapter_scale: float = 1.0,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.openclip_model = openclip_model
        self.head_type = head_type
        self.adapter_scale = adapter_scale
        self.adapter_norm = nn.LayerNorm(feature_dim) if head_type in {"ln_residual", "ln_gated_residual"} else nn.Identity()
        self.adapter_down = nn.Linear(feature_dim, adapter_dim) if head_type != "linear" else None
        self.adapter_up = nn.Linear(adapter_dim, feature_dim) if head_type != "linear" else None
        self.adapter_dropout = nn.Dropout(adapter_dropout)
        self.adapter_gate = nn.Parameter(torch.tensor(-2.0)) if head_type == "ln_gated_residual" else None
        self.head = nn.Linear(feature_dim, num_classes)
        if self.adapter_up is not None:
            nn.init.zeros_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)

    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.head_type == "linear":
            return features
        if self.adapter_down is None or self.adapter_up is None:
            raise ValueError(f"未知分类头类型：{self.head_type}")
        adapter_input = self.adapter_norm(features)
        delta = self.adapter_up(self.adapter_dropout(F.gelu(self.adapter_down(adapter_input))))
        scale = self.adapter_scale
        if self.adapter_gate is not None:
            scale = scale * torch.sigmoid(self.adapter_gate)
        return F.normalize(features + scale * delta, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.openclip_model.encode_image(images)
        features = torch.nn.functional.normalize(features, dim=1)
        features = self.adapt_features(features.float())
        return self.head(features)

    def adapter_parameters(self) -> list[nn.Parameter]:
        if self.head_type == "linear":
            return []
        params: list[nn.Parameter] = []
        if self.adapter_down is not None:
            params += list(self.adapter_down.parameters())
        if self.adapter_up is not None:
            params += list(self.adapter_up.parameters())
        if isinstance(self.adapter_norm, nn.LayerNorm):
            params += list(self.adapter_norm.parameters())
        if self.adapter_gate is not None:
            params.append(self.adapter_gate)
        return params


class LoRALinear(nn.Module):
    """冻结原 Linear，只训练低秩增量；用于小样本下的参数高效微调。"""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        device = base.weight.device
        dtype = base.weight.dtype
        self.base = base
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        for param in self.base.parameters():
            param.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


class LoRAMultiheadAttention(nn.Module):
    """给 PyTorch MultiheadAttention 加 QKV/输出投影 LoRA，专门用于 OpenCLIP ViT 小样本微调。"""

    def __init__(self, base: nn.MultiheadAttention, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if base.batch_first:
            raise ValueError("当前实现假设 MultiheadAttention 使用 [seq, batch, dim]，不支持 batch_first=True。")
        device = base.in_proj_weight.device
        dtype = base.in_proj_weight.dtype
        self.base = base
        self.embed_dim = base.embed_dim
        self.num_heads = base.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.qkv_down = nn.Linear(self.embed_dim, rank, bias=False)
        self.qkv_up = nn.Linear(rank, self.embed_dim * 3, bias=False)
        self.out_down = nn.Linear(self.embed_dim, rank, bias=False)
        self.out_up = nn.Linear(rank, self.embed_dim, bias=False)
        for param in self.base.parameters():
            param.requires_grad = False
        nn.init.kaiming_uniform_(self.qkv_down.weight, a=5**0.5)
        nn.init.zeros_(self.qkv_up.weight)
        nn.init.kaiming_uniform_(self.out_down.weight, a=5**0.5)
        nn.init.zeros_(self.out_up.weight)
        self.to(device=device, dtype=dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if key is not query or value is not query:
            # OpenCLIP ViT block 只会走 self-attention；若后续模型结构变化，回退到原模块更稳。
            return self.base(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if key_padding_mask is not None:
            return self.base(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        seq_len, batch_size, _ = query.shape
        qkv = F.linear(query, self.base.in_proj_weight, self.base.in_proj_bias)
        qkv = qkv + self.qkv_up(self.qkv_down(self.dropout(query))) * self.scaling
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(x: torch.Tensor) -> torch.Tensor:
            return x.permute(1, 0, 2).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)
        if is_causal:
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device).triu(1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim).permute(1, 0, 2)
        out = self.base.out_proj(attn)
        out = out + self.out_up(self.out_down(self.dropout(attn))) * self.scaling
        weights = None if not need_weights else weights.mean(dim=1)
        return out, weights


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    config: FinetuneConfig,
) -> torch.Tensor:
    base_loss = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    if config.pairwise_margin_weight <= 0:
        return base_loss

    class_names = [item.strip() for item in config.pairwise_class_set.split(",") if item.strip()]
    if len(class_names) < 2:
        return base_loss
    hard_classes = torch.tensor([CLASSES.index(name) for name in class_names], dtype=torch.long, device=targets.device)
    in_hard_set = (targets[:, None] == hard_classes[None, :]).any(dim=1)
    if not torch.any(in_hard_set):
        return base_loss

    hard_logits = logits[in_hard_set][:, hard_classes]
    hard_targets = targets[in_hard_set]
    hard_target_pos = hard_targets[:, None] == hard_classes[None, :]
    true_logits = hard_logits[hard_target_pos]
    other_logits = hard_logits.masked_fill(hard_target_pos, -1e9).max(dim=1).values
    pairwise_loss = F.relu(config.pairwise_margin + other_logits - true_logits).mean()
    return base_loss + config.pairwise_margin_weight * pairwise_loss


def build_rotation_transform(rotation_mode: str, rotation_degrees: float):
    """构造旋转增强；none 表示不旋转，small 表示小角度随机旋转，right_angle 表示 90 度离散旋转。"""
    if rotation_mode == "none" or rotation_degrees <= 0:
        return None
    if rotation_mode == "small":
        return transforms.RandomRotation(
            degrees=rotation_degrees,
            interpolation=transforms.InterpolationMode.BICUBIC,
            fill=0,
        )
    if rotation_mode == "right_angle":
        return transforms.RandomChoice(
            [
                transforms.Lambda(lambda image: image),
                transforms.Lambda(lambda image: image.rotate(90)),
                transforms.Lambda(lambda image: image.rotate(180)),
                transforms.Lambda(lambda image: image.rotate(270)),
            ]
        )
    raise ValueError(f"未知旋转增强模式：{rotation_mode}")


def build_train_transform(preprocess, crop_scale_min: float, jitter: float, rotation_mode: str, rotation_degrees: float):
    normalize = preprocess.transforms[-1]
    steps = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(
            224,
            scale=(crop_scale_min, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    rotation = build_rotation_transform(rotation_mode, rotation_degrees)
    if rotation is not None:
        steps.append(rotation)
    steps.extend(
        [
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
    return transforms.Compose(steps)


def build_eval_aug_transform(preprocess, crop_scale_min: float, jitter: float, rotation_mode: str, rotation_degrees: float):
    normalize = preprocess.transforms[-1]
    steps = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(
            224,
            scale=(crop_scale_min, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    rotation = build_rotation_transform(rotation_mode, rotation_degrees)
    if rotation is not None:
        steps.append(rotation)
    steps.extend(
        [
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
    return transforms.Compose(
        steps
    )


def install_mlp_lora(model: nn.Module, block_count: int, rank: int, alpha: float, dropout: float) -> None:
    blocks = getattr(getattr(model.visual, "transformer", None), "resblocks", None)
    if blocks is None:
        raise ValueError("当前 OpenCLIP visual encoder 不包含 transformer.resblocks，无法安装 LoRA。")
    for block in blocks[-block_count:]:
        mlp = getattr(block, "mlp", None)
        if mlp is None or not hasattr(mlp, "c_fc") or not hasattr(mlp, "c_proj"):
            raise ValueError("当前 OpenCLIP block 不包含 mlp.c_fc/c_proj，无法安装 MLP LoRA。")
        mlp.c_fc = LoRALinear(mlp.c_fc, rank, alpha, dropout)
        mlp.c_proj = LoRALinear(mlp.c_proj, rank, alpha, dropout)


def install_attention_lora(model: nn.Module, block_count: int, rank: int, alpha: float, dropout: float) -> None:
    """只在最后若干个 transformer block 的注意力层上安装 LoRA。"""
    blocks = getattr(getattr(model.visual, "transformer", None), "resblocks", None)
    if blocks is None:
        raise ValueError("当前 OpenCLIP visual encoder 不包含 transformer.resblocks，无法安装 Attention LoRA。")
    for block in blocks[-block_count:]:
        attn = getattr(block, "attn", None)
        if not isinstance(attn, nn.MultiheadAttention):
            raise ValueError("当前 OpenCLIP block.attn 不是 torch.nn.MultiheadAttention，无法安装 Attention LoRA。")
        block.attn = LoRAMultiheadAttention(attn, rank, alpha, dropout)


def set_visual_embedding_trainable(model: nn.Module) -> None:
    """解冻 ViT 输入嵌入相关参数，用于适配 32x32 小图的底层纹理分布。"""
    visual = model.visual
    for attr in ["class_embedding", "positional_embedding", "proj"]:
        value = getattr(visual, attr, None)
        if isinstance(value, nn.Parameter) or isinstance(value, torch.Tensor):
            value.requires_grad = True
    for attr in ["conv1", "ln_pre", "ln_post"]:
        module = getattr(visual, attr, None)
        if module is not None and hasattr(module, "parameters"):
            for param in module.parameters():
                param.requires_grad = True


def set_visual_blocks_trainable(model: nn.Module, first_count: int = 0, last_count: int = 0) -> None:
    blocks = getattr(getattr(model.visual, "transformer", None), "resblocks", None)
    if blocks is None:
        raise ValueError("当前 OpenCLIP visual encoder 不包含 transformer.resblocks，无法按 block 解冻。")
    total = len(blocks)
    selected = set(range(min(first_count, total)))
    if last_count > 0:
        selected.update(range(max(total - last_count, 0), total))
    for index in sorted(selected):
        for param in blocks[index].parameters():
            param.requires_grad = True


def parse_partial_unfreeze_strategy(strategy: str) -> Optional[tuple[bool, int, int]]:
    """解析 emb_first4_last6 / first4_last6_emb 一类部分解冻策略。"""
    has_emb = "emb" in strategy.split("_")
    first_match = re.search(r"first(\d+)", strategy)
    last_match = re.search(r"last(\d+)", strategy)
    if not has_emb and first_match is None and last_match is None:
        return None
    first_count = int(first_match.group(1)) if first_match else 0
    last_count = int(last_match.group(1)) if last_match else 0
    return has_emb, first_count, last_count


def freeze_for_strategy(model: nn.Module, strategy: str, config: Optional[FinetuneConfig] = None) -> None:
    for param in model.parameters():
        param.requires_grad = False

    partial_strategy = parse_partial_unfreeze_strategy(strategy)
    if partial_strategy is not None:
        has_emb, first_count, last_count = partial_strategy
        if has_emb:
            set_visual_embedding_trainable(model)
        set_visual_blocks_trainable(model, first_count=first_count, last_count=last_count)
        return

    if strategy == "head_only":
        return

    if strategy == "ln_proj_only":
        if hasattr(model.visual, "ln_post"):
            for param in model.visual.ln_post.parameters():
                param.requires_grad = True
        if isinstance(getattr(model.visual, "proj", None), torch.Tensor):
            model.visual.proj.requires_grad = True
        return

    if strategy == "last_block_norm_proj":
        blocks = getattr(getattr(model.visual, "transformer", None), "resblocks", None)
        if blocks is None:
            raise ValueError("当前 OpenCLIP visual encoder 不包含 transformer.resblocks，无法解冻最后 block 的 LayerNorm。")
        last_block = blocks[-1]
        for name, param in last_block.named_parameters():
            if "ln_" in name or "ln" in name:
                param.requires_grad = True
        if hasattr(model.visual, "ln_post"):
            for param in model.visual.ln_post.parameters():
                param.requires_grad = True
        if isinstance(getattr(model.visual, "proj", None), torch.Tensor):
            model.visual.proj.requires_grad = True
        return

    if strategy in {"last_block_attn_lora", "last_two_blocks_attn_lora"}:
        if config is None:
            raise ValueError("LoRA 策略需要传入 FinetuneConfig。")
        block_count = 2 if strategy == "last_two_blocks_attn_lora" else 1
        install_attention_lora(model, block_count, config.lora_rank, config.lora_alpha, config.lora_dropout)
        for module in model.modules():
            if isinstance(module, LoRAMultiheadAttention):
                for param in module.qkv_down.parameters():
                    param.requires_grad = True
                for param in module.qkv_up.parameters():
                    param.requires_grad = True
                for param in module.out_down.parameters():
                    param.requires_grad = True
                for param in module.out_up.parameters():
                    param.requires_grad = True
        if hasattr(model.visual, "ln_post"):
            for param in model.visual.ln_post.parameters():
                param.requires_grad = True
        return

    if strategy in {"last_block_mlp_lora", "last_two_blocks_mlp_lora"}:
        if config is None:
            raise ValueError("LoRA 策略需要传入 FinetuneConfig。")
        block_count = 2 if strategy == "last_two_blocks_mlp_lora" else 1
        install_mlp_lora(model, block_count, config.lora_rank, config.lora_alpha, config.lora_dropout)
        for module in model.modules():
            if isinstance(module, LoRALinear):
                for param in module.lora_down.parameters():
                    param.requires_grad = True
                for param in module.lora_up.parameters():
                    param.requires_grad = True
        if hasattr(model.visual, "ln_post"):
            for param in model.visual.ln_post.parameters():
                param.requires_grad = True
        return

    if strategy in {"last_block", "last_two_blocks", "last_block_proj"}:
        blocks = getattr(getattr(model.visual, "transformer", None), "resblocks", None)
        if blocks is None:
            raise ValueError("当前 OpenCLIP visual encoder 不包含 transformer.resblocks，无法只解冻最后 block。")
        block_count = 2 if strategy == "last_two_blocks" else 1
        for block in blocks[-block_count:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in blocks[-1].parameters():
            param.requires_grad = True
        if hasattr(model.visual, "ln_post"):
            for param in model.visual.ln_post.parameters():
                param.requires_grad = True
        if strategy == "last_block_proj" and isinstance(getattr(model.visual, "proj", None), torch.Tensor):
            model.visual.proj.requires_grad = True
        return

    raise ValueError(f"未知微调策略：{strategy}")


@torch.no_grad()
def infer_feature_dim(model: nn.Module, preprocess, device: torch.device) -> int:
    dummy = Image.new("RGB", (32, 32), "white")
    tensor = preprocess(dummy).unsqueeze(0).to(device)
    features = model.encode_image(tensor)
    return int(features.shape[-1])


def make_loaders(
    records: list[ImageRecord],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    label_to_idx: dict[str, int],
    train_transform,
    valid_transform,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_records = [records[int(idx)] for idx in train_idx]
    valid_records = [records[int(idx)] for idx in valid_idx]
    train_loader = DataLoader(
        FewShotDataset(train_records, label_to_idx, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    valid_loader = DataLoader(
        FewShotDataset(valid_records, label_to_idx, valid_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, valid_loader


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for images, _ in loader:
        logits = model(images.to(device))
        preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return np.asarray(preds, dtype=np.int64)


@torch.no_grad()
def predict_records_tta(
    model: nn.Module,
    records: list[ImageRecord],
    record_indices: np.ndarray,
    preprocess,
    aug_transform,
    batch_size: int,
    device: torch.device,
    view_seed: int,
    tta_views: int,
) -> np.ndarray:
    model.eval()
    all_logits = []
    for view_idx in range(tta_views):
        view_logits = []
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            batch_indices = record_indices[start : start + batch_size]
            tensors = []
            for local_idx, record in enumerate(batch_records):
                image = load_rgb_image(record.path)
                if view_idx == 0:
                    tensors.append(preprocess(image))
                else:
                    image_seed = view_seed + view_idx * 100_003 + int(batch_indices[local_idx]) * 17
                    tensors.append(deterministic_aug_image(image, aug_transform, image_seed))
            logits = model(torch.stack(tensors).to(device))
            view_logits.append(logits.detach().cpu())
        all_logits.append(torch.cat(view_logits, dim=0))
    mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    return mean_logits.argmax(dim=1).numpy().astype(np.int64)


def init_head_from_sklearn(
    classifier: OpenCLIPClassifier,
    features_by_view: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    config: FinetuneConfig,
    seed: int,
) -> None:
    train_features = np.vstack([features_by_view[: config.init_train_views, idx, :] for idx in train_idx])
    train_labels = np.asarray([labels[idx] for idx in train_idx for _ in range(config.init_train_views)])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.init_c,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )
    sample_weight = np.ones(len(train_labels), dtype=np.float32)
    sample_weight[train_labels == "Class_0"] = config.class0_weight
    sample_weight[train_labels == "Class_1"] = config.class1_weight
    sample_weight[train_labels == "Class_2"] = config.class2_weight
    sample_weight[train_labels == "Class_3"] = config.class3_weight
    sample_weight[train_labels == "Class_4"] = config.class4_weight
    model.fit(train_features, train_labels, logisticregression__sample_weight=sample_weight)
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    coef = logistic.coef_ / scaler.scale_[None, :]
    bias = logistic.intercept_ - (logistic.coef_ * scaler.mean_[None, :] / scaler.scale_[None, :]).sum(axis=1)
    with torch.no_grad():
        classifier.head.weight.copy_(torch.tensor(coef, dtype=classifier.head.weight.dtype, device=classifier.head.weight.device))
        classifier.head.bias.copy_(torch.tensor(bias, dtype=classifier.head.bias.dtype, device=classifier.head.bias.device))


def train_one_fold(
    records: list[ImageRecord],
    labels: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    config: FinetuneConfig,
    model_name: str,
    pretrained: str,
    cache_dir: str,
    device: torch.device,
    seed: int,
    init_features_by_view: Optional[np.ndarray] = None,
    view_seed: int = 42,
) -> tuple[np.ndarray, dict[str, float], dict[str, torch.Tensor]]:
    import open_clip

    seed_everything(seed)
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
    model = OpenCLIPClassifier(
        base_model,
        feature_dim,
        len(CLASSES),
        config.head_type,
        config.adapter_dim,
        config.adapter_scale,
        config.adapter_dropout,
    ).to(device)
    trainable_params, total_params = count_trainable_parameters(model)
    if config.init_sklearn:
        if init_features_by_view is None:
            raise ValueError("init_sklearn=True 需要传入 init_features_by_view。")
        init_head_from_sklearn(model, init_features_by_view, labels, train_idx, config, seed)

    train_transform = build_train_transform(
        preprocess,
        config.crop_scale_min,
        config.jitter,
        config.rotation_mode,
        config.rotation_degrees,
    )
    eval_aug_transform = build_eval_aug_transform(
        preprocess,
        config.crop_scale_min,
        config.jitter,
        config.rotation_mode,
        config.rotation_degrees,
    )
    valid_transform = preprocess
    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    train_loader, valid_loader = make_loaders(
        records,
        train_idx,
        valid_idx,
        label_to_idx,
        train_transform,
        valid_transform,
        config.batch_size,
        seed,
    )
    valid_records = [records[int(idx)] for idx in valid_idx]
    valid_global_indices = np.asarray(valid_idx, dtype=np.int64)

    class_weights = torch.ones(len(CLASSES), dtype=torch.float32, device=device)
    class_weights[label_to_idx["Class_0"]] = config.class0_weight
    class_weights[label_to_idx["Class_1"]] = config.class1_weight
    class_weights[label_to_idx["Class_2"]] = config.class2_weight
    class_weights[label_to_idx["Class_3"]] = config.class3_weight
    class_weights[label_to_idx["Class_4"]] = config.class4_weight
    head_params = list(model.head.parameters())
    adapter_params = model.adapter_parameters()
    backbone_params = [param for name, param in model.openclip_model.named_parameters() if param.requires_grad]
    optimizer_groups = [
        {"params": backbone_params, "lr": config.lr_backbone},
        {"params": head_params, "lr": config.lr_head},
    ]
    if adapter_params:
        optimizer_groups.append({"params": adapter_params, "lr": config.lr_adapter})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=config.weight_decay)

    valid_true = labels[valid_idx]
    pred_idx = predict_records_tta(
        model,
        valid_records,
        valid_global_indices,
        preprocess,
        eval_aug_transform,
        config.batch_size,
        device,
        view_seed,
        config.eval_tta_views,
    )
    pred_labels = np.asarray([idx_to_label[int(idx)] for idx in pred_idx])
    best_score = f1_score(valid_true, pred_labels, average="macro")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = classification_loss(logits, targets.to(device), class_weights, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        pred_idx = predict_records_tta(
            model,
            valid_records,
            valid_global_indices,
            preprocess,
            eval_aug_transform,
            config.batch_size,
            device,
            view_seed,
            config.eval_tta_views,
        )
        pred_labels = np.asarray([idx_to_label[int(idx)] for idx in pred_idx])
        score = f1_score(valid_true, pred_labels, average="macro")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    model.load_state_dict(best_state)
    pred_idx = predict_records_tta(
        model,
        valid_records,
        valid_global_indices,
        preprocess,
        eval_aug_transform,
        config.batch_size,
        device,
        view_seed,
        config.eval_tta_views,
    )
    pred_labels = np.asarray([idx_to_label[int(idx)] for idx in pred_idx])
    metrics = {
        "best_epoch": float(best_epoch),
        "macro_f1": f1_score(valid_true, pred_labels, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(valid_true, pred_labels),
        "trainable_params": float(trainable_params),
        "total_params": float(total_params),
        "trainable_ratio": float(trainable_params / max(total_params, 1)),
    }
    return pred_labels, metrics, best_state


def evaluate_config(
    records: list[ImageRecord],
    labels: np.ndarray,
    config: FinetuneConfig,
    model_name: str,
    pretrained: str,
    cache_dir: str,
    device: torch.device,
    n_splits: int,
    seed: int,
    init_features_by_view: Optional[np.ndarray] = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out_of_fold = np.empty_like(labels, dtype=object)
    fold_ids = np.zeros(len(labels), dtype=np.int32)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        print(f"微调 fold {fold}/{n_splits}：{config}")
        pred, metrics, _ = train_one_fold(
            records,
            labels,
            train_idx,
            valid_idx,
            config,
            model_name,
            pretrained,
            cache_dir,
            device,
            seed + fold,
            init_features_by_view,
            seed,
        )
        out_of_fold[valid_idx] = pred
        fold_ids[valid_idx] = fold
        fold_rows.append({"fold": fold, **metrics})
        print(
            f"fold={fold} epoch={metrics['best_epoch']:.0f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        )

    recalls = recall_score(labels, out_of_fold, labels=CLASSES, average=None, zero_division=0)
    row: dict[str, object] = {
        **asdict(config),
        "macro_f1": f1_score(labels, out_of_fold, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(labels, out_of_fold),
        "fold_macro_f1_std": float(np.std([item["macro_f1"] for item in fold_rows])),
        "fold_balanced_accuracy_std": float(np.std([item["balanced_accuracy"] for item in fold_rows])),
        "fold_best_epoch_mean": float(np.mean([item["best_epoch"] for item in fold_rows])),
        "trainable_params": int(fold_rows[0]["trainable_params"]),
        "total_params": int(fold_rows[0]["total_params"]),
        "trainable_ratio": float(fold_rows[0]["trainable_ratio"]),
    }
    for label, value in zip(CLASSES, recalls):
        row[f"recall_{label}"] = float(value)
    return row, out_of_fold, fold_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCLIP ViT-B-32 轻量微调 5-fold 实验。")
    parser.add_argument("--train_dir", default="train_few_shot")
    parser.add_argument("--output_dir", default="outputs/openclip_finetune")
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--model_name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr_backbones", nargs="*", type=float, default=[1e-6])
    parser.add_argument("--lr_heads", nargs="*", type=float, default=[1e-3])
    parser.add_argument("--weight_decays", nargs="*", type=float, default=[0.05])
    parser.add_argument("--label_smoothings", nargs="*", type=float, default=[0.05])
    parser.add_argument("--class0_weights", nargs="*", type=float, default=[1.0])
    parser.add_argument("--class1_weights", nargs="*", type=float, default=[1.0])
    parser.add_argument("--class2_weights", nargs="*", type=float, default=[1.0])
    parser.add_argument("--class3_weights", nargs="*", type=float, default=[1.0])
    parser.add_argument("--class4_weights", nargs="*", type=float, default=[1.25])
    parser.add_argument("--crop_scale_min", type=float, default=0.72)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--rotation_modes", nargs="*", default=["none"], choices=["none", "small", "right_angle"])
    parser.add_argument("--rotation_degrees", nargs="*", type=float, default=[0.0])
    parser.add_argument("--unfreezes", nargs="*", default=["last_block"])
    parser.add_argument("--init_sklearn", action="store_true", help="用 sklearn LogisticRegression 初始化 torch head。")
    parser.add_argument("--init_c", type=float, default=0.08)
    parser.add_argument("--init_train_views", type=int, default=3)
    parser.add_argument("--eval_tta_views", type=int, default=3)
    parser.add_argument("--lora_ranks", nargs="*", type=int, default=[4])
    parser.add_argument("--lora_alphas", nargs="*", type=float, default=[8.0])
    parser.add_argument("--lora_dropouts", nargs="*", type=float, default=[0.0])
    parser.add_argument("--head_types", nargs="*", default=["linear"], choices=["linear", "ln_residual", "ln_gated_residual"])
    parser.add_argument("--adapter_dims", nargs="*", type=int, default=[24])
    parser.add_argument("--adapter_scales", nargs="*", type=float, default=[1.0])
    parser.add_argument("--adapter_dropouts", nargs="*", type=float, default=[0.0])
    parser.add_argument("--lr_adapters", nargs="*", type=float, default=[1e-3])
    parser.add_argument("--pairwise_margin_weights", nargs="*", type=float, default=[0.0])
    parser.add_argument("--pairwise_margins", nargs="*", type=float, default=[0.0])
    parser.add_argument("--pairwise_class_sets", nargs="*", default=["Class_2,Class_3,Class_4"])
    return parser.parse_args()


def append_experiment_log(out_dir: Path, best: dict) -> None:
    log_path = Path("docs/experiment_log.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        "### OpenCLIP 轻量微调/旋转增强补充实验",
        "",
        f"- 输出目录：`{out_dir}`",
        f"- 解冻策略：`{best['unfreeze']}`",
        f"- 可训练参数：`{int(best.get('trainable_params', 0))}` / `{int(best.get('total_params', 0))}`，比例 `{float(best.get('trainable_ratio', 0.0)):.2%}`",
        f"- 旋转增强：`{best['rotation_mode']}`，角度参数：`{best['rotation_degrees']}`",
        f"- lr_backbone：`{best['lr_backbone']}`，lr_head：`{best['lr_head']}`",
        f"- head_type：`{best.get('head_type', 'linear')}`，adapter_dim：`{best.get('adapter_dim', 0)}`，adapter_scale：`{best.get('adapter_scale', 0.0)}`，lr_adapter：`{best.get('lr_adapter', 0.0)}`",
        f"- pairwise_margin_weight：`{best.get('pairwise_margin_weight', 0.0)}`，pairwise_margin：`{best.get('pairwise_margin', 0.0)}`，pairwise_class_set：`{best.get('pairwise_class_set', '')}`",
        f"- weight_decay：`{best['weight_decay']}`，label_smoothing：`{best['label_smoothing']}`",
        f"- 类别权重：Class_0=`{best.get('class0_weight', 1.0)}`，Class_1=`{best.get('class1_weight', 1.0)}`，Class_2=`{best['class2_weight']}`，Class_3=`{best['class3_weight']}`，Class_4=`{best['class4_weight']}`",
        f"- LoRA：rank=`{best.get('lora_rank', 0)}`，alpha=`{best.get('lora_alpha', 0.0)}`，dropout=`{best.get('lora_dropout', 0.0)}`",
        f"- macro-F1：`{float(best['macro_f1']):.4f}`",
        f"- balanced accuracy：`{float(best['balanced_accuracy']):.4f}`",
        (
            "- 每类 recall："
            + "，".join(f"{label}={float(best[f'recall_{label}']):.2f}" for label in CLASSES)
        ),
        "",
        "结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)

    records = load_train_records(args.train_dir)
    validate_images(records)
    labels = np.asarray([record.label for record in records])
    init_features_by_view = None
    if args.init_sklearn:
        init_features_by_view = load_or_extract_views(
            records,
            args.model_name,
            args.pretrained,
            max(args.init_train_views, args.eval_tta_views),
            args.cache_dir,
            args.batch_size,
            device,
            args.seed,
            force=False,
            crop_scale_min=args.crop_scale_min,
            jitter=args.jitter,
        )
    rows = []
    predictions: dict[str, np.ndarray] = {}
    fold_maps: dict[str, np.ndarray] = {}

    grid = itertools.product(
        args.unfreezes,
        args.rotation_modes,
        args.rotation_degrees,
        args.lr_backbones,
        args.lr_heads,
        args.weight_decays,
        args.label_smoothings,
        args.class0_weights,
        args.class1_weights,
        args.class2_weights,
        args.class3_weights,
        args.class4_weights,
        args.lora_ranks,
        args.lora_alphas,
        args.lora_dropouts,
        args.head_types,
        args.adapter_dims,
        args.adapter_scales,
        args.adapter_dropouts,
        args.lr_adapters,
        args.pairwise_margin_weights,
        args.pairwise_margins,
        args.pairwise_class_sets,
    )
    for (
        unfreeze,
        rotation_mode,
        rotation_degrees,
        lr_backbone,
        lr_head,
        weight_decay,
        label_smoothing,
        class0_weight,
        class1_weight,
        class2_weight,
        class3_weight,
        class4_weight,
        lora_rank,
        lora_alpha,
        lora_dropout,
        head_type,
        adapter_dim,
        adapter_scale,
        adapter_dropout,
        lr_adapter,
        pairwise_margin_weight,
        pairwise_margin,
        pairwise_class_set,
    ) in grid:
        if rotation_mode == "none" and rotation_degrees != 0:
            continue
        if rotation_mode != "none" and rotation_degrees <= 0:
            continue
        if rotation_mode == "right_angle" and rotation_degrees != 90:
            continue
        if pairwise_margin_weight <= 0 and pairwise_margin != args.pairwise_margins[0]:
            continue
        if pairwise_margin_weight <= 0 and pairwise_class_set != args.pairwise_class_sets[0]:
            continue

        config = FinetuneConfig(
            lr_backbone=lr_backbone,
            lr_head=lr_head,
            weight_decay=weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            label_smoothing=label_smoothing,
            class0_weight=class0_weight,
            class1_weight=class1_weight,
            class2_weight=class2_weight,
            class3_weight=class3_weight,
            class4_weight=class4_weight,
            crop_scale_min=args.crop_scale_min,
            jitter=args.jitter,
            rotation_mode=rotation_mode,
            rotation_degrees=rotation_degrees,
            unfreeze=unfreeze,
            init_sklearn=args.init_sklearn,
            init_c=args.init_c,
            init_train_views=args.init_train_views,
            eval_tta_views=args.eval_tta_views,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            head_type=head_type,
            adapter_dim=adapter_dim,
            adapter_scale=adapter_scale,
            adapter_dropout=adapter_dropout,
            lr_adapter=lr_adapter,
            pairwise_margin_weight=pairwise_margin_weight,
            pairwise_margin=pairwise_margin,
            pairwise_class_set=pairwise_class_set,
        )
        key = json.dumps(asdict(config), sort_keys=True)
        row, pred, fold_ids = evaluate_config(
            records,
            labels,
            config,
            args.model_name,
            args.pretrained,
            args.cache_dir,
            device,
            args.n_splits,
            args.seed,
            init_features_by_view,
        )
        row.update({"model_name": args.model_name, "pretrained": args.pretrained})
        rows.append(row)
        predictions[key] = pred
        fold_maps[key] = fold_ids
        pd.DataFrame(rows).to_csv(out_dir / "openclip_finetune_sweep_partial.csv", index=False)
        print(
            f"config done macro_f1={row['macro_f1']:.4f} "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "openclip_finetune_sweep.csv", index=False)
    best = results.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0].to_dict()
    best_config = FinetuneConfig(
        lr_backbone=float(best["lr_backbone"]),
        lr_head=float(best["lr_head"]),
        weight_decay=float(best["weight_decay"]),
        epochs=int(best["epochs"]),
        patience=int(best["patience"]),
        batch_size=int(best["batch_size"]),
        label_smoothing=float(best["label_smoothing"]),
        class0_weight=float(best.get("class0_weight", 1.0)),
        class1_weight=float(best.get("class1_weight", 1.0)),
        class2_weight=float(best["class2_weight"]),
        class3_weight=float(best["class3_weight"]),
        class4_weight=float(best["class4_weight"]),
        crop_scale_min=float(best["crop_scale_min"]),
        jitter=float(best["jitter"]),
        rotation_mode=str(best["rotation_mode"]),
        rotation_degrees=float(best["rotation_degrees"]),
        unfreeze=str(best["unfreeze"]),
        init_sklearn=bool(best["init_sklearn"]),
        init_c=float(best["init_c"]),
        init_train_views=int(best["init_train_views"]),
        eval_tta_views=int(best["eval_tta_views"]),
        lora_rank=int(best.get("lora_rank", 4)),
        lora_alpha=float(best.get("lora_alpha", 8.0)),
        lora_dropout=float(best.get("lora_dropout", 0.0)),
        head_type=str(best.get("head_type", "linear")),
        adapter_dim=int(best.get("adapter_dim", 24)),
        adapter_scale=float(best.get("adapter_scale", 1.0)),
        adapter_dropout=float(best.get("adapter_dropout", 0.0)),
        lr_adapter=float(best.get("lr_adapter", 1e-3)),
        pairwise_margin_weight=float(best.get("pairwise_margin_weight", 0.0)),
        pairwise_margin=float(best.get("pairwise_margin", 0.0)),
        pairwise_class_set=str(best.get("pairwise_class_set", "Class_2,Class_3,Class_4")),
    )
    best_key = json.dumps(asdict(best_config), sort_keys=True)
    with (out_dir / "best_openclip_finetune_config.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    pred = predictions[best_key]
    pd.DataFrame(confusion_matrix(labels, pred, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(
        out_dir / "best_confusion_matrix.csv"
    )
    oof_path = out_dir / "best_oof_predictions.csv"
    save_oof_predictions(records, labels, pred, fold_maps[best_key], oof_path)
    save_misclassified_sheets(oof_path, out_dir)
    append_experiment_log(out_dir, best)

    print("\nOpenCLIP 轻量微调当前最佳：")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
