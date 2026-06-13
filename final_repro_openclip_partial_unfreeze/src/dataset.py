from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from PIL import Image


CLASSES = [f"Class_{idx}" for idx in range(5)]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    filename: str
    label: Optional[str] = None


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_train_dir(path: Union[str, Path]) -> Path:
    """解析训练集目录，兼容解压后多一层 train_few_shot 的常见结构。"""
    root = Path(path)
    if all((root / cls).is_dir() for cls in CLASSES):
        return root

    nested = root / "train_few_shot"
    if all((nested / cls).is_dir() for cls in CLASSES):
        return nested

    raise FileNotFoundError(
        f"在 {root} 或 {nested} 下找不到类别目录 {CLASSES}。"
    )


def load_train_records(
    path: Union[str, Path], expected_per_class: Optional[int] = 50
) -> list[ImageRecord]:
    train_dir = find_train_dir(path)
    records: list[ImageRecord] = []

    for label in CLASSES:
        class_dir = train_dir / label
        files = _image_files(class_dir)
        if expected_per_class is not None and len(files) != expected_per_class:
            raise ValueError(
                f"{class_dir} 中有 {len(files)} 张图片，期望 {expected_per_class} 张。"
            )
        records.extend(ImageRecord(path=file, filename=file.name, label=label) for file in files)

    if not records:
        raise ValueError(f"训练目录中没有找到图片：{train_dir}")
    return records


def load_test_records(path: Union[str, Path]) -> list[ImageRecord]:
    test_dir = Path(path)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"测试目录不存在：{test_dir}")

    files = _image_files(test_dir)
    if not files:
        raise ValueError(f"测试目录中没有找到图片：{test_dir}")
    return [ImageRecord(path=file, filename=file.name) for file in files]


def load_rgb_image(path: Union[str, Path]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def validate_images(records: Iterable[ImageRecord], expected_size: tuple[int, int] = (32, 32)) -> None:
    bad_size: list[str] = []
    unreadable: list[str] = []

    for record in records:
        try:
            image = load_rgb_image(record.path)
        except Exception:
            unreadable.append(str(record.path))
            continue

        if image.size != expected_size:
            bad_size.append(f"{record.path} has size {image.size}")

    if unreadable:
        raise ValueError("存在无法读取的图片：\n" + "\n".join(unreadable[:20]))
    if bad_size:
        raise ValueError("存在尺寸不符合预期的图片：\n" + "\n".join(bad_size[:20]))
