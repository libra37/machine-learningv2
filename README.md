# ML-2 少样本图像分类最终方案

本文件夹是最终提交方案的最小可复现版本。

## 方法概述

最终采用单模型方案，不使用模型集成，不使用伪标签。

```text
32x32 RGB 图像
-> 弱数据增强 / TTA
-> OpenCLIP ViT-B-32/openai
-> 部分解冻微调：embedding + first3 transformer blocks + last6 transformer blocks
-> linear 分类头
-> Class_0 ~ Class_4
```

5-fold 交叉验证最佳结果：

```text
macro-F1 = 0.6987
balanced accuracy = 0.6960
```

最终提交文件：

```text
24123997.csv
```

该文件与 `outputs/final_epoch_29/submission_epoch_29.csv` 内容一致。

## 目录结构

```text
final_repro_openclip_partial_unfreeze/
  README.md
  requirements.txt
  run_final_predict.ps1
  final_config.json
  24123997.csv
  src/
    dataset.py
    openclip_aug_sweep.py
    openclip_finetune.py
    openclip_finetune_final_predict.py
  outputs/final_epoch_29/
    final_config_raw.json
    final_train_history.csv
    submission_epoch_29.csv
    test_predictions_with_confidence.csv
    test_pred_label_distribution.csv
  docs/
    experiment_log.md
```

## 环境

推荐使用课程实验环境：

```powershell
conda run -n control python ...
```

核心依赖：

```text
torch
torchvision
open_clip_torch
pandas
numpy
tqdm
Pillow
scikit-learn
```

## 数据放置

默认假设在本文件夹同级或当前目录下存在：

```text
train_few_shot/
  Class_0/
  Class_1/
  Class_2/
  Class_3/
  Class_4/

final_solution_openclip_adapter/test_shuffled/
```

如果测试集实际是嵌套目录：

```text
final_solution_openclip_adapter/test_shuffled/test_shuffled/
```

脚本会自动识别。

## 复现最终提交

在本文件夹的上一级项目根目录执行：

```powershell
cd /d "D:\机器学习2\期末"
.\final_repro_openclip_partial_unfreeze\run_final_predict.ps1
```

或直接运行：

```powershell
conda run -n control python final_repro_openclip_partial_unfreeze\src\openclip_finetune_final_predict.py `
  --train_dir train_few_shot `
  --test_dir final_solution_openclip_adapter\test_shuffled `
  --output_dir final_repro_openclip_partial_unfreeze\outputs\final_epoch_29 `
  --out final_repro_openclip_partial_unfreeze\24123997-马旋宇.csv `
  --epochs 29 `
  --unfreeze emb_first3_last6 `
  --lr_backbone 1e-5 `
  --lr_head 1e-3 `
  --weight_decay 0.10 `
  --label_smoothing 0.03 `
  --crop_scale_min 0.85 `
  --jitter 0.05 `
  --tta_views 3
```

## 说明

- `final_openclip_partial_unfreeze_model.pt` 没有放入本归档文件夹，因为文件约 600MB，不适合直接上传 GitHub。
- 复现时会重新从训练集训练最终模型，并生成 CSV。
- OpenCLIP 权重会由 `open_clip` 自动加载或从本地缓存读取。
- 若网络不可用，请提前准备 `OpenCLIP ViT-B-32/openai` 权重缓存。

