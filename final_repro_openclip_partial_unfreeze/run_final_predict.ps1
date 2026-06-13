$ErrorActionPreference = "Stop"

chcp 65001 | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

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

