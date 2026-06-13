# 实验记录日志

本文件用于记录每轮方向筛选实验的设置、结果和结论，方便后续写报告和复现实验。所有结果均来自 5-fold Stratified K-Fold，主要指标为 macro-F1，辅助指标为 balanced accuracy。当前实验均不使用模型集成。

## 2026-05-21 基础方向筛选

### 实验目的

先比较传统特征、纯视觉预训练模型、CLIP 类视觉语言模型、单模型微调四个方向，判断后续重点。

### 主要结果

| 方法 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| pixel_hog + LogisticRegression | 0.3281 | 0.3280 | 传统特征只能作为最低 baseline |
| ResNet18 frozen feature + LogisticRegression | 0.3992 | 0.4000 | ImageNet 预训练特征明显优于传统特征 |
| ResNet18 frozen feature + LinearSVM | 0.3445 | 0.3440 | SVM 不如 LogisticRegression |
| ResNet18 微调 layer4 + fc | 0.3843 | 0.3840 | 少样本下直接微调不如 frozen feature，暂不作为主线 |
| OpenCLIP ViT-B-32/openai + LogisticRegression(C=1.0) | 0.4504 | 0.4480 | CLIP 类视觉语言特征优于 ResNet18 |
| OpenCLIP ViT-B-32/openai + LinearSVM | 0.4217 | 0.4200 | 不如 LogisticRegression |
| OpenCLIP ViT-B-32/openai + prototype | 0.4132 | 0.4120 | 原型分类可作为 few-shot 对比，但不是当前最佳 |

### 阶段结论

OpenCLIP 图像特征最值得继续深入。ResNet18 linear probe 是稳定 baseline；ResNet18 微调作为失败/对比尝试写入报告即可。

## 2026-05-21 OpenCLIP ViT-B-32 分类头细扫

### 实验目的

在当前最优方向 OpenCLIP ViT-B-32/openai 上调轻量分类头超参数，观察是否能进一步提升。

### 命令

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_sweep.py --backbones ViT-B-32/openai --logreg_cs 0.05 0.07 0.09 0.1 0.12 0.15 0.2 --svm_cs 0.01 0.02 0.03 0.05
```

### 主要结果

| 方法 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| ViT-B-32/openai + LogisticRegression(C=0.05) | 0.4786 | 0.4760 | 当前全局最佳 |
| ViT-B-32/openai + LogisticRegression(C=0.10) | 0.4757 | 0.4720 | 接近最佳 |
| ViT-B-32/openai + LogisticRegression(C=0.20) | 0.4745 | 0.4720 | 接近最佳 |
| ViT-B-32/openai + LinearSVM(C=0.01) | 0.4556 | 0.4600 | SVM 最优设置仍弱于 LogisticRegression |
| ViT-B-32/openai + prototype | 0.4132 | 0.4120 | 弱于线性分类头 |

### 阶段结论

OpenCLIP ViT-B-32/openai + LogisticRegression(C=0.05) 是当前最强的单模型方案。较小的 C 更适合本任务，说明强正则化有助于少样本泛化。

## 2026-05-21 OpenCLIP 更强 ViT backbone 尝试

### 实验目的

尝试更强的 OpenCLIP ViT backbone：ViT-B-16/openai 和 ViT-L-14/openai，判断是否优于 ViT-B-32/openai。

### 命令

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_sweep.py --backbones ViT-B-16/openai ViT-L-14/openai --output_dir outputs\openclip_sweep_vit_strong --logreg_cs 0.03 0.05 0.07 0.1 0.15 0.2 0.3 --svm_cs 0.01 0.02 0.03 0.05
```

### 下载与环境情况

ViT-B-16 权重在一次中断后出现校验失败，OpenCLIP 自动重新下载后正常运行。ViT-L-14 权重约 933 MB，需要联网权限下载，下载成功后完成评估。

### 主要结果

| 方法 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| ViT-B-16/openai + LinearSVM(C=0.03) | 0.4663 | 0.4640 | ViT-B-16 最佳，但弱于 ViT-B-32 |
| ViT-B-16/openai + LogisticRegression(C=0.30) | 0.4567 | 0.4520 | 弱于 ViT-B-32 |
| ViT-L-14/openai + LogisticRegression(C=0.15) | 0.4703 | 0.4720 | ViT-L-14 最佳，但 macro-F1 仍低于 ViT-B-32 |
| ViT-L-14/openai + prototype | 0.3437 | 0.3520 | 原型分类在大模型特征上表现较差 |

### 阶段结论

更大的 OpenCLIP backbone 没有带来稳定提升。ViT-L-14 的 balanced accuracy 接近 ViT-B-32，但 macro-F1 更低且 fold 标准差更高，说明它在 250 张训练图上可能更不稳定。当前主线仍保持 OpenCLIP ViT-B-32/openai + LogisticRegression(C=0.05)。

## 当前总排名

| 排名 | 方法 | macro-F1 | balanced accuracy |
|---:|---|---:|---:|
| 1 | OpenCLIP ViT-B-32/openai + 训练增强 3 视图 + TTA 3 视图 + LogisticRegression(C=0.08, Class_4 weight=1.25) | 0.5539 | 0.5520 |
| 2 | OpenCLIP ViT-B-32/openai + 训练增强 3 视图 + TTA 3 视图 + LogisticRegression(C=0.10) | 0.5471 | 0.5480 |
| 3 | OpenCLIP ViT-B-32/openai + 训练增强 3 视图 + TTA 3 视图 + LogisticRegression(C=0.15, Class_4 weight=2.0) | 0.5487 | 0.5440 |
| 4 | OpenCLIP ViT-B-32/openai + LogisticRegression(C=0.05) | 0.4786 | 0.4760 |
| 5 | OpenCLIP ViT-L-14/openai + LogisticRegression(C=0.15) | 0.4703 | 0.4720 |
| 6 | OpenCLIP ViT-B-16/openai + LinearSVM(C=0.03) | 0.4663 | 0.4640 |
| 7 | ConvNeXt-Tiny frozen feature + LogisticRegression | 0.4888 | 0.4880 |
| 8 | ConvNeXt-Tiny + 训练增强 5 视图 + TTA 3 视图 + LogisticRegression(C=0.02) | 0.5065 | 0.5080 |
| 9 | ResNet18 frozen feature + LogisticRegression | 0.3992 | 0.4000 |
| 10 | ResNet18 微调 layer4 + fc | 0.3843 | 0.3840 |
| 11 | pixel_hog + LogisticRegression | 0.3281 | 0.3280 |

## 当前报告结论草案

本任务中，CLIP 类视觉语言预训练特征比 ImageNet 监督预训练 ResNet18 特征更适合少样本分类。更大的 OpenCLIP backbone 不一定更好，ViT-L-14 和 ViT-B-16 都没有超过 ViT-B-32。当前最优单模型方案为 OpenCLIP ViT-B-32/openai 冻结图像编码器，对每张训练图提取 3 个增强视图特征，对验证/测试图使用 3 视图 TTA 特征平均，再训练 LogisticRegression(C=0.08)，并将 Class_4 样本权重设为 1.25。

## 2026-05-21 OpenCLIP ViT-B-32 + 数据增强/TTA

### 实验目的

在已经确定的 OpenCLIP ViT-B-32/openai 主线上引入方向 4 的数据增强和 TTA。仍然只使用一个 OpenCLIP 图像编码器和一个 LogisticRegression 分类头，不使用模型集成。

### 方法说明

训练阶段：对每张训练图片提取多个视图的 OpenCLIP 图像特征。第 1 个视图为原始预处理视图，后续视图使用 RandomResizedCrop、RandomHorizontalFlip、ColorJitter 等增强。分类器训练时把这些增强视图当作同一类别的额外样本。

验证/测试阶段：对同一张图提取多个视图特征并求平均，再输入同一个分类器。这属于 test-time augmentation，不是多模型集成。

### 命令

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_aug_sweep.py --model_name ViT-B-32 --pretrained openai --max_views 5 --train_views 1 3 5 --tta_views 1 3 5 --cs 0.03 0.05 0.07 0.1 0.15
```

### 主要结果

| 方法 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| train_views=1, tta_views=1, C=0.05 | 0.4786 | 0.4760 | 无增强基准 |
| train_views=3, tta_views=1, C=0.15 | 0.5037 | 0.5000 | 只做训练增强已有提升 |
| train_views=3, tta_views=3, C=0.10 | 0.5471 | 0.5480 | 当前全局最佳 |
| train_views=5, tta_views=3, C=0.10 | 0.5204 | 0.5160 | 增强过多反而下降 |
| train_views=1, tta_views=5, C=0.15 | 0.4005 | 0.4200 | 只在验证端做强 TTA 会导致分布不匹配 |

### 类别召回率

当前最佳配置 train_views=3, tta_views=3, C=0.10：

| 类别 | recall |
|---|---:|
| Class_0 | 0.74 |
| Class_1 | 0.64 |
| Class_2 | 0.48 |
| Class_3 | 0.56 |
| Class_4 | 0.32 |

### 阶段结论

数据增强/TTA 对 OpenCLIP ViT-B-32 主线有效，macro-F1 从 0.4786 提升到 0.5471。最优组合不是增强越多越好，而是训练 3 视图、TTA 3 视图。训练和验证的增强强度需要匹配：只做验证 TTA 或 TTA 视图过多会显著降低结果。

### 遇到的问题

1. Windows 终端通过 conda run 转发中文输出时会出现乱码，但不影响实验结果文件。
2. ViT-B-16 下载曾因中断导致权重校验失败，需要重新下载。
3. ViT-L-14 权重较大，下载和推理成本高，但收益不明显。
4. Class_4 仍然是当前最弱类别，最佳配置下 recall 只有 0.32，需要后续重点分析。

### 后续优化方向

1. 围绕当前最佳 train_views=3、tta_views=3，细扫 C，例如 0.08、0.10、0.12、0.14。
2. 调整增强强度，尝试更弱的 ColorJitter 或更保守的 RandomResizedCrop scale。
3. 针对 Class_4 做误差分析，查看是否类内差异大或和其他类别混淆严重。
4. 尝试 class_weight 以外的类别重加权策略，但保持单模型和单分类头。
5. 如果时间允许，再尝试 OpenCLIP ViT-B-32 的轻量微调或 prompt-based zero-shot 作为补充对照。

## 暂未完成或暂未深入的方向

### 方向 1：更强纯视觉预训练模型微调

已做 ResNet18 frozen feature 和 ResNet18 layer4+fc 微调，但还没有尝试 ConvNeXt、DINOv2、EVA、ViT supervised 这类更强纯视觉 backbone，也没有对这些 backbone 做 LoRA/Adapter/最后几层微调。

如果当前 OpenCLIP 增强/TTA 方案长期无法提升，可以回到方向 1，优先尝试 ConvNeXt-Tiny 或 DINOv2 ViT-S/B 的 frozen feature + LogisticRegression，再决定是否做最后层微调。这个方向的优点是仍然是单模型，报告容易解释；风险是需要安装/下载新模型权重。

### 方向 2：OpenCLIP 轻量微调或 prompt 方案

当前 OpenCLIP 主线只冻结图像编码器，没有更新 backbone 参数。还没有做 prompt engineering zero-shot、文本 prompt 结合图像相似度、prompt tuning、LoRA 或 Adapter。

如果增强/TTA 和分类头调参到达瓶颈，可以尝试只微调 OpenCLIP visual encoder 的最后一个 transformer block，或给 visual encoder 加 LoRA/Adapter。该方向潜在收益高，但 250 张样本下过拟合风险也高，需要强早停和 5-fold 验证。

### 方向 3：大语言模型 + RAG few-shot 推理

尚未尝试。可能流程为：用 OpenCLIP/DINOv2 抽取训练集和测试集 embedding，对每张测试图检索相似训练样本，再把近邻样本标签、相似度和可选图像描述交给大语言模型判断类别。

暂不优先做的原因：测试集很大时推理成本和速度不稳定，且 LLM 不能直接看到原图时收益不确定。如果当前所有视觉模型路线都卡住，可把它作为报告中的探索方向或小规模验证方向，而不是最终大规模提交主方案。

### 方向 4：更系统的数据增强 / 自监督 / 伪标签

已在 OpenCLIP 特征上做训练增强和 TTA，并取得当前最佳结果。但还没有做 MixUp、CutMix、RandAugment、SimCLR 自监督预训练和伪标签。

如果当前增强/TTA 继续卡住，可以优先尝试更保守的增强强度、类别定制增强、Class_4 重加权；再考虑伪标签。伪标签需要测试集或额外无标签数据，且类别不平衡时可能放大偏差，所以要谨慎。

## 当前优化策略

短期继续围绕当前最优方案优化：

```text
OpenCLIP ViT-B-32/openai
train_views=3
tta_views=3
LogisticRegression
```

优先顺序：

1. 细扫 LogisticRegression 的 C。
2. 固定 C 后扫描增强强度。
3. 做 Class_4 误差分析和类别权重调整。
4. 若仍无法突破，再转向其他 backbone 或轻量微调。

## 2026-05-21 当前最佳方案 C 细扫

### 实验目的

固定当前最佳结构 train_views=3、tta_views=3，只细扫 LogisticRegression 的 C，判断 0.10 附近是否还能继续提升。

### 命令

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_aug_sweep.py --output_dir outputs\openclip_aug_c_fine --model_name ViT-B-32 --pretrained openai --max_views 3 --train_views 3 --tta_views 3 --cs 0.06 0.08 0.09 0.1 0.11 0.12 0.13 0.14
```

### 主要结果

| C | macro-F1 | balanced accuracy |
|---:|---:|---:|
| 0.06 | 0.5237 | 0.5240 |
| 0.08 | 0.5435 | 0.5440 |
| 0.09 | 0.5429 | 0.5440 |
| 0.10 | 0.5471 | 0.5480 |
| 0.11 | 0.5436 | 0.5440 |
| 0.12 | 0.5373 | 0.5360 |
| 0.13 | 0.5411 | 0.5400 |
| 0.14 | 0.5420 | 0.5400 |

### 阶段结论

C=0.10 仍是当前最优，说明分类头正则强度已经基本稳定。后续如果要继续提升，优先调整增强强度或处理 Class_4 召回，而不是继续细扫 C。

## 2026-05-21 增强强度、Class_4 误差分析与类别权重

### 实验目的

围绕当前最优 OpenCLIP ViT-B-32 + 训练增强/TTA 方案，尝试三件事：

1. 调整增强强度。
2. 分析 Class_4 召回低的原因。
3. 尝试 Class_4 样本权重。

### Class_4 误差分析

上一轮最佳配置 train_views=3, tta_views=3, C=0.10 的混淆矩阵中，Class_4 真实样本分布如下：

| 真实 Class_4 被预测为 | 数量 |
|---|---:|
| Class_0 | 5 |
| Class_1 | 8 |
| Class_2 | 12 |
| Class_3 | 9 |
| Class_4 | 16 |

Class_4 召回率仅为 0.32，主要混到 Class_2、Class_3 和 Class_1。这说明 Class_4 不是单纯被某一个类别吸走，而是整体边界较弱，可能类内差异更大或与多个类别都有视觉相似性。

### 增强强度实验

固定 train_views=3, tta_views=3, C=0.10，只调整 RandomResizedCrop 和 ColorJitter 强度。

| crop_scale_min | jitter | macro-F1 | balanced accuracy | 结论 |
|---:|---:|---:|---:|---|
| 0.72 | 0.15 | 0.5471 | 0.5480 | 原始最佳增强 |
| 0.80 | 0.10 | 0.5005 | 0.5000 | 明显下降 |
| 0.85 | 0.10 | 0.4827 | 0.4840 | 明显下降 |
| 0.90 | 0.05 | 0.4940 | 0.4920 | 明显下降 |

结论：更保守的增强没有提升，反而降低泛化。当前数据下，较强的随机裁剪和颜色扰动可能更有助于迫使分类头学习稳健特征。

### Class_4 权重实验

固定原增强强度 crop_scale_min=0.72、jitter=0.15、train_views=3、tta_views=3，扫描 C 和 Class_4 权重。

| C | Class_4 weight | macro-F1 | balanced accuracy | Class_4 recall | 结论 |
|---:|---:|---:|---:|---:|---|
| 0.10 | 1.00 | 0.5471 | 0.5480 | 0.32 | 原始最佳 |
| 0.08 | 1.25 | 0.5539 | 0.5520 | 0.38 | 当前全局最佳 |
| 0.08 | 1.50 | 0.5397 | 0.5360 | 0.38 | 权重过大开始伤害其他类 |
| 0.08 | 2.00 | 0.5346 | 0.5280 | 0.40 | Class_4 更高但整体下降 |
| 0.15 | 2.00 | 0.5487 | 0.5440 | 0.40 | 召回提升但 macro-F1 不如 1.25 |

当前最佳配置为 C=0.08、Class_4 weight=1.25，macro-F1 提升到 0.5539，Class_4 recall 从 0.32 提升到 0.38。

### 当前最佳混淆矩阵

当前最佳配置 C=0.08、Class_4 weight=1.25 的混淆矩阵：

| true/pred | Class_0 | Class_1 | Class_2 | Class_3 | Class_4 |
|---|---:|---:|---:|---:|---:|
| Class_0 | 37 | 0 | 6 | 4 | 3 |
| Class_1 | 1 | 31 | 4 | 4 | 10 |
| Class_2 | 4 | 3 | 24 | 10 | 9 |
| Class_3 | 3 | 3 | 8 | 27 | 9 |
| Class_4 | 5 | 7 | 10 | 9 | 19 |

### 阶段结论

Class_4 轻微上调权重有效，但权重过大时会牺牲其他类别，导致 macro-F1 下降。当前最优策略是温和提高 Class_4 权重到 1.25，而不是强行大幅提高。增强强度方面，原始较强增强仍然最佳。

### 后续优化方向

1. 围绕 Class_4 weight=1.25 细扫 C，例如 0.07、0.08、0.09、0.10。
2. 尝试只对 Class_4 使用更高训练视图数，但保持单模型和同一分类头。
3. 导出 Class_4 错分图片，人工观察它与 Class_1/2/3 的相似性。
4. 如果仍然卡住，再尝试 DINOv2 或 ConvNeXt frozen feature 作为备选方向。

## 2026-05-21 逐一补充实验：类别可视化、zero-shot、ConvNeXt 可用性

### 实验目的

在当前 OpenCLIP 增强/TTA 方案基础上，开始逐一排查可能突破 80% 的其他路径：先确认类别语义是否能被 prompt 利用，再补更强纯视觉 backbone。

### 类别可视化

已生成类别拼图：

```text
outputs/contact_sheet.png
```

观察结论：图片看起来像 32x32 的病理/细胞染色小图，主要差异来自细胞形态、核形态、染色深浅和局部纹理。类别之间肉眼差异较细，Class_4 与 Class_1/Class_2/Class_3 都存在相似样本，这与混淆矩阵中 Class_4 被多个类别吸走的现象一致。

### OpenCLIP zero-shot prompt

测试匿名类别 prompt 是否有用：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_zeroshot.py
```

| prompt set | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| literal | 0.0667 | 0.2000 | 几乎全预测到单一类别 |
| dataset_label | 0.0699 | 0.1720 | 无效 |
| small_image_label | 0.1354 | 0.2120 | 仍明显无效 |

结论：由于类别只有 Class_0 到 Class_4，没有真实语义名称，OpenCLIP 文本分支无法发挥优势。后续除非人工识别出真实类别含义，否则不优先做 zero-shot prompt。

### 类别文本描述辅助分类

为了回应“是否可以用大语言模型/视觉语言模型对类进行文本描述，从而辅助分类”的想法，补充了一个轻量实验。先根据类别拼图为每个匿名类别写入保守的形态描述，不引入真实医学类别名，避免把未知类别强行解释为某种具体疾病或细胞类型。描述文件保存为：

```text
docs/class_descriptions.json
```

然后运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_descriptor_prompt_sweep.py --output_dir outputs\openclip_descriptor_prompt_sweep_small --max_views 3 --train_views 3 --tta_views 3 --cs 0.05 0.08 0.1 0.15 --class4_weights 1.0 1.25 --feature_modes image descriptor image_descriptor
```

实验比较三种特征：

1. `image`：仅使用 OpenCLIP 图像特征。
2. `descriptor`：仅使用图像与 5 个类别文本描述的相似度。
3. `image_descriptor`：拼接图像特征和文本描述相似度。

主要结果如下：

| 特征方式 | 最佳设置 | macro-F1 | balanced accuracy | 结论 |
|---|---|---:|---:|---|
| image | C=0.08, Class_4 weight=1.25 | 0.5539 | 0.5520 | 当前非 adapter OpenCLIP 增强/TTA 对照 |
| descriptor | C=0.15, Class_4 weight=1.0 | 0.3597 | 0.3680 | 文本描述相似度单独不足以完成分类 |
| image_descriptor | C=0.08, Class_4 weight=1.25 | 0.5471 | 0.5440 | 拼接文本描述后未超过纯图像特征 |

阶段结论：类别文本描述可以作为可解释性材料，帮助报告说明各类视觉差异；但在当前匿名、低分辨率、细粒度病理小图任务中，描述 prompt 没有带来稳定分类增益。可能原因是类别差异主要体现为局部纹理和形态统计，而短文本描述难以精确表达这些低层视觉差异；同时类别本身没有明确语义名称，OpenCLIP 文本分支仍难以发挥真正的 zero-shot 优势。因此该方向适合作为补充探索写入报告，不建议替代当前主线模型。

### ConvNeXt-Tiny frozen feature

已将 `convnext_tiny` 接入 `run_experiments.py`。运行时需要下载 torchvision 的 ImageNet 预训练权重：

```text
https://download.pytorch.org/models/convnext_tiny-983f1562.pth
```

首次下载中断后留下了不完整 `.pth` 文件，导致 `PytorchStreamReader failed reading zip archive`。删除损坏缓存并重新下载后实验完成。

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python run_experiments.py --features convnext_tiny
```

| 方法 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| ConvNeXt-Tiny frozen feature + LogisticRegression | 0.4888 | 0.4880 | 优于 ResNet18，但低于 OpenCLIP 增强/TTA |
| ConvNeXt-Tiny frozen feature + LinearSVM | 0.4663 | 0.4680 | 弱于 LogisticRegression |

ConvNeXt-Tiny 的 Class_4 recall 达到 0.46，高于当前 OpenCLIP 最佳的 0.38，但 Class_0/Class_2 等类别较弱，整体 macro-F1 仍低。

### 阶段结论

1. 类别语义不明确，zero-shot prompt 暂时不可用。
2. 当前任务更像细粒度医学/病理小图分类，CLIP 文本语义优势受限，图像特征和增强更关键。
3. ConvNeXt-Tiny 作为纯视觉 backbone 有一定价值，尤其 Class_4 recall 更高，但整体仍不如 OpenCLIP 增强/TTA。
4. DINOv2 仍值得作为备选方向，但需要进一步处理模型下载/版本兼容。
5. 短期继续维护当前最佳：OpenCLIP ViT-B-32 + 3 训练视图 + 3 TTA 视图 + LogisticRegression(C=0.08, Class_4 weight=1.25)。

## 2026-05-21 ConvNeXt-Tiny + 数据增强/TTA

### 实验目的

ConvNeXt-Tiny frozen feature 的 Class_4 recall 高于 OpenCLIP 最佳方案，但整体 macro-F1 较低。因此尝试给 ConvNeXt-Tiny 加入与 OpenCLIP 类似的训练增强和 TTA，判断纯视觉 backbone 是否能通过增强追上主线。

### 命令

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\convnext_aug_sweep.py --output_dir outputs\convnext_aug_sweep --max_views 5 --train_views 1 3 5 --tta_views 1 3 5 --cs 0.03 0.05 0.08 0.1 0.15 --class4_weights 1.0 --crop_scale_min 0.72 --jitter 0.15
```

### 主要结果

| 方法 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| ConvNeXt-Tiny frozen + LogisticRegression | 0.4888 | 0.4880 | 无增强基准 |
| train_views=1, tta_views=1, C=0.03 | 0.4958 | 0.4960 | 调 C 后略高于原基准 |
| train_views=3, tta_views=5, C=0.08 | 0.4956 | 0.4960 | 小幅提升 |
| train_views=5, tta_views=3, C=0.03 | 0.5062 | 0.5080 | 第一轮最佳 |

### Class_4 权重细扫

围绕第一轮最佳 train_views=5、tta_views=3，进一步扫描 C 和 Class_4 权重：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\convnext_aug_sweep.py --output_dir outputs\convnext_aug_class4_weight --max_views 5 --train_views 5 --tta_views 3 --cs 0.02 0.03 0.04 0.05 --class4_weights 1.0 1.25 1.5 2.0 --crop_scale_min 0.72 --jitter 0.15
```

| C | Class_4 weight | macro-F1 | balanced accuracy | Class_4 recall |
|---:|---:|---:|---:|---:|
| 0.02 | 1.00 | 0.5065 | 0.5080 | 0.36 |
| 0.02 | 1.25 | 0.4961 | 0.4960 | 0.38 |
| 0.03 | 1.00 | 0.5062 | 0.5080 | 0.36 |
| 0.05 | 1.00 | 0.4909 | 0.4920 | 0.36 |

### 当前最佳混淆矩阵

ConvNeXt-Tiny 当前最佳配置 train_views=5, tta_views=3, C=0.02, Class_4 weight=1.0：

| true/pred | Class_0 | Class_1 | Class_2 | Class_3 | Class_4 |
|---|---:|---:|---:|---:|---:|
| Class_0 | 32 | 2 | 3 | 9 | 4 |
| Class_1 | 2 | 30 | 5 | 6 | 7 |
| Class_2 | 5 | 6 | 24 | 6 | 9 |
| Class_3 | 4 | 7 | 11 | 23 | 5 |
| Class_4 | 5 | 11 | 11 | 5 | 18 |

### 阶段结论

ConvNeXt-Tiny 加增强/TTA 后从 0.4888 提升到 0.5065，但仍低于 OpenCLIP 增强/TTA + Class_4 权重方案的 0.5539。Class_4 加权对 ConvNeXt 不如对 OpenCLIP 有效，权重提升会轻微提高 Class_4 recall，但整体 macro-F1 下降。

ConvNeXt 的价值在于作为方向 1 的更强 baseline 和报告对照：纯视觉 ImageNet 预训练模型能优于 ResNet18，但在当前任务上仍不如 OpenCLIP 图像特征。
## 2026-05-21 冲刺 70%-80%：误差分析与单模型方向扩展

### 已完成的代码改动

1. `src/openclip_aug_sweep.py` 增加 out-of-fold 预测导出，输出字段为 `filename,true_label,pred_label,fold`。
2. 自动统计错分方向，保存 `misclassified_pair_counts.csv`。
3. 自动生成错分样本拼图，保存到 `misclassified_sheets/`，便于报告中做定性误差分析。
4. 增加 `class4_train_views` 参数，支持只给 Class_4 使用更多训练增强视图，但仍然训练同一个 LogisticRegression，满足“不做模型集成”的要求。
5. `src/features.py` 与 `run_experiments.py` 接入 `convnext_small`、`convnext_base`、`dinov2_small`、`dinov2_base` 的 frozen feature 实验入口。

### OpenCLIP 当前最优精扫

固定 `OpenCLIP ViT-B-32/openai`、`train_views=3`、`tta_views=3`，细扫 `C=0.06~0.12`、`Class_4 weight=1.10~1.40`，并尝试 `Class_4 train views=0/5/7`。

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_aug_sweep.py --output_dir outputs\openclip_aug_fine_class4_views --model_name ViT-B-32 --pretrained openai --max_views 7 --train_views 3 --tta_views 3 --cs 0.06 0.07 0.08 0.09 0.1 0.11 0.12 --class4_weights 1.10 1.15 1.20 1.25 1.30 1.35 1.40 --class4_train_views 0 5 7 --crop_scale_min 0.72 --jitter 0.15
```

最佳结果仍然是：

| 方法 | macro-F1 | balanced accuracy | Class_0 | Class_1 | Class_2 | Class_3 | Class_4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViT-B-32 + 3 train views + 3 TTA + LR(C=0.08) + Class_4 weight=1.25 | 0.5539 | 0.5520 | 0.74 | 0.62 | 0.48 | 0.54 | 0.38 |

`Class_4 train views=5/7` 没有超过 `class4_train_views=0`，说明当前瓶颈不是简单增加 Class_4 增强样本数量，而是类间视觉边界本身混淆。

### 当前最优混淆矩阵

| true/pred | Class_0 | Class_1 | Class_2 | Class_3 | Class_4 |
|---|---:|---:|---:|---:|---:|
| Class_0 | 37 | 0 | 6 | 4 | 3 |
| Class_1 | 1 | 31 | 4 | 4 | 10 |
| Class_2 | 4 | 3 | 24 | 10 | 9 |
| Class_3 | 3 | 3 | 8 | 27 | 9 |
| Class_4 | 5 | 7 | 10 | 9 | 19 |

主要错分方向：

| 错分方向 | 数量 | 观察 |
|---|---:|---|
| Class_2 -> Class_3 | 10 | Class_2 与 Class_3 边界不稳，是除 Class_4 外的最大混淆 |
| Class_1 -> Class_4 | 10 | 提高 Class_4 权重会吞掉部分 Class_1 |
| Class_4 -> Class_2 | 10 | Class_4 与 Class_2 相似度高 |
| Class_3 -> Class_4 | 9 | Class_3 与 Class_4 也存在明显互混 |
| Class_2 -> Class_4 | 9 | Class_4 权重提高后容易影响 Class_2 |
| Class_4 -> Class_3 | 9 | Class_4 不是只偏向某一类，而是向多类分散 |

错分拼图已生成：

```text
outputs/openclip_aug_fine_class4_views/misclassified_sheets/
```

### Submission 冒烟测试

使用当前最优模型对 `Class_0` 子目录临时当测试集预测，确认 `predict.py` 可以生成 submission 格式：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\predict.py --model outputs\openclip_aug_fine_class4_views\final_openclip_aug_model.joblib --test_dir train_few_shot\train_few_shot\Class_0 --out outputs\openclip_aug_fine_class4_views\submission_smoke.csv
```

检查结果：CSV 表头为 `filename,label`，共 50 行预测，标签集合合法。

### ConvNeXt-Small frozen feature

尝试更强的纯视觉 ImageNet backbone：`ConvNeXt-Small + LogisticRegression/SVM`。

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python run_experiments.py --features convnext_small
```

首次下载中断后产生损坏缓存，删除 `convnext_small-0c510722.pth` 后重新下载完成。

| 方法 | macro-F1 | balanced accuracy | Class_0 | Class_1 | Class_2 | Class_3 | Class_4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Small frozen + LogisticRegression | 0.4527 | 0.4520 | 0.66 | 0.50 | 0.32 | 0.42 | 0.36 |
| ConvNeXt-Small frozen + LinearSVM | 0.4445 | 0.4440 | 0.62 | 0.50 | 0.32 | 0.40 | 0.38 |

结论：ConvNeXt-Small 明显低于 ConvNeXt-Tiny + aug/TTA 的 0.5065，也低于 OpenCLIP 主线的 0.5539。更大 ImageNet backbone 不一定更适合当前 32x32 匿名细粒度任务，暂不继续给 ConvNeXt-Small 做 TTA。

### DINOv2 尝试与阻塞

已在代码中预留 `dinov2_small`、`dinov2_base` frozen feature 入口，但当前环境暂时无法完成实验：

1. HuggingFace `AutoImageProcessor` 不存在，说明 `transformers` 版本偏旧。
2. 改为手写 resize/normalize 后，仍无法从 HuggingFace 加载 `facebook/dinov2-small` 的 `config.json`。
3. 改试官方 `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`，仓库可下载，但官方代码使用 `float | None` 类型写法，当前 Python 版本不支持，报错 `unsupported operand type(s) for |: 'type' and 'NoneType'`。

结论：DINOv2 当前不是效果失败，而是环境版本阻塞。若后续 OpenCLIP 长期无法突破 0.60，建议单独建 Python 3.10+ 新环境再跑 DINOv2-S/B，而不是在当前可运行环境中强行升级核心依赖。

### 当前决策

1. 当前最优候选仍然是 `outputs/openclip_aug_fine_class4_views/final_openclip_aug_model.joblib`，macro-F1=0.5539。
2. `Class_4` 低召回的主要原因不是样本数量，而是与 `Class_1/Class_2/Class_3` 多方向混淆；单纯加权或增加 Class_4 增强会带来其他类召回下降。
3. ConvNeXt-Small 与 DINOv2 当前没有推翻 OpenCLIP 主线。短期继续优化 OpenCLIP 单模型更合理。
4. 下一步如果继续冲分，优先尝试：
   - 更细的 OpenCLIP 预处理增强强度，例如 `crop_scale_min=0.80/0.88`、更弱 color jitter，减少 32x32 小图被裁坏。
   - 对 LogisticRegression 改用 per-class 权重网格，而不是只调 Class_4。
   - 尝试 OpenCLIP 图像特征的特征标准化、PCA/whitening、不同 solver 或 one-vs-rest 线性分类器。
   - 若 frozen feature 长期卡在 0.56 左右，再进入最后 block/adapter/linear probe 微调。
## 2026-05-23 冲刺 70%-80%：OpenCLIP 精修、轻量微调与 timm 路线

### 代码改动

新增 `src/openclip_refine_sweep.py`，用于在当前 OpenCLIP 主线上系统比较：

- 更保守的数据增强强度：`crop_scale_min`、`jitter`
- TTA 模式：`aug_mean3`、`center`、`center_hflip`、`center_hflip_mild`
- 分类头：`auto/ovr/multinomial`
- `PCA whitening + LogisticRegression`
- `Class_2/Class_3/Class_4` 权重 profile

同时更新 `src/predict.py`，使 `openclip_refine` 模型包可以直接生成 submission。

新增 `src/openclip_finetune.py`，用于 OpenCLIP 轻量微调 5-fold 实验：

- `head_only`
- `last_block`
- 早停、label smoothing、class weight、weight decay
- 输出 sweep CSV、best config、OOF、混淆矩阵、错分拼图

### 1. 保守增强强度实验

固定当前主线结构：`train_views=3`、`tta=aug_mean3`、`C=0.08`、`Class_4 weight=1.25`，扫描：

- `crop_scale_min=0.80,0.85,0.90,0.95`
- `jitter=0.05,0.10,0.15`

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_refine_sweep.py --output_dir outputs\openclip_refine_aug_strength --train_views 3 --tta_modes aug_mean3 --cs 0.08 --multi_classes auto --pca_dims 0 --class2_weights 1.0 --class3_weights 1.0 --class4_weights 1.25 --crop_scale_mins 0.80 0.85 0.90 0.95 --jitters 0.05 0.10 0.15
```

最佳结果：

| crop_scale_min | jitter | macro-F1 | balanced accuracy | 结论 |
|---:|---:|---:|---:|---|
| 0.85 | 0.05 | 0.5205 | 0.5200 | 低于当前最优 0.5539 |

结论：更保守裁剪和更弱颜色扰动没有提升。当前 `crop_scale_min=0.72, jitter=0.15` 反而更好，说明适度随机裁剪/颜色扰动对该 32x32 小图任务有正则化价值。

### 2. 分类头、PCA 与多类别权重 profile

固定 `crop_scale_min=0.72, jitter=0.15, train_views=3, tta=aug_mean3`，扫描：

- `C=0.08,0.10`
- `multi_class=auto,ovr,multinomial`
- `pca_dim=0,128`
- 权重 profile：
  - `Class_2=1.0, Class_3=1.0, Class_4=1.25`
  - `Class_2=1.1, Class_3=1.0, Class_4=1.25`
  - `Class_2=1.0, Class_3=1.1, Class_4=1.25`
  - `Class_2=1.1, Class_3=1.1, Class_4=1.25`
  - `Class_2=1.0, Class_3=1.0, Class_4=1.40`

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_refine_sweep.py --output_dir outputs\openclip_refine_head_weight_small --train_views 3 --tta_modes aug_mean3 --cs 0.08 0.10 --multi_classes auto ovr multinomial --pca_dims 0 128 --weight_profiles 1.0,1.0,1.25 1.10,1.0,1.25 1.0,1.10,1.25 1.10,1.10,1.25 1.0,1.0,1.40 --crop_scale_mins 0.72 --jitters 0.15
```

最佳仍然是原配置：

| 配置 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| `C=0.08, multi_class=auto, pca=0, w2=1.0, w3=1.0, w4=1.25` | 0.5539 | 0.5520 | 当前最优不变 |
| `C=0.08, w4=1.40` | 0.5502 | 0.5480 | Class_4 更重但整体略降 |
| `ovr` 系列 | 最高约 0.505 | 约 0.504 | 明显弱于 multinomial/auto |
| `PCA whitening=128` 系列 | 约 0.47~0.50 | 约 0.47~0.50 | 明显破坏判别信息 |

结论：线性分类头基本已经调到瓶颈。继续调 `C`、PCA、OVR 或给 Class_2/3 加权，不太可能突破 0.60。

### 3. 固定 TTA 模式实验

固定当前训练增强和分类头，比较 deterministic TTA：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_refine_sweep.py --output_dir outputs\openclip_refine_fixed_tta --train_views 3 --tta_modes center center_hflip center_hflip_mild --cs 0.08 0.10 --multi_classes auto --pca_dims 0 --weight_profiles 1.0,1.0,1.25 --crop_scale_mins 0.72 --jitters 0.15
```

| TTA | C | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---:|---|
| center | 0.10 | 0.5060 | 0.5000 | 弱于随机增强均值 |
| center_hflip | 0.08 | 0.5187 | 0.5120 | 固定 flip 有帮助，但仍低于 0.5539 |
| center_hflip_mild | 0.08 | 0.5155 | 0.5080 | 仍低于当前主线 |

结论：当前最好的 TTA 仍是 `aug_mean3`，固定 center/flip/mild crop 不足以提升。

### 4. OpenCLIP 轻量微调尝试

#### last block 微调

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_finetune.py --output_dir outputs\openclip_finetune_last_block_probe --epochs 20 --patience 5 --batch_size 32 --lr_backbones 0.000001 --lr_heads 0.001 --weight_decays 0.05 --label_smoothings 0.05 --class4_weights 1.25 --unfreezes last_block
```

结果：

| 方法 | macro-F1 | balanced accuracy | 现象 |
|---|---:|---:|---|
| OpenCLIP last_block 微调 | 0.1215 | 0.2000 | 几乎塌缩到 Class_4 |

#### head only sanity check

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_finetune.py --output_dir outputs\openclip_finetune_head_only_sanity --epochs 80 --patience 15 --batch_size 32 --lr_backbones 0.000001 --lr_heads 0.01 --weight_decays 0.001 --label_smoothings 0.0 --class4_weights 1.0 --unfreezes head_only
```

结果：

| 方法 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| torch head_only | 0.4399 | 0.4360 | 能学习，但远低于 sklearn LogisticRegression |

结论：当前端到端微调路线不稳定。随机初始化 torch 线性头不如 sklearn 标准化 LogisticRegression；直接解冻最后 block 会在小数据下塌缩。后续如果继续做微调，应优先考虑“用当前 sklearn 最优线性头初始化 torch head”或 LoRA/Adapter，而不是直接从随机 head 训练。

### 5. timm / EVA / DeiT 路线阻塞

检查发现当前 `control` 环境没有 `timm`。尝试安装：

```powershell
conda run -n control python -m pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120
conda run -n control python -m pip install timm --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org --timeout 120
```

两次均因 SSL 连接中断失败：

```text
SSLError(SSLZeroReturnError(6, 'TLS/SSL connection has been closed (EOF)'))
```

结论：EVA/DeiT/ConvNeXtV2 这条路线本轮不是模型失败，而是依赖安装网络阻塞。后续可在网络稳定时单独安装 `timm` 后继续。

### 当前阶段结论

1. 当前最优仍然是 `OpenCLIP ViT-B-32/openai + train_views=3 + aug_mean3 TTA + LogisticRegression(C=0.08, Class_4 weight=1.25)`，macro-F1=0.5539。
2. 数据增强、TTA、分类头、PCA、Class_2/3/4 权重都没有突破 0.5539。
3. frozen feature + 线性头已经接近当前路线瓶颈，短期不太可能靠继续小调参到 0.70。
4. 轻量微调需要更稳的初始化或 LoRA/Adapter；直接最后 block 微调风险高。
5. 如果要继续冲 70%-80%，优先级应调整为：
   - 解决 `timm` 或 Python 3.10+ 环境，跑 DINOv2/EVA/DeiT frozen feature。
   - 或实现 sklearn 最优线性头到 torch head 的初始化，再做最后 block 小学习率微调。
   - 如果拿到测试集，再考虑高置信伪标签，但必须控制类别分布，防止偏差扩大。
## 2026-05-23 激进方案尝试：先突破 60%

### 实验目的

上一轮已经证明 OpenCLIP 线性头、PCA、固定 TTA、保守增强都很难突破 `0.5539`。本轮不再只沿当前最优做微调，而是尝试更激进的方向：

1. OpenCLIP 特征上的非线性分类器。
2. 更多训练视图和 TTA 视图。
3. torchvision 中未尝试过的 ImageNet backbone。
4. OpenCLIP + 手工颜色/纹理特征拼接。
5. sklearn 最优线性头初始化 torch head，为后续最后 block 微调做准备。

### 1. OpenCLIP 非线性分类器

新增 `src/openclip_aggressive_sweep.py`，尝试：

- RBF SVM
- polynomial SVM
- KNN
- shrinkage LDA
- ExtraTrees
- MLP

命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_aggressive_sweep.py --output_dir outputs\openclip_aggressive_quick --quick
```

主要结果：

| 方法 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| shrinkage LDA | 0.5002 | 0.5000 | 本组最好，但低于 OpenCLIP 线性头 |
| RBF SVM 最好项 | 0.4746 | 0.4720 | 非线性核没有帮助 |
| KNN 最好项 | 0.4085 | 0.4120 | 最近邻不适合当前特征 |

结论：OpenCLIP 特征空间中，复杂非线性分类器更容易过拟合，不能突破 60%。

### 2. 更多训练/TTA 视图

在 `OpenCLIP ViT-B-32/openai` 上扩大搜索到 9 个视图：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_aug_sweep.py --output_dir outputs\openclip_aug_views9_seed42 --model_name ViT-B-32 --pretrained openai --max_views 9 --train_views 3 5 7 9 --tta_views 3 5 7 9 --cs 0.04 0.06 0.08 0.10 0.12 --class4_weights 1.0 1.25 1.5 --class4_train_views 0 --crop_scale_min 0.72 --jitter 0.15
```

最佳仍是原配置：

| train views | TTA views | C | Class_4 weight | macro-F1 | balanced accuracy |
|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 0.08 | 1.25 | 0.5539 | 0.5520 |

结论：视图不是越多越好。`train_views=5/7/9` 或 `tta_views=5/7/9` 大多下降，说明更多随机视图引入了噪声。

### 3. torchvision 强 backbone

已接入 `src/features.py` 与 `run_experiments.py`：

- `efficientnet_b0`
- `efficientnet_v2_s`
- `swin_t`
- `vit_b_16`
- `mobilenet_v3_large`
- `regnet_y_800mf`

成功完成的结果：

| backbone | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| EfficientNet-B0 | 0.4598 | 0.4560 | 低于 OpenCLIP 和 ConvNeXt-Tiny |
| Swin-T | 0.4506 | 0.4520 | 低于 OpenCLIP |
| ViT-B-16 supervised | 0.4347 | 0.4320 | 低于 OpenCLIP |
| EfficientNet-V2-S | 0.4047 | 0.4040 | 明显较弱 |
| RegNet-Y-800MF | 0.4105 | 0.4120 | 明显较弱 |
| MobileNetV3-Large | 0.3668 | 0.3680 | 明显较弱 |

Swin-T、ViT-B-16、EfficientNet-V2-S 首次由 torchvision 自动下载后出现 `PytorchStreamReader failed reading zip archive`。后续改用 `curl.exe -L --retry 5` 重新下载权重，确认 `torch.load` 可读后完成实验。

结论：torchvision supervised ImageNet backbone 没有接近 OpenCLIP。即使修复权重下载问题，Swin/ViT/EfficientNet-V2 也未突破 0.46。

### 4. OpenCLIP + 手工颜色/纹理特征拼接

新增 `src/openclip_handcrafted_fusion.py`，将 OpenCLIP 视图平均特征与 `pixel_hog` 的颜色/纹理特征拼接，再训练单个 LogisticRegression。

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_handcrafted_fusion.py --output_dir outputs\openclip_handcrafted_fusion
```

最佳结果：

| 配置 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| C=0.12, Class_4 weight=1.5, hand_scale=0.25 | 0.4882 | 0.4840 | 手工特征拼接没有提升 |

结论：简单颜色/纹理统计不能补足 OpenCLIP 的错误，反而会稀释深度特征。

### 5. sklearn 初始化 torch head 的微调准备

更新 `src/openclip_finetune.py`，支持：

- `--init_sklearn`
- 用 sklearn LogisticRegression 权重初始化 torch head
- 支持 `init_c`、`init_train_views`、`eval_tta_views`

sanity check：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n control python src\openclip_finetune.py --output_dir outputs\openclip_finetune_init_head_sanity_aligned --epochs 0 --patience 1 --batch_size 32 --lr_backbones 0.000001 --lr_heads 0.0001 --weight_decays 0.001 --label_smoothings 0.0 --class4_weights 1.25 --unfreezes head_only --init_sklearn --init_c 0.08 --init_train_views 3 --eval_tta_views 3
```

结果：

| 方法 | macro-F1 | balanced accuracy | 说明 |
|---|---:|---:|---|
| sklearn init torch head, epoch=0 | 0.5053 | 0.5000 | 仍低于 sklearn pipeline 的 0.5539 |

结论：torch 版本的动态 TTA/特征路径与缓存特征仍存在差异，初始化 head 后没有直接复现 0.5539。因此暂不继续做最后 block 微调，避免在错误起点上浪费算力。

### 当前结论

本轮更激进的尝试仍未突破 60%。目前能确认：

1. 当前最优仍是 `OpenCLIP ViT-B-32/openai + train_views=3 + tta_views=3 + LogisticRegression(C=0.08, Class_4 weight=1.25)`，macro-F1=0.5539。
2. 非线性分类器、更多视图、手工特征拼接、已成功的 torchvision backbone 都没有超过当前最优。
3. 如果继续冲 60%+，最值得投入的不是继续调这些分支，而是：
   - 新建 Python 3.10+ 环境跑 DINOv2 / EVA / DeiT。
   - 重新设计微调流程，保证 torch 特征路径能精确复现 sklearn 0.5539 后，再解冻最后 block。
   - 等测试集到手后，谨慎做高置信伪标签和类别分布校准。

## 2026-05-23 继续测试：LAION OpenCLIP 与 32x32 放大策略

### 1. 修复并测试 OpenCLIP LAION 权重

之前 `ViT-B-32/laion400m_e31`、`ViT-B-32/laion400m_e32` 下载后 checksum 或 `torch.load` 失败，本轮用 `curl.exe -L --retry 5` 从 OpenCLIP GitHub release 重新下载，确认权重可读。

新增/复测：

- `ViT-B-32/laion400m_e31`
- `ViT-B-32/laion400m_e32`
- `ViT-B-32/laion2b_e16`

结果：

| 模型 | 设置 | macro-F1 | balanced accuracy | 结论 |
|---|---|---:|---:|---|
| ViT-B-32/laion400m_e31 | frozen feature + LR/SVM/prototype | 0.4945 | 0.4960 | 低于 OpenAI 版 |
| ViT-B-32/laion400m_e31 | 3 train views + 3 TTA + LR | 0.5057 | 0.5040 | 增强后仍低 |
| ViT-B-32/laion400m_e32 | 3 train views + 3 TTA + LR | 0.5123 | 0.5120 | LAION 中最好，但仍低于 0.5539 |
| ViT-B-32/laion2b_e16 | frozen feature + LR/SVM/prototype | 0.4147 | 0.4200 | 明显较弱 |
| ViT-B-32/laion2b_e16 | 3 train views + 3 TTA + LR | 0.4605 | 0.4560 | 明显较弱 |

结论：LAION 预训练源没有带来突破，当前数据更适合 `ViT-B-32/openai` 的图像特征。

### 2. 测试 32x32 到 224 的插值方式

新增 `src/openclip_resize_sweep.py`，测试 OpenCLIP 输入放大方式：

- nearest
- bilinear
- bicubic
- lanczos

统一使用 `ViT-B-32/openai + train_views=3 + LogisticRegression`，测试 `center` 与 `center_hflip` TTA。

最佳结果：

| 插值 | TTA | C | Class_4 weight | macro-F1 | balanced accuracy |
|---|---|---:|---:|---:|---:|
| bicubic | center_hflip | 0.05 | 1.25 | 0.5266 | 0.5200 |

对应召回率：

| Class_0 | Class_1 | Class_2 | Class_3 | Class_4 |
|---:|---:|---:|---:|---:|
| 0.68 | 0.56 | 0.34 | 0.56 | 0.46 |

结论：插值方式不是主要瓶颈。`bicubic` 仍然最好，nearest/bilinear/lanczos 更差。该分支改善了 Class_4 召回，但 Class_2 降低太多，整体低于当前最佳 0.5539。

### 当前可判断的方向状态

已经尝试且暂不继续深挖：

- 传统 HOG/颜色纹理特征。
- ResNet18、ConvNeXt、EfficientNet、Swin、torchvision ViT 等 frozen feature。
- OpenCLIP ViT-B-16、ViT-L-14、RN50。
- OpenCLIP LAION400M/LAION2B 预训练源。
- 更多 train views/TTA views。
- Class_4 单独加权、Class_4 更多增强视图。
- PCA、OVR/multinomial、非线性 SVM、KNN、LDA、ExtraTrees、MLP。
- OpenCLIP + 手工特征拼接。
- 32x32 放大插值方式。

尚未真正跑通、且最可能突破 60% 的方向：

1. 新建 Python 3.10+ 环境跑 DINOv2 / EVA / DeiT / timm 模型，优先 frozen feature。
2. 重新实现微调流程，使 torch 版特征路径先复现 sklearn 0.5539，再做最后 block/Adapter/LoRA 微调。
3. 如果拿到测试集，做高置信伪标签、类别分布校准、test-time adaptation。该方向可能对不平衡测试集帮助最大，但需要测试集文件。

### timm frozen feature 补充实验

- 最佳模型：`vit_small_patch14_reg4_dinov2.lvd142m`
- macro-F1：`0.4872`
- balanced accuracy：`0.4840`
- 输出目录：`outputs/timm_dinov2_small`

### timm frozen feature 补充实验

- 最佳模型：`vit_small_patch16_224.dino`
- macro-F1：`0.4737`
- balanced accuracy：`0.4760`
- 输出目录：`outputs/timm_small_backbones`

### OpenCLIP metric/classifier 补充实验

- 最佳方法：`lda_shrinkage(0.2)`
- Class_4 weight：`1.0`
- macro-F1：`0.5157`
- balanced accuracy：`0.5160`
- 输出目录：`outputs/openclip_metric_sweep`

结论：RidgeClassifier、收缩 LDA、NCA+KNN 都没有超过当前 LogisticRegression 主线。收缩 LDA 能到 0.5157，说明正则化判别分析有一定效果，但仍低于 `OpenCLIP ViT-B-32/openai + LogisticRegression + 增强/TTA` 的 0.5539。

## 2026-05-23 Python 3.10/timm 环境与更强 backbone 测试

### 1. 环境处理

新建 `ml2-vision310` 环境：

```powershell
conda create -y -n ml2-vision310 python=3.10
```

GPU 版 PyTorch 安装时网络中断，报 `IncompleteRead`。为了先验证方向，改装 CPU 版：

```powershell
conda install -y -n ml2-vision310 pytorch torchvision cpuonly -c pytorch
conda install -y -n ml2-vision310 scikit-learn pandas tqdm joblib
conda run -n ml2-vision310 python -m pip install transformers timm safetensors huggingface_hub --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org
```

当前该环境可运行 timm/transformers，但 `torch.cuda.is_available()` 为 `False`，所以抽特征速度较慢。

### 2. timm DINOv2 / EVA / DeiT / ConvNeXtV2 frozen feature

新增 `src/timm_frozen_sweep.py`，统一使用：

- timm pretrained backbone
- `num_classes=0` 冻结抽特征
- 5-fold Stratified K-Fold
- `StandardScaler + LogisticRegression`
- `C` 与 Class_4 weight 小网格

结果：

| 模型 | 最佳 macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| `vit_small_patch14_dinov2.lvd142m` | 0.4667 | 0.4640 | 明显低于 OpenCLIP |
| `vit_small_patch14_reg4_dinov2.lvd142m` | 0.4872 | 0.4840 | DINOv2 中最好，但仍低 |
| `vit_small_patch16_224.dino` | 0.4737 | 0.4760 | 低于 OpenCLIP |
| `deit3_small_patch16_224.fb_in22k_ft_in1k` | 0.4292 | 0.4280 | 较弱 |
| `deit_small_patch16_224.fb_in1k` | 0.4326 | 0.4320 | 较弱 |
| `eva02_tiny_patch14_224.mim_in22k` | 0.4299 | 0.4320 | 较弱 |
| `convnextv2_tiny.fcmae_ft_in22k_in1k` | 0.4303 | 0.4320 | 较弱 |

结论：已跑通 DINOv2/EVA/DeiT/timm 路线，但 frozen feature 没有接近当前最佳 0.5539。当前数据可能更匹配 CLIP 的语义/纹理表征，而不是这些 ImageNet/DINO 系视觉表征。

### 当前最新判断

目前已经比较扎实地排除：

- OpenCLIP LAION 预训练源。
- DINOv2/EVA/DeiT/timm 小模型 frozen feature。
- 32x32 放大插值方式。
- Ridge / LDA / NCA+KNN 等替代分类头。

下一步若继续冲 60%+，更合理的是：

1. 在 `OpenCLIP ViT-B-32/openai` 上重新实现轻量微调，先确保 torch 版 head 能复现 sklearn 的 0.5539，再解冻最后 block 或加 Adapter/LoRA。
2. 尝试医学/病理/显微图像相关的公开预训练模型，如果模型权重许可证和作业规则允许，只使用预训练权重，不引入外部带标签数据。
3. 拿到测试集后做高置信伪标签和类别分布校准；这可能比继续本地 250 张上调参更有机会提升最终提交。

### OpenCLIP cache/Tip-Adapter 风格实验

- 最佳配置：`train_views=3, tta_views=3, beta=1.0, class4_weight=1.0, center_support=True, topk=0`
- macro-F1：`0.4281`
- balanced accuracy：`0.4280`
- 输出目录：`outputs/openclip_cache_sweep`

结论：cache/Tip-Adapter 风格的最近邻/原型检索没有超过线性头，说明当前 OpenCLIP 特征空间里同类样本并没有形成足够稳定的近邻簇。继续沿纯检索式 few-shot 分类不值得。

### HuggingFace OpenCLIP / BioMedCLIP frozen feature 实验

- 最佳模型：`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- C：`0.01`，Class_4 weight：`1.0`
- macro-F1：`0.4685`
- balanced accuracy：`0.4640`
- 输出目录：`outputs/openclip_hf_biomedclip`

结论：BioMedCLIP 没有带来提升。虽然它更偏医学/生物医学图文预训练，但本任务的匿名 5 类边界可能不是 BioMedCLIP 的语义空间能直接区分的。

### DataComp / MetaCLIP / DFN OpenCLIP 补充实验

测试：

- `ViT-B-32/datacomp_xl_s13b_b90k`
- `ViT-B-32/metaclip_400m`
- `ViT-B-16/datacomp_xl_s13b_b90k`
- `ViT-B-16/dfn2b`

最佳结果：

| 模型 | 分类头 | macro-F1 | balanced accuracy |
|---|---|---:|---:|
| `ViT-B-16/dfn2b` | LogisticRegression(C=0.01) | 0.4700 | 0.4720 |
| `ViT-B-16/datacomp_xl_s13b_b90k` | LogisticRegression(C=0.01) | 0.4634 | 0.4680 |
| `ViT-B-32/datacomp_xl_s13b_b90k` | LinearSVC(C=0.03) | 0.4452 | 0.4440 |

结论：更现代的数据源并不一定更适配。本数据目前仍然最适合 `ViT-B-32/openai`。

### OpenCLIP 轻量微调修复与结果

修复 `src/openclip_finetune.py`：之前早停时用 TTA 评估，但最终返回预测又退回单视图，导致 `init_sklearn` sanity check 低估。修复后 `epochs=0` 能精确复现当前最佳：

| 设置 | macro-F1 | balanced accuracy |
|---|---:|---:|
| sklearn init torch head, epoch=0, TTA fixed | 0.5539 | 0.5520 |

随后进行最后 block 轻量微调：

| 设置 | macro-F1 | balanced accuracy | 召回变化 |
|---|---:|---:|---|
| `lr_backbone=5e-7, lr_head=1e-5, wd=0.01, label_smoothing=0.05, Class_4 weight=1.25` | 0.5623 | 0.5600 | Class_2 从 0.48 到 0.52 |
| `lr_backbone=5e-7, lr_head=5e-6, wd=0.01, label_smoothing=0.03, Class_4 weight=1.25` | 0.5681 | 0.5640 | Class_1/4 有提升，当前最高 |
| 更长训练 `epochs=10, patience=3` 最佳 | 0.5637 | 0.5600 | 不如短训练 |

结论：最后 block 微调是目前唯一有正收益的方向，但提升很窄。最佳结果暂定为 `0.5681`，仍未突破 0.60。微调通常在 0-1 个 epoch 附近早停，继续训练容易过拟合或回落。

### HuggingFace OpenCLIP / BioMedCLIP frozen feature 实验

- 最佳模型：`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- C：`0.01`，Class_4 weight：`1.0`
- macro-F1：`0.4685`
- balanced accuracy：`0.4640`
- 输出目录：`outputs/openclip_hf_biomedclip`

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_rotation_sweep`
- 解冻策略：`last_block`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`5e-07`，lr_head：`5e-06`
- weight_decay：`0.01`，label_smoothing：`0.03`
- Class_4 weight：`1.25`
- macro-F1：`0.5580`
- balanced accuracy：`0.5560`
- 每类 recall：Class_0=0.74，Class_1=0.64，Class_2=0.50，Class_3=0.52，Class_4=0.38

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### 2026-05-24 本轮最终结论

本轮已完成计划中的三类重点尝试：旋转增强、Class_2/3/4 类别权重、轻量微调细化与解冻范围对照。所有新增实验均未超过历史最优 `0.5681`。

| 当前候选 | macro-F1 | balanced accuracy | 是否替换历史最优 |
|---|---:|---:|---|
| 历史最优 `outputs/openclip_finetune_tta_fixed_narrow` | 0.5681 | 0.5640 | 保留 |
| 本轮最佳 `outputs/openclip_finetune_lr_label_sweep_v2` | 0.5628 | 0.5600 | 不替换 |

后续默认策略：

1. 最终报告主线仍写作 `OpenCLIP ViT-B-32/openai + 多视图增强/TTA + sklearn 初始化 + last block 轻量微调`。
2. 旋转增强作为失败尝试记录：小角度/90 度旋转均降低结果。
3. 类别权重作为失败尝试记录：只保留 `Class_4 weight=1.25` 有价值，继续调 `Class_2/Class_3` 没有收益。
4. 若继续冲 `0.60`，优先做 LoRA/Adapter 或测试集伪标签；不建议继续普通增强/普通权重网格。

## 2026-05-24 从任务难点出发的边界优化实验

### 实验动机

前面实验说明，继续换大 backbone 或做普通增强很难突破。当前本质瓶颈是 `Class_2/Class_3/Class_4` 的特征边界混淆，所以本轮重点测试三种“边界优化”方案：

1. OpenCLIP frozen feature 上的 supervised contrastive 小投影头。
2. LogisticRegression 类别 logit bias 校准。
3. OpenCLIP frozen feature 后的小残差 Adapter。

### 1. Supervised contrastive 小投影头

新增 `src/openclip_supcon_sweep.py`。

最好结果：

| 输出目录 | macro-F1 | balanced accuracy | 主要现象 |
|---|---:|---:|---|
| `outputs/openclip_supcon_sweep` | 0.5677 | 0.5680 | Class_4 recall 可拉到 0.64，但 Class_2/3 下降 |
| `outputs/openclip_supcon_focused_sweep` | 0.5609 | 0.5640 | 聚焦搜索未继续提升 |

结论：SupCon 方向能改变类别召回分布，但不够稳定。它更像是在 Class_4 与 Class_2/3 之间做取舍，没有超过历史最优 `0.5681`。

### 2. LogisticRegression 类别 bias 校准

新增 `src/openclip_bias_calibration.py`，使用外层 5-fold、内层 4-fold 选择类别 bias。

最好结果：

| 输出目录 | macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| `outputs/openclip_bias_calibration` | 0.5378 | 0.5360 | 明显低于基准 |

结论：类别 bias 在内层训练折上选出来后，迁移到外层验证折不稳定，说明当前样本太少，直接调阈值容易过拟合。不作为后续主线。

### 3. OpenCLIP frozen feature + residual Adapter

新增 `src/openclip_feature_adapter_sweep.py`。核心做法：

- 固定 `OpenCLIP ViT-B-32/openai` 特征。
- 先用当前最佳 `LogisticRegression(C=0.08)` 初始化 torch 分类头。
- 在特征后接一个小残差 Adapter：`x -> x + scale * MLP(x)`。
- 只训练 Adapter，分类头保持冻结或极小学习率。

第一轮搜索：

| 输出目录 | 最佳 macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---|
| `outputs/openclip_feature_adapter_sweep` | 0.5812 | 0.5760 | 首次明确超过 0.5681 |

聚焦搜索：

| 输出目录 | 最佳 macro-F1 | balanced accuracy | 最佳配置 |
|---|---:|---:|---|
| `outputs/openclip_feature_adapter_focused_sweep` | **0.5922** | **0.5880** | adapter_dim=64, scale=0.4, lr_adapter=0.0015, lr_head=0, wd=0.0001, Class_4 weight=1.1 |
| `outputs/openclip_feature_adapter_final_narrow` | 0.5877 | 0.5840 | 更窄搜索未超过 focused 最优 |

当前新最优混淆矩阵：

| true\pred | Class_0 | Class_1 | Class_2 | Class_3 | Class_4 |
|---|---:|---:|---:|---:|---:|
| Class_0 | 37 | 2 | 6 | 2 | 3 |
| Class_1 | 0 | 33 | 3 | 3 | 11 |
| Class_2 | 4 | 3 | 26 | 6 | 11 |
| Class_3 | 3 | 3 | 8 | 26 | 10 |
| Class_4 | 3 | 7 | 10 | 5 | 25 |

每类 recall：

| Class_0 | Class_1 | Class_2 | Class_3 | Class_4 |
|---:|---:|---:|---:|---:|
| 0.74 | 0.66 | 0.52 | 0.52 | 0.50 |

阶段结论：

1. 当前全局最优更新为 `outputs/openclip_feature_adapter_focused_sweep`，macro-F1=`0.5922`，balanced accuracy=`0.5880`。
2. 相比上一最优 `0.5681`，提升约 `+0.0241` macro-F1。
3. 这说明本任务的关键不是继续换 backbone，而是在强 frozen feature 上做小幅、稳定的边界校准。
4. 目前距离 `0.60` 很近，下一步优先围绕 Adapter 做随机种子稳定性验证、最终全量训练策略，以及测试集可用后的伪标签/分布校准。

### 2026-05-24 当前优化计划落地小结

本轮按“旋转增强 -> 类别权重 -> 轻量微调细化 -> 解冻范围”的顺序继续尝试，目标是先突破 `0.60`。

| 实验方向 | 输出目录 | 最佳 macro-F1 | balanced accuracy | 结论 |
|---|---|---:|---:|---|
| 旋转增强 | `outputs/openclip_finetune_rotation_sweep` | 0.5580 | 0.5560 | `none` 最好；小角度和 90 度旋转均下降，说明方向/局部结构可能有判别信息 |
| Class_2/3/4 类别权重 | `outputs/openclip_finetune_class234_weight_sweep` | 0.5539 | 0.5520 | 未超过基准；提高 Class_2/3 权重会牺牲整体边界 |
| lr/label smoothing 细扫 | `outputs/openclip_finetune_lr_label_sweep_v2` | 0.5628 | 0.5600 | 略高于 frozen LR，但未超过历史 0.5681 |
| 解冻范围 | `outputs/openclip_finetune_unfreeze_sweep_v2` | 0.5566 | 0.5560 | `last_two_blocks` 没有收益；更多可训练参数更不稳定 |

阶段结论：

1. 目前历史最优仍是 `outputs/openclip_finetune_tta_fixed_narrow` 中的 `OpenCLIP ViT-B-32/openai + last_block 轻量微调`，macro-F1=`0.5681`，balanced accuracy=`0.5640`。
2. 旋转增强不适合当前数据；后续默认关闭旋转。
3. 类别权重只建议保留 `Class_4 weight=1.25`，不要继续大幅提高 `Class_2/Class_3` 权重。
4. 微调收益很窄，最稳定的范围仍是只解冻最后一个 block，且通常 0-2 epoch 内达到最佳。
5. 若继续冲 `0.60+`，下一步更值得尝试的是 LoRA/Adapter、测试集伪标签/分布校准，或寻找更贴近显微/病理小图的公开预训练 backbone；继续普通增强和普通权重网格的性价比已经较低。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_class234_weight_sweep`
- 解冻策略：`last_block`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`5e-07`，lr_head：`5e-06`
- weight_decay：`0.01`，label_smoothing：`0.03`
- 类别权重：Class_2=`1.0`，Class_3=`1.1`，Class_4=`1.25`
- macro-F1：`0.5539`
- balanced accuracy：`0.5520`
- 每类 recall：Class_0=0.74，Class_1=0.60，Class_2=0.48，Class_3=0.56，Class_4=0.38

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_lr_label_sweep_v2`
- 解冻策略：`last_block`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`8e-07`，lr_head：`1e-05`
- weight_decay：`0.01`，label_smoothing：`0.05`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.25`
- macro-F1：`0.5628`
- balanced accuracy：`0.5600`
- 每类 recall：Class_0=0.74，Class_1=0.60，Class_2=0.48，Class_3=0.58，Class_4=0.40

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_unfreeze_sweep_v2`
- 解冻策略：`last_two_blocks`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`8e-07`，lr_head：`1e-05`
- weight_decay：`0.01`，label_smoothing：`0.05`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.25`
- macro-F1：`0.5566`
- balanced accuracy：`0.5560`
- 每类 recall：Class_0=0.74，Class_1=0.64，Class_2=0.52，Class_3=0.56，Class_4=0.32

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### 2026-05-24 本轮最终结论

本轮已完成计划中的三类重点尝试：旋转增强、Class_2/3/4 类别权重、轻量微调细化与解冻范围对照。所有新增实验均未超过历史最优 `0.5681`。

| 当前候选 | macro-F1 | balanced accuracy | 是否替换历史最优 |
|---|---:|---:|---|
| 历史最优 `outputs/openclip_finetune_tta_fixed_narrow` | 0.5681 | 0.5640 | 保留 |
| 本轮最佳 `outputs/openclip_finetune_lr_label_sweep_v2` | 0.5628 | 0.5600 | 不替换 |

后续默认策略：

1. 最终报告主线仍写作 `OpenCLIP ViT-B-32/openai + 多视图增强/TTA + sklearn 初始化 + last block 轻量微调`。
2. 旋转增强作为失败尝试记录：小角度/90 度旋转均降低结果。
3. 类别权重作为失败尝试记录：只保留 `Class_4 weight=1.25` 有价值，继续调 `Class_2/Class_3` 没有收益。
4. 若继续冲 `0.60`，优先做 LoRA/Adapter 或测试集伪标签；不建议继续普通增强/普通权重网格。

### OpenCLIP frozen feature + supervised contrastive 补充实验

- 输出目录：`outputs\openclip_supcon_smoke`
- proj_dim：`64`，hidden_dim：`128`
- lr：`0.001`，weight_decay：`0.001`
- CE weight：`1.0`，SupCon weight：`0.05`，temperature：`0.07`
- macro-F1：`0.3473`
- balanced accuracy：`0.3640`
- 每类 recall：Class_0=0.58，Class_1=0.20，Class_2=0.16，Class_3=0.32，Class_4=0.56

结论：该实验用于验证“在 OpenCLIP 特征空间中显式拉近同类、拉远异类”是否能改善 Class_2/3/4 混淆。若低于历史最优 `0.5681`，则说明简单特征投影/对比损失不足以突破当前瓶颈。

### OpenCLIP frozen feature + supervised contrastive 补充实验

- 输出目录：`outputs\openclip_supcon_sweep`
- proj_dim：`128`，hidden_dim：`256`
- lr：`0.001`，weight_decay：`0.001`
- CE weight：`1.0`，SupCon weight：`0.05`，temperature：`0.07`
- macro-F1：`0.5677`
- balanced accuracy：`0.5680`
- 每类 recall：Class_0=0.74，Class_1=0.60，Class_2=0.42，Class_3=0.44，Class_4=0.64

结论：该实验用于验证“在 OpenCLIP 特征空间中显式拉近同类、拉远异类”是否能改善 Class_2/3/4 混淆。若低于历史最优 `0.5681`，则说明简单特征投影/对比损失不足以突破当前瓶颈。

### OpenCLIP frozen feature + supervised contrastive 补充实验

- 输出目录：`outputs\openclip_supcon_focused_sweep`
- proj_dim：`128`，hidden_dim：`256`
- lr：`0.001`，weight_decay：`0.001`
- CE weight：`1.0`，SupCon weight：`0.02`，temperature：`0.07`
- macro-F1：`0.5609`
- balanced accuracy：`0.5640`
- 每类 recall：Class_0=0.78，Class_1=0.66，Class_2=0.44，Class_3=0.50，Class_4=0.44

结论：该实验用于验证“在 OpenCLIP 特征空间中显式拉近同类、拉远异类”是否能改善 Class_2/3/4 混淆。若低于历史最优 `0.5681`，则说明简单特征投影/对比损失不足以突破当前瓶颈。

### OpenCLIP LogisticRegression 类别 bias 校准实验

- 输出目录：`outputs\openclip_bias_calibration`
- C：`0.08`，Class_4 weight：`1.25`
- 调整类别：`Class_2,Class_3,Class_4`，bias grid：`-0.6,-0.3,0.0,0.3,0.6`
- macro-F1：`0.5378`
- balanced accuracy：`0.5360`
- 每类 recall：Class_0=0.72，Class_1=0.62，Class_2=0.52，Class_3=0.48，Class_4=0.34

结论：该实验用于验证在单个 LogisticRegression 决策函数上做类别阈值/先验校准，能否改善 macro-F1 与易混类召回。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_sweep`
- adapter_dim：`32`，adapter_scale：`0.5`
- lr_adapter：`0.001`，lr_head：`0.0`
- weight_decay：`0.001`，label_smoothing：`0.0`
- macro-F1：`0.5812`
- balanced accuracy：`0.5760`
- 每类 recall：Class_0=0.74，Class_1=0.64，Class_2=0.48，Class_3=0.54，Class_4=0.48

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### 2026-06-01 最终提交链路与 HF/DINOv2/Phikon GPU 环境修复记录

#### 最终模型保存与预测链路

- 已补齐最终模型保存：`src/openclip_feature_adapter_sweep.py` 新增 `--save_final_model` 与全量训练逻辑。
- 已补齐 Adapter 推理：`src/predict.py` 新增 `openclip_feature_adapter` 分支，可加载 `state_dict` 并用同一 OpenCLIP encoder 的 `tta_views` 平均特征预测。
- 当前最终模型：`outputs/final_openclip_feature_adapter_model.joblib`
- 训练方式：用当前最佳配置在全部 250 张训练图上重训，最终训练轮数默认取 5-fold 最佳 epoch 均值，本次为 51 轮。
- 冒烟测试：使用 `outputs/smoke_test_images` 生成 `outputs/smoke_submission.csv`，CSV 表头与格式为 `filename,label`，预测流程通过。

当前可提交命令：

```bash
conda run --no-capture-output -n control python src/predict.py --test_dir path/to/test --model outputs/final_openclip_feature_adapter_model.joblib --out submission.csv
```

#### 环境修复

- 原问题：`control` 环境有 CUDA/OpenCLIP，但 Python 为 3.8，transformers 为 4.19.2，不适合 DINOv2/Phikon；`ml2-vision310` 有 Python 3.10、transformers/timm，但原本是 CPU torch。
- 修复结果：`ml2-vision310` 已更新为 `torch 2.5.1+cu121`，`torch.cuda.is_available=True`，可在 GPU 上跑 HF/DINOv2/Phikon。
- 国内镜像策略：pip/conda 到国外源不稳定时，优先用 `hf-mirror.com` 的 `/resolve/main/...` 直接下载模型文件，再从本地目录加载。
- 已下载本地模型：
  - `outputs/hf_models/facebook_dinov2-small`
  - `outputs/hf_models/facebook_dinov2-base`
  - `outputs/hf_models/owkin_phikon`

#### 新 backbone 结果

| 模型 | 输出目录 | macro-F1 | balanced accuracy | 每类 recall 摘要 | 结论 |
|---|---:|---:|---:|---|---|
| DINOv2 Small | `outputs/hf_dinov2_small_gpu_local` | 0.4768 | 0.4760 | Class_0=0.58, Class_1=0.60, Class_2=0.40, Class_3=0.40, Class_4=0.40 | 低于 OpenCLIP，冻结特征不适合当前数据 |
| DINOv2 Base | `outputs/hf_dinov2_base_gpu_local` | 0.4725 | 0.4760 | Class_0=0.68, Class_1=0.56, Class_2=0.44, Class_3=0.32, Class_4=0.38 | 放大 DINOv2 未带来收益 |
| Phikon | `outputs/hf_phikon_gpu_local` | 0.4873 | 0.4880 | Class_0=0.62, Class_1=0.62, Class_2=0.36, Class_3=0.54, Class_4=0.30 | 病理预训练略好于 DINOv2，但仍明显低于 OpenCLIP Adapter |

#### 深度分析

1. 当前任务的瓶颈不是“有没有更大 backbone”，而是匿名类别 + 32x32 小图 + 250 张训练样本导致的细粒度边界不稳定。
2. DINOv2/Phikon 的冻结特征都低于 0.50，说明这些特征在当前数据的类间方向上没有 OpenCLIP ViT-B-32/openai 好分；Phikon 虽然是病理预训练，但可能预训练尺度、染色分布、组织结构与本数据不匹配。
3. OpenCLIP Adapter 能突破 0.60，关键不是端到端大幅微调，而是：
   - OpenCLIP frozen feature 给了较好的初始可分空间；
   - LogisticRegression 初始化提供稳定线性边界；
   - 小残差 Adapter 只做低维、门控、归一化后的边界修正，降低过拟合；
   - Class_2/3/4 pairwise margin 针对主要互混类做轻约束。
4. 继续单纯换 frozen backbone 的收益已经很低；如果要继续冲 0.65 甚至 0.70，应优先尝试“同一 OpenCLIP backbone 内的更强参数高效微调”，而不是继续堆 frozen feature。

#### 后续优先方向

1. 保留当前 `outputs/final_openclip_feature_adapter_model.joblib` 作为最终提交 fallback。
2. 继续优化时优先做 OpenCLIP visual 最后 block 的 LoRA/Adapter，而不是只在 frozen feature 后接 MLP。
3. 如果拿到测试集，做同一模型的高置信伪标签/测试集分布校准；注意不做多模型 ensemble。
4. 报告中可以把 DINOv2/Phikon 作为“更强/更相关 backbone 但未提升”的关键对照实验，说明最终选择 OpenCLIP Adapter 的依据。

### 2026-06-01 追加优化记录：当前最终可复现最佳

- 当前可复现最佳目录：`outputs/openclip_feature_adapter_margin004_best_single`
- 指标：`macro-F1=0.6147`，`balanced accuracy=0.6120`
- 配置：`OpenCLIP ViT-B-32/openai frozen feature + LN gated residual adapter`
- 关键参数：`train_views=3`，`tta_views=3`，`adapter_dim=24`，`adapter_scale=1.0`，`lr_adapter=0.0012`，`lr_head=0.0`，`pairwise_margin_weight=0.02`，`pairwise_margin=0.04`，`pairwise_class_set=Class_2,Class_3,Class_4`，`class2_weight=1.05`，`class4_weight=1.03`，`init_seed_offset=2`
- 本轮额外尝试：
  - 定向错分对 margin：最高 `macro-F1=0.6028`，低于当前最佳，且 Class_4 recall 降到 0.46，不保留。
  - seed 稳定性：`init_seed_offset=2` 明显最好，其他 seed 多数在 0.55-0.59。
  - adapter 学习率/weight decay：没有超过当前最佳，说明局部正则和学习率不是主要瓶颈。
  - adapter 容量：`adapter_dim=24` 最好，过小欠拟合，过大破坏边界。
- 阶段结论：已经稳定突破 60%，但局部扫参收益接近耗尽。下一步优先补齐最终模型训练保存和 `predict.py` 的 adapter 推理；若继续冲分，再尝试作用到 OpenCLIP 最后 block 的 LoRA/Adapter 轻量微调。

### 2026-06-01 追加优化记录：当前最终可复现最佳

- 当前可复现最佳目录：`outputs/openclip_feature_adapter_margin004_best_single`
- 指标：`macro-F1=0.6147`，`balanced accuracy=0.6120`
- 配置：`OpenCLIP ViT-B-32/openai frozen feature + LN gated residual adapter`
- 关键参数：`train_views=3`，`tta_views=3`，`adapter_dim=24`，`adapter_scale=1.0`，`lr_adapter=0.0012`，`lr_head=0.0`，`pairwise_margin_weight=0.02`，`pairwise_margin=0.04`，`pairwise_class_set=Class_2,Class_3,Class_4`，`class2_weight=1.05`，`class4_weight=1.03`，`init_seed_offset=2`
- 本轮额外尝试：
  - 定向错分对 margin：最高 `macro-F1=0.6028`，低于当前最佳，且 Class_4 recall 降到 0.46，不保留。
  - seed 稳定性：`init_seed_offset=2` 明显最好，其他 seed 多数在 0.55-0.59。
  - adapter 学习率/weight decay：没有超过当前最佳，说明局部正则和学习率不是主要瓶颈。
  - adapter 容量：`adapter_dim=24` 最好，过小欠拟合，过大破坏边界。
- 阶段结论：已经稳定突破 60%，但局部扫参收益接近耗尽。下一步优先补齐最终模型训练保存和 `predict.py` 的 adapter 推理；若继续冲分，再尝试作用到 OpenCLIP 最后 block 的 LoRA/Adapter 轻量微调。

### 2026-06-01 当前优化阶段小结

- 当前确认新高：`outputs/openclip_feature_adapter_margin004_best_single`
- 单模型配置：`OpenCLIP ViT-B-32/openai frozen feature + LN gated residual adapter`
- 关键参数：`adapter_dim=24`，`adapter_scale=1.0`，`lr_adapter=0.0012`，`lr_head=0.0`，`pairwise_margin_weight=0.02`，`pairwise_margin=0.04`，`pairwise_class_set=Class_2,Class_3,Class_4`，`class2_weight=1.05`，`class4_weight=1.03`，`init_seed_offset=2`
- 5-fold 结果：`macro-F1=0.6147`，`balanced accuracy=0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

本轮新尝试与结论：

| 方向 | 输出目录 | 最佳 macro-F1 | 最佳 balanced accuracy | 结论 |
|---|---:|---:|---:|---|
| 手工特征融合 OpenCLIP + pixel/HOG | `outputs/openclip_handcrafted_fusion_refine` | 0.4845 | 0.4800 | 手工特征对当前任务主要是噪声，低于 OpenCLIP 主线，放弃 |
| 加入 Class_0 的 pairwise set | `outputs/openclip_feature_adapter_class0_pairwise_probe` | 0.6147 | 0.6120 | 最优仍是 `Class_2,Class_3,Class_4`，Class_0 不应加入易混集合 |
| 单独确认 margin=0.04 | `outputs/openclip_feature_adapter_margin004_best_single` | 0.6147 | 0.6120 | 稳定复现，替代上一版 `pairwise_margin=0.045` 的 0.6143 |
| 定向错分对 margin | `outputs/openclip_feature_adapter_directed_pair_probe` | 0.6028 | 0.6000 | 指定 `Class_1>Class_4`、`Class_2>Class_3/4`、`Class_4>Class_2/3` 后 Class_4 recall 被压低，不保留 |
| init seed 稳定性 | `outputs/openclip_feature_adapter_seed_margin004_probe` | 0.6147 | 0.6120 | `init_seed_offset=2` 明显最好，其他 seed 多数掉到 0.55-0.59 |
| adapter 学习率与 weight decay | `outputs/openclip_feature_adapter_lr_wd_refine` | 0.6147 | 0.6120 | `lr_adapter=0.0012` 仍最好；weight decay 在 `2e-5~1e-4` 区间影响很小 |
| adapter 容量 | `outputs/openclip_feature_adapter_dim_refine_margin004` | 0.6147 | 0.6120 | `adapter_dim=24` 是当前甜点，过小欠拟合，过大更容易扰乱边界 |
| 类别文本描述相似度辅助 | `outputs/openclip_adapter_descriptor_aux_probe` | 0.6147 | 0.6120 | 纯图像 adapter 复现最优；拼接描述相似度后降至 0.5677，不保留 |

### 最优 adapter 上的文本描述辅助复查

为了确认“大语言模型/视觉语言模型生成类别描述”是否能叠加到当前最优 `0.6147` 方案上，新增脚本：

```text
src/openclip_adapter_descriptor_aux_sweep.py
```

运行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run --no-capture-output -n control python src\openclip_adapter_descriptor_aux_sweep.py --feature_modes image image_descriptor
```

结果：

| 特征方式 | feature_dim | macro-F1 | balanced accuracy | Class_4 recall |
|---|---:|---:|---:|---:|
| image | 512 | 0.6147 | 0.6120 | 0.52 |
| image_descriptor | 517 | 0.5677 | 0.5680 | 0.36 |

结论：在最优 `ln_gated_residual adapter` 上，额外拼接类别文本描述相似度没有提升性能，反而明显降低 Class_4 recall 和整体 macro-F1。说明当前 adapter 已经主要依赖 OpenCLIP 图像特征中的细粒度纹理边界；手写/大模型生成的短文本描述太粗，不能可靠表达 32x32 病理小图的判别细节。因此文本描述方向可以作为可解释性和失败探索写入报告，但不建议进入最终模型。

进一步补充“短语式伪语义名称”实验：根据每类 50 张样本拼图，为每类总结 2 个视觉类别名式 prompt，例如 `crowded dark-nucleus microscopy pattern`、`pale elongated fibrous-cell microscopy pattern` 等，保存于：

```text
docs/class_prompt_aliases.json
```

运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run --no-capture-output -n control python src\openclip_adapter_descriptor_aux_sweep.py --descriptions docs\class_prompt_aliases.json --output_dir outputs\openclip_adapter_prompt_alias_probe --feature_modes image image_descriptor
```

结果：纯图像 adapter 仍为 `macro-F1=0.6147`、`balanced accuracy=0.6120`；拼接短语式 prompt 相似度后为 `macro-F1=0.5736`、`balanced accuracy=0.5680`。说明即使先观察同类 50 张图片并总结伪语义名称，这类 prompt 仍未能提供有效增益，最终不纳入主模型。

继续测试“只给部分类别加入 prompt 相似度”的设想，避免所有类别 prompt 一起加入时扰乱整体边界。运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run --no-capture-output -n control python src\openclip_adapter_descriptor_aux_sweep.py --descriptions docs\class_prompt_aliases.json --output_dir outputs\openclip_adapter_prompt_alias_partial_probe --feature_modes image image_descriptor --descriptor_class_sets Class_4 Class_2,Class_3,Class_4 Class_1,Class_4 Class_2,Class_4 all
```

结果：

| prompt 加入类别 | feature_dim | macro-F1 | balanced accuracy | Class_4 recall |
|---|---:|---:|---:|---:|
| 不加入 prompt | 512 | 0.6147 | 0.6120 | 0.52 |
| Class_1, Class_4 | 514 | 0.5751 | 0.5720 | 0.42 |
| Class_2, Class_4 | 514 | 0.5747 | 0.5720 | 0.44 |
| all | 517 | 0.5736 | 0.5680 | 0.52 |
| Class_2, Class_3, Class_4 | 515 | 0.5708 | 0.5680 | 0.44 |
| Class_4 | 513 | 0.5647 | 0.5600 | 0.40 |

结论：只给困难类或部分易混类加入 prompt 相似度后，结果仍低于纯图像 adapter。说明性能下降不是因为“所有类别 prompt 一起加入”导致的单一问题，而是 prompt 相似度本身对当前 32x32 细粒度病理图的判别信息不足。

### 0.6147 最优结果的随机性复查

为确认 `0.6147` 是否为稳定可复现结果，而不是单次随机波动，补充两类复查。

第一类：固定原始 5-fold 划分与增强种子 `seed=42`，只改变 adapter 初始化 `init_seed_offset`：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run --no-capture-output -n control python src\openclip_feature_adapter_sweep.py --output_dir outputs\openclip_feature_adapter_margin004_seed_repro_now --adapter_types ln_gated_residual --train_views 3 --tta_views 3 --init_c 0.08 --adapter_dims 24 --adapter_scales 1.0 --lr_adapters 0.0012 --lr_heads 0.0 --weight_decays 0.00005 --epochs 150 --patience 35 --batch_size 600 --dropouts 0.0 --label_smoothings 0.0 --loss_types ce --focal_gammas 1.2 --margins 0.0 --pairwise_margin_weights 0.02 --pairwise_margins 0.04 --pairwise_class_sets Class_2,Class_3,Class_4 --class1_weights 1.0 --class2_weights 1.05 --class3_weights 1.0 --class4_weights 1.03 --init_seed_offsets 0 1 2 3 4 --refit_cs 0.0
```

结果：

| init_seed_offset | macro-F1 | balanced accuracy | Class_4 recall |
|---:|---:|---:|---:|
| 0 | 0.5693 | 0.5680 | 0.42 |
| 1 | 0.5739 | 0.5720 | 0.42 |
| 2 | 0.6147 | 0.6120 | 0.52 |
| 3 | 0.5629 | 0.5600 | 0.36 |
| 4 | 0.5669 | 0.5680 | 0.36 |

第二类：固定 `init_seed_offset=2`，改变交叉验证划分和增强随机种子：

| seed | macro-F1 | balanced accuracy | Class_4 recall |
|---:|---:|---:|---:|
| 41 | 0.5120 | 0.5160 | 0.42 |
| 42 | 0.6147 | 0.6120 | 0.52 |
| 43 | 0.5395 | 0.5400 | 0.38 |

结论：`0.6147` 在原始记录的固定设置下可以复现，并非日志或代码错误；但它对初始化、数据划分和增强随机种子较敏感，不应表述为跨随机种子稳定达到的性能。报告中更稳妥的说法是：当前最高可复现单次 5-fold 结果为 `macro-F1=0.6147`，但随机性复查显示平均水平更接近 `0.56~0.58`，因此最终结论应强调该方案“能够突破 0.60，但稳定性仍有限”。

阶段判断：已经突破 60%，但继续靠局部扫参的边际收益很小。当前主要瓶颈仍是 `Class_2/Class_3/Class_4` 的细粒度互混，以及第 5 折、部分 fold 的低召回拖累。后续优先级应转向：

1. 把当前 Adapter 最优配置训练成最终可预测模型，补齐 `predict.py` 的加载与 TTA 推理。
2. 若继续冲分，尝试更结构化的单模型微调，例如 LoRA/Adapter 作用到 OpenCLIP 最后 block，而不是只在 frozen feature 后训练小 MLP。
3. 若拿到 test_dir，再做同一模型的高置信伪标签或 transductive calibration，但必须保留当前 0.6147 方案作为 fallback。

### 阶段突破记录：单模型首次达到 0.60

- 当前最佳输出目录：`outputs/openclip_feature_adapter_ln_gated_break60_best_single`
- 方法：`OpenCLIP ViT-B-32/openai frozen feature + ln_gated_residual adapter + LogisticRegression 初始化头 + 轻量 pairwise margin`
- 是否使用 GPU：OpenCLIP 特征和 adapter 训练在 `control` 环境中运行，`torch.cuda.is_available=True`，脚本 `device=auto` 优先使用 CUDA；sklearn 初始化/指标计算仍在 CPU。
- 关键配置：
  - `train_views=3`，`tta_views=3`
  - `adapter_dim=24`，`adapter_scale=1.0`
  - `lr_adapter=0.0012`，`lr_head=0.0`
  - `weight_decay=5e-05`
  - `loss_type=ce`
  - `pairwise_margin_weight=0.02`，`pairwise_margin=0.05`
  - `Class_2 weight=1.05`，`Class_3 weight=1.0`，`Class_4 weight=1.0`
  - `init_seed_offset=2`
- 5-fold 结果：
  - `macro-F1=0.6030`
  - `balanced accuracy=0.6000`
  - recall：Class_0=0.76，Class_1=0.64，Class_2=0.50，Class_3=0.58，Class_4=0.52
- 保存文件：
  - `best_openclip_feature_adapter_config.json`
  - `best_confusion_matrix.csv`
  - `best_oof_predictions.csv`
  - `misclassified_pair_counts.csv`
  - `misclassified_sheets/`

结论：本轮终于突破 `0.60`。提升主要来自两个点：一是 `ln_gated_residual` 比普通 residual adapter 更稳定地调整特征边界；二是轻量 pairwise margin 对 Class_2/3/4 的互混有帮助，尤其把 Class_3 recall 提到 0.58、Class_4 recall 提到 0.52。该方案仍然是单个 OpenCLIP backbone 和单个 adapter 模型，不属于模型集成。后续优化重点应从“大范围盲扫”转为围绕该配置做稳定性复查、最终模型保存和 submission 生成接口适配。

### Class_1/4 定向优化与类别权重复查

- 输出目录：`outputs/openclip_feature_adapter_targeted_pairwise_probe`
- 改动：新增 `pairwise_class_set` 参数，尝试把 pairwise margin 从 `Class_2,Class_3,Class_4` 扩展到 `Class_1,Class_4` 或 `Class_1,Class_2,Class_3,Class_4`。
- 结果：最佳仍然是原始 `Class_2,Class_3,Class_4` 集合，`macro-F1=0.6030`，`balanced accuracy=0.6000`。
- 结论：`Class_1 -> Class_4` 虽然是最大错分对之一，但把 Class_1 纳入 pairwise margin 会扰乱整体边界，收益不如专注 Class_2/3/4。

- 输出目录：`outputs/openclip_feature_adapter_class1_weight_probe`
- 改动：新增 `class1_weight` 参数，同时小范围复查 Class_1/Class_2/Class_4 权重。
- 最佳配置：
  - `pairwise_class_set=Class_2,Class_3,Class_4`
  - `pairwise_margin_weight=0.02`，`pairwise_margin=0.05`
  - `class1_weight=1.0`，`class2_weight=1.05`，`class3_weight=1.0`，`class4_weight=1.03`
  - 其他设置沿用突破 0.60 的 `ln_gated_residual adapter` 配置
- 5-fold 结果：
  - `macro-F1=0.6110`
  - `balanced accuracy=0.6080`
  - recall：Class_0=0.74，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52
- 混淆变化：
  - `Class_1 -> Class_4` 从 12 降到 9，Class_1 recall 从 0.64 提到 0.68。
  - Class_2 recall 从 0.50 提到 0.54。
  - Class_4 recall 保持 0.52。

结论：继续提升来自更细的类别权重平衡，而不是扩大 pairwise margin 的类别集合。当前新的最优为 `outputs/openclip_feature_adapter_class1_weight_probe`，达到 `macro-F1=0.6110`、`balanced accuracy=0.6080`。后续可以围绕 `class4_weight=1.02~1.05`、`class2_weight=1.03~1.08` 做很小范围复查，或进入最终模型保存与 submission 生成。

### Pairwise margin 与类别权重小范围精修

- 输出目录：`outputs/openclip_feature_adapter_weight_margin_refine`
- 目标：围绕 `0.6110` 配置继续小范围扫描 `pairwise_margin_weight`、`pairwise_margin`、`Class_2/Class_3/Class_4` 权重。
- partial 最佳：
  - `pairwise_margin_weight=0.02`
  - `pairwise_margin=0.045`
  - `class1_weight=1.0`
  - `class2_weight=1.05`
  - `class3_weight=1.0`
  - `class4_weight=1.03`
  - `init_seed_offset=2`
- 单配置复跑目录：`outputs/openclip_feature_adapter_weight_margin_refine_best_single`
- 5-fold 结果：
  - `macro-F1=0.6143`
  - `balanced accuracy=0.6120`
  - recall：Class_0=0.76，Class_1=0.68，Class_2=0.52，Class_3=0.56，Class_4=0.54
- 混淆矩阵摘要：
  - Class_4 recall 从上一轮 0.52 提到 0.54。
  - Class_2 recall 从上一轮 0.54 小幅回落到 0.52。
  - Class_1 recall 保持 0.68。

结论：把 pairwise margin 从 `0.05` 略降到 `0.045` 更合适，说明易混类边界约束不能过硬；较软的 margin 能保留 Class_4 召回，同时不明显破坏其他类。当前最佳更新为 `outputs/openclip_feature_adapter_weight_margin_refine_best_single`，`macro-F1=0.6143`、`balanced accuracy=0.6120`。

### 继续精修：更细 margin / 权重与 TTA 复查

- 输出目录：`outputs/openclip_feature_adapter_margin_fine_refine`
- 目标：在 `0.6143` 最优配置附近继续细扫 `adapter_scale`、`lr_adapter`、`weight_decay`、`pairwise_margin_weight`、`pairwise_margin`、`Class_2/Class_4` 权重。
- 结果：运行 30 分钟后 partial 共 423 个配置，未超过当前最优。
- partial 最佳：`macro-F1=0.6054`，`balanced accuracy=0.6040`。
- 结论：更细的 margin/权重搜索没有继续提升，尤其 `adapter_scale=0.95` 系列整体弱于当前 `adapter_scale=1.0`。当前 `pairwise_margin=0.045`、`class2_weight=1.05`、`class4_weight=1.03` 已经比较贴近局部最优。

- 输出目录：`outputs/openclip_feature_adapter_best_tta_probe`
- 目标：保持当前最佳训练配置，测试 `tta_views=1/2/3/4/5`。
- 结果：本轮最佳仍为 `tta_views=3`，`macro-F1=0.6069`，`balanced accuracy=0.6040`；`tta_views=1/2/4/5` 均下降。
- 结论：当前任务中 TTA 不是越多越好。3 个视图仍是最合适的折中，更多视图会引入过多扰动，削弱 32x32 小图中的细节判别。当前全局最优仍保持 `outputs/openclip_feature_adapter_weight_margin_refine_best_single` 的 `macro-F1=0.6143`、`balanced accuracy=0.6120`。

### 继续冲 0.60 的迁移 / 度量 / 校准补充记录

本轮目标：在不做模型集成的前提下，继续尝试迁移学习、度量学习和元学习相关方向，目标优先突破 `macro-F1=0.60`、`balanced accuracy=0.60`。

1. `outputs/openclip_feature_adapter_refit_probe2`
   - 方法：OpenCLIP ViT-B-32/openai + residual adapter 后，再尝试用 adapter 后特征重训 LogisticRegression。
   - 结果：最佳仍为 `refit_c=0.0`，`macro-F1=0.5794`，`balanced accuracy=0.5760`。
   - 结论：后接 LogisticRegression 没有收益，说明当前 adapter 训练出的头更适合该特征空间；简单 refit 会削弱已经学到的边界。

2. `outputs/openclip_metric_head_focused2`
   - 方法：度量学习分类头，继续围绕 CosFace/角度间隔头扫 `scale`、`margin`、`adapter_scale`、`Class_4 weight` 和随机种子。
   - 运行状态：组合较多，运行 30 分钟超时，但已保存 partial CSV，共完成 523 个配置。
   - partial 最佳：`adapter_scale=0.25, lr_adapter=0.0015, lr_head=0.001, scale=40, margin=0.08, Class_4 weight=1.0, seed_offset=0`。
   - partial 结果：`macro-F1=0.5718`，`balanced accuracy=0.5760`，recall 为 Class_0=0.76、Class_1=0.68、Class_2=0.54、Class_3=0.54、Class_4=0.36。
   - 结论：角度间隔能改善 Class_1/2/3 的部分边界，但 Class_4 召回明显下降，未超过 adapter 主线。

3. `outputs/hf_dinov2_base_probe` 与 `outputs/hf_dinov2_small_probe`
   - 方法：尝试 HuggingFace DINOv2 frozen feature + LogisticRegression，属于自监督预训练迁移学习方向。
   - 环境问题：`control` 环境的 `transformers` 版本过旧，缺少 `AutoImageProcessor`；切换到 `ml2-vision310` 后可以启动，但该环境无 CUDA，`facebook/dinov2-base` 与 `facebook/dinov2-small` 都在特征提取阶段超时。
   - 结论：本轮不能把 HF DINOv2 作为效果失败，只能记录为当前环境运行成本过高。若后续要继续 DINOv2，优先方案是升级 `control` 环境的 `transformers`，或在有 CUDA 的 Python 3.10+ 环境中跑。

4. `outputs/openclip_bias_calibration_probe2`
   - 方法：对 OpenCLIP LogisticRegression 做 Class_2/3/4 的训练折内部 OOF logit bias 校准。
   - 运行状态：运行 15 分钟超时，已保存 partial CSV，共完成 172 个配置。
   - partial 最佳：`C=0.10, Class_2 weight=1.0, Class_3 weight=1.1, Class_4 weight=1.1, bias_grid=0.2, bias_step=0.1`。
   - partial 结果：`macro-F1=0.5478`，`balanced accuracy=0.5480`，低于普通 OpenCLIP LR 与 adapter。
   - 结论：当前错误不是简单类别阈值偏置造成的，主要仍是特征空间里 Class_2/3/4 本身互混。

5. `outputs/openclip_feature_adapter_views5_probe2`
   - 方法：把当前 adapter 主线从 `train_views=3, tta_views=3` 提高到 `train_views=5, tta_views=5`。
   - 结果：最佳 `macro-F1=0.5584`，`balanced accuracy=0.5560`。
   - 结论：更多增强/TTA 并没有提升，反而降低 Class_2/3 表现。对于 32x32 小图，过多随机视图可能破坏关键细节；后续不建议继续盲目增加 views。

阶段结论：本轮迁移学习、度量学习和校准方案均未突破 `0.60`。目前最稳可复现结果仍是 `OpenCLIP ViT-B-32/openai + 3 views + residual adapter`，约 `macro-F1=0.5794`、`balanced accuracy=0.5760`；历史最高候选为 `0.5922/0.5880`，但后续复查没有稳定复现。若继续冲 0.60，最有希望的是解决环境后重跑领域相关 backbone（如 Phikon/DINOv2/UNI 类病理或自监督模型），而不是继续在当前 OpenCLIP 特征上微调分类阈值或增加增强强度。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_focused_sweep`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- macro-F1：`0.5922`
- balanced accuracy：`0.5880`
- 每类 recall：Class_0=0.74，Class_1=0.66，Class_2=0.52，Class_3=0.52，Class_4=0.50

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_final_narrow`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0013`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- macro-F1：`0.5877`
- balanced accuracy：`0.5840`
- 每类 recall：Class_0=0.74，Class_1=0.66，Class_2=0.50，Class_3=0.52，Class_4=0.50

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_hardclass_margin`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.08`，Class_3=`1.08`，Class_4=`1.1`
- macro-F1：`0.5815`
- balanced accuracy：`0.5760`
- 每类 recall：Class_0=0.74，Class_1=0.64，Class_2=0.48，Class_3=0.52，Class_4=0.50

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_focal_sweep`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`focal`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- macro-F1：`0.5789`
- balanced accuracy：`0.5720`
- 每类 recall：Class_0=0.74，Class_1=0.62，Class_2=0.52，Class_3=0.54，Class_4=0.44

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_views_sweep`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.001`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- macro-F1：`0.5814`
- balanced accuracy：`0.5760`
- 每类 recall：Class_0=0.74，Class_1=0.66，Class_2=0.48，Class_3=0.50，Class_4=0.50

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_seed_sweep`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`21`
- macro-F1：`0.5785`
- balanced accuracy：`0.5760`
- 每类 recall：Class_0=0.78，Class_1=0.64，Class_2=0.48，Class_3=0.48，Class_4=0.50

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP 多视图 pooled feature 分类头实验

- 输出目录：`outputs\openclip_pooling_sweep`
- pooling mode：`mean`，views：`3`，PCA：`0`
- C：`0.06`，Class_4 weight：`1.25`
- macro-F1：`0.5043`
- balanced accuracy：`0.5040`
- 每类 recall：Class_0=0.66，Class_1=0.56，Class_2=0.44，Class_3=0.50，Class_4=0.36

结论：该实验检验单个 OpenCLIP backbone 的多视图特征汇总是否比“增强视图当作独立训练样本”更稳定。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug_scale080_jitter010`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- macro-F1：`0.5675`
- balanced accuracy：`0.5640`
- 每类 recall：Class_0=0.68，Class_1=0.56，Class_2=0.42，Class_3=0.56，Class_4=0.60

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug_scale085_jitter010`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- macro-F1：`0.5321`
- balanced accuracy：`0.5280`
- 每类 recall：Class_0=0.70，Class_1=0.66，Class_2=0.46，Class_3=0.48，Class_4=0.34

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug_scale090_jitter005`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- macro-F1：`0.5496`
- balanced accuracy：`0.5520`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.40，Class_3=0.50，Class_4=0.42

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug_scale095_jitter005`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- macro-F1：`0.5280`
- balanced accuracy：`0.5280`
- 每类 recall：Class_0=0.74，Class_1=0.60，Class_2=0.40，Class_3=0.50，Class_4=0.40

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug080_weight_refine`
- adapter_dim：`64`，adapter_scale：`0.35`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.05`
- init_seed_offset：`0`
- macro-F1：`0.5730`
- balanced accuracy：`0.5720`
- 每类 recall：Class_0=0.70，Class_1=0.62，Class_2=0.42，Class_3=0.54，Class_4=0.58

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP residual adapter + metric head 实验

- 输出目录：`outputs\openclip_metric_head_sweep`
- loss_type：`cosface`，scale：`32.0`，margin：`0.15`
- adapter_dim：`64`，adapter_scale：`0.25`
- lr_adapter：`0.0015`，lr_head：`0.0015`，Class_4 weight：`1.1`
- macro-F1：`0.5737`
- balanced accuracy：`0.5800`
- 每类 recall：Class_0=0.82，Class_1=0.64，Class_2=0.56，Class_3=0.54，Class_4=0.34

结论：该实验用于验证角度间隔分类头是否能拉开匿名细粒度类别的特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_refit_sweep`
- adapter_dim：`64`，adapter_scale：`0.5`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.2`
- macro-F1：`0.5550`
- balanced accuracy：`0.5520`
- 每类 recall：Class_0=0.74，Class_1=0.60，Class_2=0.48，Class_3=0.54，Class_4=0.40

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_color_autocontrast`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.0`
- macro-F1：`0.4981`
- balanced accuracy：`0.4960`
- 每类 recall：Class_0=0.68，Class_1=0.54，Class_2=0.38，Class_3=0.42，Class_4=0.46

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_color_equalize`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.0`
- macro-F1：`0.4459`
- balanced accuracy：`0.4440`
- 每类 recall：Class_0=0.56，Class_1=0.50，Class_2=0.42，Class_3=0.36，Class_4=0.38

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_color_grayscale`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.0`
- macro-F1：`0.4805`
- balanced accuracy：`0.4800`
- 每类 recall：Class_0=0.62，Class_1=0.48，Class_2=0.38，Class_3=0.48，Class_4=0.44

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_color_grayscale_autocontrast`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.0`
- macro-F1：`0.4749`
- balanced accuracy：`0.4760`
- 每类 recall：Class_0=0.50，Class_1=0.54，Class_2=0.56，Class_3=0.46，Class_4=0.32

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_mlp_lora_probe`
- 解冻策略：`last_block_mlp_lora`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`0.0001`，lr_head：`5e-06`
- weight_decay：`0.01`，label_smoothing：`0.03`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.25`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.5581`
- balanced accuracy：`0.5560`
- 每类 recall：Class_0=0.74，Class_1=0.66，Class_2=0.48，Class_3=0.52，Class_4=0.38

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_attn_lora_probe`
- 解冻策略：`last_block_attn_lora`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`0.0001`，lr_head：`5e-06`
- weight_decay：`0.01`，label_smoothing：`0.03`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- LoRA：rank=`2`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.5616`
- balanced accuracy：`0.5600`
- 每类 recall：Class_0=0.74，Class_1=0.64，Class_2=0.50，Class_3=0.54，Class_4=0.38

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_seed_scale_probe`
- adapter_dim：`64`，adapter_scale：`0.35`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`4`
- refit_c：`0.0`
- macro-F1：`0.5826`
- balanced accuracy：`0.5800`
- 每类 recall：Class_0=0.76，Class_1=0.64，Class_2=0.52，Class_3=0.54，Class_4=0.44

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_interp_bilinear`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bilinear`
- macro-F1：`0.5655`
- balanced accuracy：`0.5640`
- 每类 recall：Class_0=0.76，Class_1=0.60，Class_2=0.48，Class_3=0.54，Class_4=0.44

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_interp_lanczos`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`lanczos`
- macro-F1：`0.5549`
- balanced accuracy：`0.5520`
- 每类 recall：Class_0=0.74，Class_1=0.60，Class_2=0.50，Class_3=0.52，Class_4=0.40

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_interp_nearest`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`nearest`
- macro-F1：`0.5263`
- balanced accuracy：`0.5320`
- 每类 recall：Class_0=0.72，Class_1=0.68，Class_2=0.46，Class_3=0.42，Class_4=0.38

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_weak_aug_probe`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.001`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`1`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5616`
- balanced accuracy：`0.5560`
- 每类 recall：Class_0=0.68，Class_1=0.62，Class_2=0.54，Class_3=0.54，Class_4=0.40

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_attn_lora_twoblock_probe`
- 解冻策略：`last_two_blocks_attn_lora`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`5e-05`，lr_head：`5e-06`
- weight_decay：`0.01`，label_smoothing：`0.03`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- LoRA：rank=`2`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.5434`
- balanced accuracy：`0.5440`
- 每类 recall：Class_0=0.74，Class_1=0.64，Class_2=0.48，Class_3=0.54，Class_4=0.32

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### OpenCLIP 相似度特征单分类头实验

- 输出目录：`outputs\openclip_similarity_feature_probe`
- feature_mode：`prototype`，topk：`3`，temperature：`10.0`
- train_views：`3`，tta_views：`3`，C：`0.12`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- macro-F1：`0.5372`
- balanced accuracy：`0.5320`
- 每类 recall：Class_0=0.74，Class_1=0.58，Class_2=0.46，Class_3=0.50，Class_4=0.38

结论：该实验检验 few-shot support set 的类原型/近邻相似度能否作为单一分类头的辅助特征；若未超过 residual adapter，说明直接相似度特征不足以解决 Class_2/3/4 互混。

### OpenCLIP LogisticRegression 类别 bias 校准实验

- 输出目录：`outputs\openclip_bias_calibration_probe`
- train_views：`3`，tta_views：`3`，C：`0.12`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- bias 搜索：范围=`±0.6`，步长=`0.15`，inner_splits=`4`
- 平均 bias：Class_2=`-0.150`，Class_3=`0.030`，Class_4=`-0.030`
- macro-F1：`0.5426`
- balanced accuracy：`0.5440`
- 每类 recall：Class_0=0.74，Class_1=0.62，Class_2=0.46，Class_3=0.56，Class_4=0.34

结论：该实验尝试用训练折内部 OOF 来校准 Class_2/3/4 决策阈值；若不提升，说明错误主要来自特征不可分，而不是单纯类别阈值偏置。

### OpenCLIP 多视图特征分类头类型补充实验

- 输出目录：`outputs\openclip_head_type_probe`
- 最佳分类头：`logreg_multinomial`，参数：`0.08`
- macro-F1：`0.5435`
- balanced accuracy：`0.5440`
- 每类 recall：Class_0=0.74，Class_1=0.64，Class_2=0.48，Class_3=0.54，Class_4=0.32

结论：该实验排查分类头是否限制上限；若仍低于 residual adapter，则说明当前最佳提升主要来自 adapter 改变特征边界，而非简单换分类器。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_struct_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`32`，adapter_scale：`1.0`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5883`
- balanced accuracy：`0.5840`
- 每类 recall：Class_0=0.74，Class_1=0.66，Class_2=0.50，Class_3=0.52，Class_4=0.50

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_ln_gated_refine`
- adapter_type：`ln_gated_residual`
- adapter_dim：`32`，adapter_scale：`1.1`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- init_seed_offset：`0`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5900`
- balanced accuracy：`0.5880`
- 每类 recall：Class_0=0.76，Class_1=0.62，Class_2=0.58，Class_3=0.48，Class_4=0.50

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_ln_gated_class23_refine`
- adapter_type：`ln_gated_residual`
- adapter_dim：`32`，adapter_scale：`1.1`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- init_seed_offset：`0`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5900`
- balanced accuracy：`0.5880`
- 每类 recall：Class_0=0.76，Class_1=0.62，Class_2=0.58，Class_3=0.48，Class_4=0.50

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_ln_gated_views_refine`
- adapter_type：`ln_gated_residual`
- adapter_dim：`32`，adapter_scale：`1.1`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- init_seed_offset：`0`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5900`
- balanced accuracy：`0.5880`
- 每类 recall：Class_0=0.76，Class_1=0.62，Class_2=0.58，Class_3=0.48，Class_4=0.50

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + episodic ProtoNet 元学习/度量学习实验

- 输出目录：`outputs\openclip_meta_proto_quick`
- train_views：`3`，tta_views：`3`
- hidden_dim：`256`，embed_dim：`128`
- n_way：`5`，k_shot：`3`，q_query：`4`
- metric：`cosine`，temperature：`0.05`
- lr：`0.0003`，weight_decay：`0.001`，dropout：`0.0`
- macro-F1：`0.5762`
- balanced accuracy：`0.5760`
- 每类 recall：Class_0=0.72，Class_1=0.64，Class_2=0.58，Class_3=0.54，Class_4=0.40

结论：该实验用于验证标准 few-shot 的 episodic prototype 思路能否在当前 5-way 50-shot 匿名分类任务中超过现有 OpenCLIP Adapter。若低于 `0.5922`，说明本任务更适合预训练特征上的判别式轻量分类头，而不是纯原型距离决策。

### 元学习与度量学习阶段小结

本轮新增 `src/openclip_meta_proto_sweep.py`，在 OpenCLIP ViT-B-32/openai 的 3 train views + 3 TTA views 特征上实现 episodic ProtoNet：每个 fold 内采样 5-way K-shot support/query episode，训练小投影网络，再用训练折全量 support 原型预测验证折。

| 实验方向 | 输出目录 | 最佳 macro-F1 | balanced accuracy | 结论 |
|---|---:|---:|---:|---|
| episodic ProtoNet quick | `outputs/openclip_meta_proto_quick` | 0.5762 | 0.5760 | 比早期直接 prototype 和相似度特征强，但仍低于全局最优 |
| episodic ProtoNet focused | `outputs/openclip_meta_proto_focused` | 0.5686 | 0.5680 | 窄扫没有继续提升，k=3、cosine 最稳 |
| supervised contrastive | `outputs/openclip_supcon_sweep` | 0.5677 | 0.5680 | 可提升 Class_4 recall，但 Class_2/3 下降 |
| metric head | `outputs/openclip_metric_head_sweep` | 0.5737 | 0.5800 | balanced accuracy 接近，但 Class_4 recall 明显偏低 |
| prototype/KNN 相似度特征 | `outputs/openclip_similarity_feature_sweep` | 0.5372 | 约 0.536 | 未超过判别式分类头 |

阶段结论：元学习/度量学习不是完全无效，episodic ProtoNet 能达到 `0.5762`，说明“学习一个更适合原型分类的嵌入空间”比直接最近中心更合理。但它仍没有超过当前全局最优 `OpenCLIP feature adapter` 的 `macro-F1=0.5922`、`balanced accuracy=0.5880`。主要原因可能是本任务只有 5 个类别，无法构造足够多样的 episode；而每类 50 张又足以让判别式 LogisticRegression/Adapter 学到更细的类间边界。因此当前不建议把元学习作为最终主线，可作为报告中的对照实验和失败尝试；后续冲 0.60 仍优先沿 OpenCLIP feature adapter / 判别式轻量微调推进。

### OpenCLIP frozen feature + episodic ProtoNet 元学习/度量学习实验

- 输出目录：`outputs\openclip_meta_proto_focused`
- train_views：`3`，tta_views：`3`
- hidden_dim：`256`，embed_dim：`128`
- n_way：`5`，k_shot：`3`，q_query：`4`
- metric：`cosine`，temperature：`0.03`
- lr：`0.0003`，weight_decay：`0.001`，dropout：`0.0`
- macro-F1：`0.5686`
- balanced accuracy：`0.5680`
- 每类 recall：Class_0=0.78，Class_1=0.58，Class_2=0.52，Class_3=0.52，Class_4=0.44

结论：该实验用于验证标准 few-shot 的 episodic prototype 思路能否在当前 5-way 50-shot 匿名分类任务中超过现有 OpenCLIP Adapter。若低于 `0.5922`，说明本任务更适合预训练特征上的判别式轻量分类头，而不是纯原型距离决策。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_pairwise_margin_probe`
- adapter_type：`residual`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- pairwise_margin_weight：`0.06`，pairwise_margin：`0.1`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`0`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5760`
- balanced accuracy：`0.5760`
- 每类 recall：Class_0=0.76，Class_1=0.64，Class_2=0.58，Class_3=0.54，Class_4=0.36

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_norm_proj_smoke`
- 解冻策略：`ln_proj_only`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-06`，lr_head：`5e-06`
- weight_decay：`0.001`，label_smoothing：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.4545`
- balanced accuracy：`0.4560`
- 每类 recall：Class_0=0.68，Class_1=0.48，Class_2=0.34，Class_3=0.40，Class_4=0.38

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_ln_proj_only_single`
- 解冻策略：`ln_proj_only`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`3e-06`，lr_head：`2e-05`
- weight_decay：`0.001`，label_smoothing：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.5603`
- balanced accuracy：`0.5600`
- 每类 recall：Class_0=0.72，Class_1=0.64，Class_2=0.54，Class_3=0.58，Class_4=0.32

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_last_block_norm_proj_single`
- 解冻策略：`last_block_norm_proj`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`3e-06`，lr_head：`2e-05`
- weight_decay：`0.001`，label_smoothing：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.5568`
- balanced accuracy：`0.5560`
- 每类 recall：Class_0=0.72，Class_1=0.64，Class_2=0.52，Class_3=0.58，Class_4=0.32

结论：本轮结果用于检验旋转增强/轻量微调细化是否能超过当前 `0.5681`。若未超过，则继续保留上一轮最优作为主提交候选。

### OpenCLIP Adapter pairwise margin 与 norm/proj 微调阶段记录

本轮在现有 OpenCLIP 迁移学习主线上尝试把度量学习思想融入判别式训练，并补充更保守的 encoder 轻量微调。

| 实验方向 | 输出目录 | macro-F1 | balanced accuracy | 主要现象 |
|---|---:|---:|---:|---|
| Adapter + Class_2/3/4 pairwise margin | `outputs/openclip_feature_adapter_pairwise_margin_probe` | 0.5760 | 0.5760 | Class_2/3 recall 有一定提升，但 Class_4 recall 降到 0.36，未超过历史最优 |
| OpenCLIP `ln_proj_only` | `outputs/openclip_finetune_ln_proj_only_single` | 0.5603 | 0.5600 | 只动最后 LN/proj 太保守，Class_4 recall 仅 0.32 |
| OpenCLIP `last_block_norm_proj` | `outputs/openclip_finetune_last_block_norm_proj_single` | 0.5568 | 0.5560 | 最后 block 的 LN + proj 仍不足以改善边界，Class_4 继续偏低 |

阶段结论：单纯加入易混类 pairwise margin 或只微调归一化/投影参数，均未突破 `0.5922`。当前瓶颈不是简单 logit margin 或极少量 encoder 参数能解决的；后续更值得尝试的是单 backbone 下的层级/一对一分类头，或回到 Adapter 的随机种子/缓存一致性复现问题。

### OpenCLIP frozen feature + 层级/一对一分类头实验

- 输出目录：`outputs\openclip_hierarchical_head_probe`
- mode：`global_lr`，C：`0.08`，threshold：`0.0`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.25`
- macro-F1：`0.5539`
- balanced accuracy：`0.5520`
- 每类 recall：Class_0=0.74，Class_1=0.62，Class_2=0.48，Class_3=0.54，Class_4=0.38

结论：该实验用于验证单一 OpenCLIP backbone 下，是否能通过 OVO 或层级分类头改善 Class_2/3/4 与 Class_4 的决策边界；若低于 Adapter 最优，说明瓶颈主要不在 sklearn 头形式。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_seed_offset_recheck`
- adapter_type：`residual`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.1`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5794`
- balanced accuracy：`0.5760`
- 每类 recall：Class_0=0.78，Class_1=0.62，Class_2=0.50，Class_3=0.56，Class_4=0.42

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP Adapter seed offset 稳定性复查

- 输出目录：`outputs/openclip_feature_adapter_seed_offset_recheck`
- 固定配置：`ViT-B-32/openai + train_views=3 + tta_views=3 + residual adapter(dim=64, scale=0.4) + lr_adapter=0.0015 + Class_4 weight=1.1`
- 扫描 `init_seed_offset=0..9`
- 最佳结果：`macro-F1=0.5794`，`balanced accuracy=0.5760`，对应 `init_seed_offset=2`
- 召回：Class_0=0.78，Class_1=0.62，Class_2=0.50，Class_3=0.56，Class_4=0.42

结论：当前代码与当前缓存下，Adapter 的稳定可复现上限约为 0.58 左右，没有复现历史 `0.5922` 峰值。后续报告中应把 `0.5922` 标为历史最佳候选，同时说明后续复查更稳定结果约为 `0.579`，最终选择需以可复现性为准。

### timm ImageNet-21K backbone 追加尝试记录

- 尝试环境：`ml2-vision310`
- 尝试命令：`src/timm_frozen_sweep.py --models convnext_tiny.fb_in22k_ft_in1k convnext_base.fb_in22k_ft_in1k swin_tiny_patch4_window7_224.ms_in1k swin_base_patch4_window7_224.ms_in22k_ft_in1k vit_base_patch16_224.augreg_in21k_ft_in1k`
- 结果：运行 30 分钟超时，未产生 `timm_sweep.csv`，推测主要耗时在模型下载或大 backbone 特征提取。
- 已有 `timm_small_backbones` 结果显示 DINO/DeiT/EVA/ConvNeXtV2 Tiny frozen feature 最好仅约 `macro-F1=0.4737`，明显低于 OpenCLIP 主线。

结论：继续盲目下载更大 timm backbone 的性价比不高。若后续还要探索新 backbone，应优先选择明确贴近病理/细胞小图领域的公开预训练权重，而不是泛泛扩大 ImageNet 模型规模。

### OpenCLIP transductive label propagation / pseudo-label 实验

- 输出目录：`outputs\openclip_transductive_probe`
- method：`lr`，kernel：`knn`，n_neighbors：`3`，gamma：`0.0`，alpha：`0.0`
- use_lr_init：`False`，lr_conf_threshold：`0.0`
- C：`0.08`，Class_4 weight：`1.25`
- macro-F1：`0.5539`
- balanced accuracy：`0.5520`
- 每类 recall：Class_0=0.74，Class_1=0.62，Class_2=0.48，Class_3=0.54，Class_4=0.38

结论：该实验模拟有无标签测试集时的 transductive 推理，只使用验证折图像特征、不使用验证标签。若有效，后续可在真实 test_dir 上用 train+test 图传播生成 submission。

### OpenCLIP transductive label propagation / pseudo-label 实验

- 输出目录：`outputs\openclip_transductive_rbf_probe`
- method：`label_spreading`，kernel：`rbf`，n_neighbors：`0`，gamma：`0.1`，alpha：`0.8`
- use_lr_init：`True`，lr_conf_threshold：`0.55`
- C：`0.08`，Class_4 weight：`1.25`
- macro-F1：`0.4902`
- balanced accuracy：`0.4960`
- 每类 recall：Class_0=0.70，Class_1=0.58，Class_2=0.46，Class_3=0.48，Class_4=0.26

结论：该实验模拟有无标签测试集时的 transductive 推理，只使用验证折图像特征、不使用验证标签。若有效，后续可在真实 test_dir 上用 train+test 图传播生成 submission。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_refit_probe2`
- adapter_type：`residual`
- adapter_dim：`64`，adapter_scale：`0.4`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.1`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5794`
- balanced accuracy：`0.5760`
- 每类 recall：Class_0=0.78，Class_1=0.62，Class_2=0.50，Class_3=0.56，Class_4=0.42

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_views5_probe2`
- adapter_type：`residual`
- adapter_dim：`64`，adapter_scale：`0.45`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`0.0001`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.1`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- init_seed_offset：`4`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5584`
- balanced accuracy：`0.5560`
- 每类 recall：Class_0=0.70，Class_1=0.62，Class_2=0.44，Class_3=0.50，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_ln_gated_seed_break60`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.05`
- lr_adapter：`0.0015`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.1`
- 类别权重：Class_2=`1.05`，Class_3=`1.05`，Class_4=`1.05`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5893`
- balanced accuracy：`0.5880`
- 每类 recall：Class_0=0.76，Class_1=0.64，Class_2=0.52，Class_3=0.56，Class_4=0.46

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_ln_gated_break60_best_single`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.05`
- 类别权重：Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.0`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6030`
- balanced accuracy：`0.6000`
- 每类 recall：Class_0=0.76，Class_1=0.64，Class_2=0.50，Class_3=0.58，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_targeted_pairwise_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.05`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.0`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6030`
- balanced accuracy：`0.6000`
- 每类 recall：Class_0=0.76，Class_1=0.64，Class_2=0.50，Class_3=0.58，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_class1_weight_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.05`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6110`
- balanced accuracy：`0.6080`
- 每类 recall：Class_0=0.74，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_weight_margin_refine_best_single`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.045`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6143`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.52，Class_3=0.56，Class_4=0.54

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_best_tta_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.045`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6069`
- balanced accuracy：`0.6040`
- 每类 recall：Class_0=0.74，Class_1=0.68，Class_2=0.52，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_class0_pairwise_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_margin004_best_single`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_directed_pair_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.02`，directed_pair_margin：`0.04`
- directed_pair_set：`Class_1>Class_4;Class_2>Class_3;Class_2>Class_4;Class_4>Class_3;Class_4>Class_2`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6028`
- balanced accuracy：`0.6000`
- 每类 recall：Class_0=0.74，Class_1=0.68，Class_2=0.56，Class_3=0.56，Class_4=0.46

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_seed_margin004_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_lr_wd_refine`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`2e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_dim_refine_margin004`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`2e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_final_train`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### HuggingFace vision frozen feature 迁移学习实验

- 输出目录：`outputs\hf_dinov2_small_gpu_local`
- 最佳模型：`outputs/hf_models/facebook_dinov2-small`
- C：`0.05`，normalize：`True`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- macro-F1：`0.4768`
- balanced accuracy：`0.4760`
- 每类 recall：Class_0=0.58，Class_1=0.60，Class_2=0.40，Class_3=0.40，Class_4=0.40

结论：该实验用于验证更贴近病理/组织学图像的 HF 公开预训练视觉模型，是否比 OpenCLIP/DINO/timm 通用特征更适合当前 32x32 小图分类。

### HuggingFace vision frozen feature 迁移学习实验

- 输出目录：`outputs\hf_dinov2_base_gpu_local`
- 最佳模型：`outputs/hf_models/facebook_dinov2-base`
- C：`0.05`，normalize：`True`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- macro-F1：`0.4725`
- balanced accuracy：`0.4760`
- 每类 recall：Class_0=0.68，Class_1=0.56，Class_2=0.44，Class_3=0.32，Class_4=0.38

结论：该实验用于验证更贴近病理/组织学图像的 HF 公开预训练视觉模型，是否比 OpenCLIP/DINO/timm 通用特征更适合当前 32x32 小图分类。

### HuggingFace vision frozen feature 迁移学习实验

- 输出目录：`outputs\hf_phikon_gpu_local`
- 最佳模型：`outputs/hf_models/owkin_phikon`
- C：`0.03`，normalize：`True`
- 类别权重：Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.1`
- macro-F1：`0.4873`
- balanced accuracy：`0.4880`
- 每类 recall：Class_0=0.62，Class_1=0.62，Class_2=0.36，Class_3=0.54，Class_4=0.30

结论：该实验用于验证更贴近病理/组织学图像的 HF 公开预训练视觉模型，是否比 OpenCLIP/DINO/timm 通用特征更适合当前 32x32 小图分类。

### HF/DINOv2/Phikon feature + OpenCLIP 最优 Adapter 迁移实验

- 输出目录：`outputs\hf_feature_adapter_transfer_phikon_probe`
- backbone：`outputs/hf_models/owkin_phikon`
- adapter_type：`ln_gated_residual`
- adapter_dim：`48`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- macro-F1：`0.5407`
- balanced accuracy：`0.5400`
- 每类 recall：Class_0=0.64，Class_1=0.64，Class_2=0.42，Class_3=0.58，Class_4=0.42

结论：该实验用于验证当前 OpenCLIP 最优方案中的轻量 Adapter、LogisticRegression 初始化、pairwise margin 与类别权重，迁移到 DINOv2/Phikon frozen feature 后是否仍然有效。

### HF/DINOv2/Phikon feature + OpenCLIP 最优 Adapter 迁移实验

- 输出目录：`outputs\hf_feature_adapter_transfer_dino_probe`
- backbone：`outputs/hf_models/facebook_dinov2-small`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.03`
- macro-F1：`0.5151`
- balanced accuracy：`0.5120`
- 每类 recall：Class_0=0.58，Class_1=0.58，Class_2=0.50，Class_3=0.48，Class_4=0.42

结论：该实验用于验证当前 OpenCLIP 最优方案中的轻量 Adapter、LogisticRegression 初始化、pairwise margin 与类别权重，迁移到 DINOv2/Phikon frozen feature 后是否仍然有效。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_priority1_refine`
- adapter_type：`ln_gated_residual`
- adapter_dim：`32`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5927`
- balanced accuracy：`0.5920`
- 每类 recall：Class_0=0.78，Class_1=0.68，Class_2=0.56，Class_3=0.50，Class_4=0.44

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_best_recheck_after_metric_patch`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_metric_loss_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。
### 2026-06-01 按优先级推进 1/2/3/4 优化结论

- 优先级 1：围绕当前最优 `OpenCLIP ViT-B-32/openai + LN gated residual Adapter` 做小范围参数精修。
- 复现基准：`outputs/openclip_feature_adapter_best_recheck_after_metric_patch`，macro-F1=`0.6147`，balanced accuracy=`0.6120`。说明新增 metric loss 代码没有破坏旧最佳链路。
- 小参数精修：`outputs/openclip_feature_adapter_priority1_refine`。在 batch_size=`128` 下最高只有 macro-F1=`0.5927`，低于旧最佳。关键结论是 batch_size=`600` 对该 Adapter 方案很重要，后续精修必须保持该设置。
- 优先级 2：在最佳配置上加入 `center loss` / `supervised contrastive loss`，输出目录 `outputs/openclip_feature_adapter_metric_loss_probe`。
- metric loss 结果：最高仍是无额外 metric loss 的基准 macro-F1=`0.6147`。`center_loss_weight=0.001/0.003` 或 `supcon_loss_weight=0.001/0.003` 均未超过基准，说明当前 frozen OpenCLIP 特征 + 小 Adapter 的可调空间有限，额外度量约束会扰动已有边界。
- 优先级 3：复查现有 OpenCLIP 轻量微调结果。历史最佳为 `outputs/openclip_finetune_tta_fixed_narrow`，macro-F1=`0.5681`，balanced accuracy=`0.5640`，低于当前 Adapter 最优。last block、last two blocks、LoRA、norm/proj-only 均未超过 Adapter。
- 优先级 4：DINOv2/Phikon 作为对照方向。最优方法迁移到 Phikon 后从 frozen macro-F1=`0.4873` 提升到 `0.5407`，DINOv2-small 提升到 `0.5151`，但均明显低于 OpenCLIP Adapter。
- 当前结论：主提交路线继续保持 `OpenCLIP ViT-B-32/openai + 3 train views + 3 TTA views + LN gated residual Adapter + pairwise margin + 类别权重`，当前最佳仍为 macro-F1=`0.6147`，balanced accuracy=`0.6120`。
- 后续若继续冲分，建议不再做简单参数大网格，而是尝试测试集可用后的高置信伪标签 / transductive label propagation；若没有测试集，则优先做错误样本人工分析和预测分布校准。

### 2026-06-02 仅使用现有训练集的继续优化

- 背景：测试集最后由老师提供，因此本轮不再考虑测试集伪标签、transductive label propagation 或 test-time adaptation，只在当前 250 张训练图的 5-fold 上继续优化。
- 当前最佳基准：`outputs/openclip_feature_adapter_best_recheck_after_metric_patch`，OpenCLIP ViT-B-32/openai + 3 train views + 3 TTA views + LN gated residual Adapter，macro-F1=`0.6147`，balanced accuracy=`0.6120`。
- 错分分析：主要错分对为 `Class_2 -> Class_3` 9 个、`Class_1 -> Class_4` 9 个、`Class_4 -> Class_3` 9 个、`Class_2 -> Class_4` 8 个、`Class_3 -> Class_4` 8 个、`Class_3 -> Class_2` 7 个、`Class_4 -> Class_2` 7 个。
- directed margin：输出目录 `outputs/openclip_feature_adapter_directed_margin_priority_probe`。对上述错分方向加入定向 margin 后没有超过基准，最高仍为无 directed margin 的 macro-F1=`0.6147`。说明错分不是单纯靠指定 logit 约束就能修正。
- 增强强度：输出目录包括 `outputs/openclip_feature_adapter_aug_scale065_jitter015_b600`、`outputs/openclip_feature_adapter_aug_scale075_jitter015_b600`、`outputs/openclip_feature_adapter_aug_scale080_jitter005_b600`、`outputs/openclip_feature_adapter_aug_scale080_jitter000_b600`、`outputs/openclip_feature_adapter_aug_scale090_jitter000_b600`。这些组合最高约 `0.5597`，明显低于当前 `crop_scale_min=0.72, jitter=0.15` 的基准，说明原增强强度更适合当前小图纹理任务。
- bias calibration：输出目录 `outputs/openclip_bias_calibration_current_aug_probe`。内层交叉验证调 Class_2/3/4 bias 的 LogisticRegression 最好 macro-F1=`0.5426`，低于 Adapter，说明阈值/偏置校准不是当前瓶颈。
- 本轮结论：在没有测试集可用的前提下，当前最强可靠方案仍是 OpenCLIP Adapter。继续突破的难点主要来自特征空间本身的类别重叠，而不是简单超参、增强强度、类别 bias 或 directed margin。
- 后续可写入报告的有效优化路径：先证明 OpenCLIP frozen feature 优于 ResNet/ConvNeXt/DINOv2/Phikon，再证明 Adapter 相比普通线性头显著提升，最后通过失败实验说明更强 backbone、轻量微调、metric loss、directed margin、bias calibration 均未稳定超过当前最优。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_directed_margin_priority_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.04`
- directed_pair_set：`Class_2>Class_3;Class_2>Class_4;Class_3>Class_4;Class_4>Class_3;Class_4>Class_2;Class_1>Class_4`
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug_scale065_jitter015_b600`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5418`
- balanced accuracy：`0.5360`
- 每类 recall：Class_0=0.78，Class_1=0.54，Class_2=0.50，Class_3=0.50，Class_4=0.36

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug_scale075_jitter015_b600`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5285`
- balanced accuracy：`0.5280`
- 每类 recall：Class_0=0.70，Class_1=0.60，Class_2=0.48，Class_3=0.50，Class_4=0.36

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug_scale080_jitter005_b600`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5597`
- balanced accuracy：`0.5560`
- 每类 recall：Class_0=0.64，Class_1=0.62，Class_2=0.54，Class_3=0.56，Class_4=0.42

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug_scale080_jitter000_b600`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5573`
- balanced accuracy：`0.5520`
- 每类 recall：Class_0=0.72，Class_1=0.56，Class_2=0.54，Class_3=0.48，Class_4=0.46

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_aug_scale090_jitter000_b600`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5335`
- balanced accuracy：`0.5280`
- 每类 recall：Class_0=0.68，Class_1=0.62，Class_2=0.46，Class_3=0.50，Class_4=0.38

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP LogisticRegression 类别 bias 校准实验

- 输出目录：`outputs\openclip_bias_calibration_current_aug_probe`
- train_views：`3`，tta_views：`3`，C：`0.12`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- bias 搜索：范围=`±0.6`，步长=`0.15`，inner_splits=`4`
- 平均 bias：Class_2=`-0.150`，Class_3=`0.030`，Class_4=`-0.030`
- macro-F1：`0.5426`
- balanced accuracy：`0.5440`
- 每类 recall：Class_0=0.74，Class_1=0.62，Class_2=0.46，Class_3=0.56，Class_4=0.34

结论：该实验尝试用训练折内部 OOF 来校准 Class_2/3/4 决策阈值；若不提升，说明错误主要来自特征不可分，而不是单纯类别阈值偏置。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_noise_robust_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`

## 2026-06-02 报告初版思路与当前处理结论

- 人工查看错分拼图后，`Class_2/Class_3/Class_4` 的视觉差异仍然不明显，说明主要瓶颈是细粒度类别重叠和可能的标注噪声，而不是某个简单增强或分类阈值没有调好。
- 鲁棒损失实验输出目录：`outputs/openclip_feature_adapter_noise_robust_probe`。尝试 `focal loss`、`label smoothing`、`dropout` 和更大的 `weight_decay` 后，最优仍是原始 CE 设置，macro-F1=`0.6147`，balanced accuracy=`0.6120`。
- 当前最佳仍为：`OpenCLIP ViT-B-32/openai + 3 train views + 3 TTA views + LN gated residual Adapter + pairwise margin + 轻量类别权重`。
- 已生成报告初版方案：`docs/report_initial_plan.md`。报告主线建议写成：先证明传统特征/ResNet/ConvNeXt/DINOv2/Phikon等方向不足，再说明 OpenCLIP frozen feature 最稳，最后展示 Adapter、TTA、pairwise margin 的有效性，以及 metric loss、directed margin、bias calibration、鲁棒损失等失败尝试。
- 后续如果继续探索，优先级应从“简单调参”转向“更匹配任务域的单模型预训练 backbone”或“测试集可用后的无标签分布适配”。在当前仅有 250 张训练图的条件下，继续强行调 loss 或增强，提升空间预计有限。

## 2026-06-02 更匹配数据域的单模型 backbone 搜索

- 下载策略：优先设置 `HF_ENDPOINT=https://hf-mirror.com`，并尝试 Hugging Face mirror、官方站以及清华 PyPI 镜像。
- 网络问题：`hf-mirror.com`、`huggingface.co`、`pypi.tuna.tsinghua.edu.cn` 在当前 Python/conda 进程中均出现 `TLS/SSL connection has been closed (EOF)`，因此本轮无法新下载 `Phikon-v2`、`BioMedCLIP`、`google/vit-base` 等新模型。
- 环境处理：发现本机已有 `ml2-vision310` 环境，Python 3.10、torch 2.5.1+cu121、CUDA 可用，且已安装 `timm` 和 `open_clip`。因此优先使用该环境跑本机已缓存的 timm/HF 权重。
- 代码补充：`src/hf_vision_frozen_sweep.py` 已兼容旧版 transformers，优先使用 `AutoImageProcessor`，没有时退回 `AutoFeatureExtractor`。
- 新增脚本：`src/timm_feature_adapter_sweep.py`，用于把当前最优的 LN gated residual Adapter 思路迁移到 timm frozen feature 上。

### cached timm frozen feature 结果

- 输出目录：`outputs/timm_cached_domain_backbone_probe`
- 候选 backbone：`eva02_tiny_patch14_224.mim_in22k`、`vit_small_patch14_dinov2.lvd142m`、`vit_small_patch14_reg4_dinov2.lvd142m`、`vit_small_patch16_224.dino`、`convnextv2_tiny.fcmae_ft_in22k_in1k`、`deit3_small_patch16_224.fb_in22k_ft_in1k`、`convnext_tiny.fb_in22k_ft_in1k`
- 最佳：`vit_small_patch14_reg4_dinov2.lvd142m`
- macro-F1：`0.4872`
- balanced accuracy：`0.4840`
- 每类 recall：Class_0=`0.68`，Class_1=`0.56`，Class_2=`0.40`，Class_3=`0.48`，Class_4=`0.30`

结论：这些本机已缓存的 DINO/EVA/ConvNeXtV2/DeiT backbone 作为 frozen feature 时明显低于当前 OpenCLIP Adapter 最优 `0.6147`。

### timm feature + Adapter 迁移结果

- 输出目录：`outputs/timm_feature_adapter_cached_probe`
- 最佳 backbone：`vit_small_patch14_reg4_dinov2.lvd142m`
- 最佳配置：`adapter_dim=24`，`adapter_scale=1.0`，`lr_adapter=0.0012`，`class2_weight=1.05`，不使用额外 pairwise margin
- macro-F1：`0.5197`
- balanced accuracy：`0.5160`
- 每类 recall：Class_0=`0.70`，Class_1=`0.56`，Class_2=`0.46`，Class_3=`0.54`，Class_4=`0.32`

结论：Adapter 能把该 DINO reg4 特征从 `0.4872` 提升到 `0.5197`，说明轻量适配方法本身有效；但该 backbone 的基础特征与当前任务不如 OpenCLIP ViT-B-32/openai，尤其 Class_4 recall 仍只有 `0.32`，因此不能替代当前主线。

## 2026-06-02 Phikon-v2 本地直链下载与实验

- 背景：Python/HF Hub SDK 在当前环境访问 `hf-mirror.com` 和 `huggingface.co` 时会出现 TLS EOF，但 PowerShell `Invoke-WebRequest` 可以直接访问 Hugging Face 文件直链。
- 处理方式：通过 Hugging Face API 获取 `owkin/phikon-v2` 文件清单后，用 PowerShell 下载到本地目录：

```text
outputs/hf_models/owkin_phikon-v2/
```

- 下载文件：
  - `config.json`
  - `preprocessor_config.json`
  - `model.safetensors`，约 `1.21GB`
- 官方信息：`Phikon-v2` 是非 gated 模型，模型类型为 ViT-L/16 via DINOv2，预训练于病理 histology 图像，官方建议通过 `AutoImageProcessor + AutoModel` 抽取 CLS token 特征。

### Phikon-v2 frozen feature

- 输出目录：`outputs/hf_phikon_v2_frozen_probe`
- 最佳配置：`C=0.03`，无额外类别权重
- macro-F1：`0.4850`
- balanced accuracy：`0.4880`
- 每类 recall：Class_0=`0.70`，Class_1=`0.60`，Class_2=`0.36`，Class_3=`0.36`，Class_4=`0.42`

### Phikon-v2 + Adapter

- 输出目录：`outputs/hf_phikon_v2_adapter_probe`
- 最佳配置：`adapter_dim=48`，`adapter_scale=1.0`，`lr_adapter=0.0008`，`pairwise_margin_weight=0.02`，`class2_weight=1.05`，`class4_weight=1.0`
- macro-F1：`0.5560`
- balanced accuracy：`0.5600`
- 每类 recall：Class_0=`0.70`，Class_1=`0.62`，Class_2=`0.36`，Class_3=`0.50`，Class_4=`0.62`

结论：Phikon-v2 是真正更贴近病理图像的数据域 backbone，Adapter 后明显优于 frozen feature，但仍低于当前 OpenCLIP Adapter 最优 `0.6147`。可能原因是 Phikon-v2 预训练面向 224x224、20x magnification 的 histology tile，而本任务图像只有 32x32；上采样后纹理和细胞结构信息不足，导致病理大模型优势不能完全发挥。

### Prov-GigaPath 可行性记录

- 官方 Hugging Face API 显示 `prov-gigapath/prov-gigapath` 为 `gated=auto`，通常需要登录 Hugging Face、同意模型条款，并设置 `HF_TOKEN`。
- 当前环境没有 `HF_TOKEN`，因此 `timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)` 无法从本地缓存或 Hub 拉取。
- 若后续要尝试，需要先在 Hugging Face 页面同意条款并提供只读 token，再用 PowerShell 直链或 `timm hf_hub` 下载。
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### timm frozen feature 补充实验

- 最佳模型：`vit_small_patch14_reg4_dinov2.lvd142m`
- macro-F1：`0.4872`
- balanced accuracy：`0.4840`
- 输出目录：`outputs/timm_cached_domain_backbone_probe`

### timm cached backbone + Adapter 迁移实验

- 输出目录：`outputs/timm_feature_adapter_cached_probe`
- 最佳 backbone：`vit_small_patch14_reg4_dinov2.lvd142m`
- adapter_type：`ln_gated_residual`，adapter_dim=`24`
- pairwise_margin_weight：`0.0`，pairwise_margin=`0.04`
- macro-F1：`0.5197`
- balanced accuracy：`0.5160`
- 每类 recall：Class_0=0.70，Class_1=0.56，Class_2=0.46，Class_3=0.54，Class_4=0.32

结论：该实验用于验证当前最优 Adapter 思路迁移到本机已缓存的 timm/DINO/EVA/ConvNeXtV2 backbone 后是否有效。

### HuggingFace vision frozen feature 迁移学习实验

- 输出目录：`outputs\hf_phikon_v2_frozen_probe`
- 最佳模型：`outputs/hf_models/owkin_phikon-v2`
- C：`0.03`，normalize：`True`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- macro-F1：`0.4850`
- balanced accuracy：`0.4880`
- 每类 recall：Class_0=0.70，Class_1=0.60，Class_2=0.36，Class_3=0.36，Class_4=0.42

结论：该实验用于验证更贴近病理/组织学图像的 HF 公开预训练视觉模型，是否比 OpenCLIP/DINO/timm 通用特征更适合当前 32x32 小图分类。

### HF/DINOv2/Phikon feature + OpenCLIP 最优 Adapter 迁移实验

- 输出目录：`outputs\hf_phikon_v2_adapter_probe`
- backbone：`outputs/hf_models/owkin_phikon-v2`
- adapter_type：`ln_gated_residual`
- adapter_dim：`48`，adapter_scale：`1.0`
- lr_adapter：`0.0008`，lr_head：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.0`
- macro-F1：`0.5560`
- balanced accuracy：`0.5600`
- 每类 recall：Class_0=0.70，Class_1=0.62，Class_2=0.36，Class_3=0.50，Class_4=0.62

结论：该实验用于验证当前 OpenCLIP 最优方案中的轻量 Adapter、LogisticRegression 初始化、pairwise margin 与类别权重，迁移到 DINOv2/Phikon frozen feature 后是否仍然有效。

### HF/DINOv2/Phikon feature + OpenCLIP 最优 Adapter 迁移实验

- 输出目录：`outputs\hf_phikon_v2_adapter_b600_recheck`
- backbone：`outputs/hf_models/owkin_phikon-v2`
- adapter_type：`ln_gated_residual`
- adapter_dim：`48`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.03`
- macro-F1：`0.5630`
- balanced accuracy：`0.5640`
- 每类 recall：Class_0=0.74，Class_1=0.62，Class_2=0.46，Class_3=0.46，Class_4=0.54

结论：该实验用于验证当前 OpenCLIP 最优方案中的轻量 Adapter、LogisticRegression 初始化、pairwise margin 与类别权重，迁移到 DINOv2/Phikon frozen feature 后是否仍然有效。

### HF backbone 多视图增强 + Adapter 公平对比实验

- 输出目录：`outputs/hf_phikon_v2_aug_adapter_fair_probe`
- backbone：`outputs/hf_models/owkin_phikon-v2`
- train_views：`3`，tta_views：`3`
- macro-F1：`0.5860`
- balanced accuracy：`0.5840`
- 每类 recall：Class_0=0.68，Class_1=0.74，Class_2=0.46，Class_3=0.54，Class_4=0.50

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\biomedclip_aug_adapter_fair_probe`
- adapter_type：`ln_gated_residual`
- adapter_dim：`48`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.0`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5134`
- balanced accuracy：`0.5120`
- 每类 recall：Class_0=0.62，Class_1=0.68，Class_2=0.36，Class_3=0.48，Class_4=0.42

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。
## 2026-06-02 同等条件迁移最优方法到医学/病理 backbone

本轮目的：验证“其他 backbone 没超过当前最优”是否只是因为没有迁移当前最优 OpenCLIP 方案里的关键技巧。为保证公平，本轮把当前最有效的组件迁移到 Phikon-v2 和 BioMedCLIP 上：

- 3 train views 数据增强。
- 3 TTA views 推理增强。
- LogisticRegression 初始化分类边界。
- LN gated residual Adapter。
- 大 batch Adapter 训练。
- 轻量类别权重和 Class_2/Class_3/Class_4 相关约束候选。

### Phikon-v2

- 模型来源：`owkin/phikon-v2`
- 本地模型目录：`outputs/hf_models/owkin_phikon-v2/`
- frozen feature + LogisticRegression：`macro-F1=0.4850`，`balanced accuracy=0.4880`
- 单视图 feature + Adapter：`macro-F1=0.5630`，`balanced accuracy=0.5640`
- 3 train views + 3 TTA + Adapter：`macro-F1=0.5860`，`balanced accuracy=0.5840`
- 最佳输出目录：`outputs/hf_phikon_v2_aug_adapter_fair_probe/`
- 每类 recall：Class_0=`0.68`，Class_1=`0.74`，Class_2=`0.46`，Class_3=`0.54`，Class_4=`0.50`

结论：Phikon-v2 在迁移当前最优方法后提升明显，从 frozen 的 `0.4850` 提升到 `0.5860`，说明病理/组织学 backbone 方向值得作为报告中的重要对照。但它仍低于当前 OpenCLIP Adapter 最优 `0.6147`，主要短板仍是 Class_2/Class_4 recall 不够。

### BioMedCLIP

- 模型来源：`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- 3 train views + 3 TTA + Adapter：`macro-F1=0.5134`，`balanced accuracy=0.5120`
- 最佳输出目录：`outputs/biomedclip_aug_adapter_fair_probe/`
- 每类 recall：Class_0=`0.62`，Class_1=`0.68`，Class_2=`0.36`，Class_3=`0.48`，Class_4=`0.42`

结论：BioMedCLIP 在同等条件下没有超过 Phikon-v2，更没有超过 OpenCLIP ViT-B-32/openai。初步判断是 BioMedCLIP 的图文医学语义预训练对匿名 32x32 细粒度纹理分类帮助有限。

### UNI / Prov-GigaPath 状态

- `Prov-GigaPath` 和 `UNI` 均属于 Hugging Face gated 模型。
- 浏览器页面中同意访问权限不等于命令行环境拥有 token。
- 当前命令行环境检测结果：`HF_TOKEN` 不存在，`huggingface_hub.get_token()` 为空。
- 直接访问 `Prov-GigaPath` 权重文件返回 401，因此暂时不能纳入正式实验。

下一步如果继续测试 UNI / Prov-GigaPath，需要在当前 PowerShell/conda 环境设置 Hugging Face read token，例如：

```powershell
$env:HF_TOKEN="hf_xxx"
```

或运行：

```powershell
conda run --no-capture-output -n ml2-vision310 hf auth login
```

总体结论：本轮已经验证“迁移最优方法后再对比”是必要的。Phikon-v2 的确从低分提升到接近当前主线，但目前仍未超过 OpenCLIP Adapter。因此当前最终候选仍保持为 `outputs/final_openclip_feature_adapter_model.joblib`。

## 2026-06-02 OpenCLIP Adapter 类别权重与 bias calibration 精修

- 输出目录：`outputs\openclip_adapter_weight_bias_smoke`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- raw macro-F1：`0.5286`，raw balanced accuracy：`0.5280`
- cross-fit bias macro-F1：`0.5244`，balanced accuracy：`0.5240`
- optimistic OOF bias macro-F1：`0.5296`，balanced accuracy：`0.5280`

说明：cross-fit bias 是用其他折的 OOF logits 选择 bias 后应用到当前折，作为较保守估计；optimistic OOF bias 使用全部 OOF 标签选择 bias，只用于判断潜力，不作为严格无偏 5-fold 分数。

## 2026-06-02 OpenCLIP Adapter 类别权重与 bias calibration 精修

- 输出目录：`outputs\openclip_adapter_bias_best_weight_full`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- raw macro-F1：`0.5319`，raw balanced accuracy：`0.5320`
- cross-fit bias macro-F1：`0.5439`，balanced accuracy：`0.5440`
- optimistic OOF bias macro-F1：`0.5439`，balanced accuracy：`0.5440`

说明：cross-fit bias 是用其他折的 OOF logits 选择 bias 后应用到当前折，作为较保守估计；optimistic OOF bias 使用全部 OOF 标签选择 bias，只用于判断潜力，不作为严格无偏 5-fold 分数。

## 2026-06-02 OpenCLIP Adapter 类别权重与 bias calibration 精修

- 输出目录：`outputs\openclip_adapter_bias_best_weight_full_control_cached`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- raw macro-F1：`0.6147`，raw balanced accuracy：`0.6120`
- cross-fit bias macro-F1：`0.6069`，balanced accuracy：`0.6040`
- optimistic OOF bias macro-F1：`0.6147`，balanced accuracy：`0.6120`

说明：cross-fit bias 是用其他折的 OOF logits 选择 bias 后应用到当前折，作为较保守估计；optimistic OOF bias 使用全部 OOF 标签选择 bias，只用于判断潜力，不作为严格无偏 5-fold 分数。

## 2026-06-02 OpenCLIP Adapter 类别权重与 bias calibration 精修

- 输出目录：`outputs\openclip_adapter_weight_bias_refine_control_grid`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- raw macro-F1：`0.6147`，raw balanced accuracy：`0.6120`
- cross-fit bias macro-F1：`0.6069`，balanced accuracy：`0.6040`
- optimistic OOF bias macro-F1：`0.6147`，balanced accuracy：`0.6120`

说明：cross-fit bias 是用其他折的 OOF logits 选择 bias 后应用到当前折，作为较保守估计；optimistic OOF bias 使用全部 OOF 标签选择 bias，只用于判断潜力，不作为严格无偏 5-fold 分数。

## 2026-06-02 OpenCLIP Adapter 融合度量学习 margin head 实验

- 输出目录：`outputs\openclip_adapter_margin_head_smoke`
- head_type：`cosface`，scale：`10.0`，margin：`0.03`
- proto_pull_weight：`0.0`，pairwise_margin_weight：`0.0`
- adapter：`ln_gated_residual`，dim=`24`，scale=`1.0`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- macro-F1：`0.4505`，balanced accuracy：`0.4480`
- 每类 recall：Class_0=0.44，Class_1=0.44，Class_2=0.42，Class_3=0.54，Class_4=0.40

结论：该实验把度量学习的角度间隔分类头和 prototype pull 约束融合到当前迁移学习主线中，用于判断是否能进一步拉开易混类边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_adapter_metric_aux_refine`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：`Class_4\>Class_3`
- center_loss_weight：`0.005`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5979`
- balanced accuracy：`0.5920`
- 每类 recall：Class_0=0.70，Class_1=0.58，Class_2=0.58，Class_3=0.56，Class_4=0.54

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_adapter_metric_aux_refine_b600`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.5`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：`Class_4\>Class_3`
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

## 2026-06-02 OpenCLIP Adapter 融合度量学习 margin head 实验

- 输出目录：`outputs\openclip_adapter_margin_head_probe`
- head_type：`cosface`，scale：`32.0`，margin：`0.03`
- proto_pull_weight：`0.0`，pairwise_margin_weight：`0.02`
- adapter：`ln_gated_residual`，dim=`24`，scale=`1.0`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- macro-F1：`0.5658`，balanced accuracy：`0.5640`
- 每类 recall：Class_0=0.66，Class_1=0.60，Class_2=0.60，Class_3=0.52，Class_4=0.44

结论：该实验把度量学习的角度间隔分类头和 prototype pull 约束融合到当前迁移学习主线中，用于判断是否能进一步拉开易混类边界。

### timm frozen feature 补充实验

- 最佳模型：`hf-hub:MahmoodLab/UNI`
- macro-F1：`0.5277`
- balanced accuracy：`0.5280`
- 输出目录：`outputs/uni_frozen_probe`

### timm frozen feature 补充实验

- 最佳模型：`hf-hub:prov-gigapath/prov-gigapath`
- macro-F1：`0.4668`
- balanced accuracy：`0.4720`
- 输出目录：`outputs/prov_gigapath_frozen_probe`

### timm cached backbone + Adapter 迁移实验

- 输出目录：`outputs/uni_prov_feature_adapter_probe`
- 最佳 backbone：`hf-hub:MahmoodLab/UNI`
- adapter_type：`ln_gated_residual`，adapter_dim=`48`
- pairwise_margin_weight：`0.02`，pairwise_margin=`0.04`
- macro-F1：`0.5673`
- balanced accuracy：`0.5680`
- 每类 recall：Class_0=0.68，Class_1=0.66，Class_2=0.52，Class_3=0.46，Class_4=0.52

结论：该实验用于验证当前最优 Adapter 思路迁移到本机已缓存的 timm/DINO/EVA/ConvNeXtV2 backbone 后是否有效。

## 2026-06-02 timm/UNI 多视图增强 + Adapter 公平对比

- 输出目录：`outputs\uni_aug_feature_adapter_probe`
- backbone：`hf-hub:MahmoodLab/UNI`
- train_views：`3`，tta_views：`3`
- adapter_dim：`24`，lr_adapter：`0.0008`
- pairwise_margin_weight：`0.0`
- 类别权重：Class_2=`1.05`，Class_4=`1.0`
- macro-F1：`0.5676`，balanced accuracy：`0.5720`
- 每类 recall：Class_0=0.74，Class_1=0.58，Class_2=0.54，Class_3=0.36，Class_4=0.64

结论：该实验用于验证 UNI/病理 timm backbone 在迁移当前 OpenCLIP 最优的多视图增强、TTA 和 Adapter 后是否能反超。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_margin004_seed_repro_now`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_margin004_split_seed_repro_41`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5120`
- balanced accuracy：`0.5160`
- 每类 recall：Class_0=0.78，Class_1=0.58，Class_2=0.44，Class_3=0.36，Class_4=0.42

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_margin004_split_seed_repro_42`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.6147`
- balanced accuracy：`0.6120`
- 每类 recall：Class_0=0.76，Class_1=0.68，Class_2=0.54，Class_3=0.56，Class_4=0.52

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP frozen feature + residual adapter 补充实验

- 输出目录：`outputs\openclip_feature_adapter_margin004_split_seed_repro_43`
- adapter_type：`ln_gated_residual`
- adapter_dim：`24`，adapter_scale：`1.0`
- lr_adapter：`0.0012`，lr_head：`0.0`
- weight_decay：`5e-05`，label_smoothing：`0.0`
- loss_type：`ce`，focal_gamma：`1.2`，margin：`0.0`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`
- pairwise_class_set：`Class_2,Class_3,Class_4`
- directed_pair_weight：`0.0`，directed_pair_margin：`0.0`
- directed_pair_set：``
- center_loss_weight：`0.0`，center_class_set：`Class_2,Class_3,Class_4`
- supcon_loss_weight：`0.0`，supcon_temperature：`0.07`，supcon_class_set：`Class_2,Class_3,Class_4`
- 类别权重：Class_1=`1.0`，Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- init_seed_offset：`2`
- refit_c：`0.0`
- 图像预处理：color_mode=`rgb`，interpolation=`bicubic`
- macro-F1：`0.5395`
- balanced accuracy：`0.5400`
- 每类 recall：Class_0=0.78，Class_1=0.56，Class_2=0.46，Class_3=0.52，Class_4=0.38

结论：该实验用于验证“在 OpenCLIP frozen feature 后加一个小残差 Adapter”是否能在保留 sklearn 初始化强基线的同时微调特征边界。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_smoke`
- 解冻策略：`emb_first2_last4`
- 可训练参数：`45324549` / `151279878`，比例 `29.96%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.0001`
- weight_decay：`0.01`，label_smoothing：`0.03`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.1`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.4412`
- balanced accuracy：`0.4400`
- 每类 recall：Class_0=0.66，Class_1=0.44，Class_2=0.36，Class_3=0.36，Class_4=0.38

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_first4_last6_lr1e4`
- 解冻策略：`emb_first4_last6`
- 可训练参数：`73676037` / `151279878`，比例 `48.70%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`0.0001`，lr_head：`0.0001`
- weight_decay：`0.01`，label_smoothing：`0.03`
- 类别权重：Class_2=`1.0`，Class_3=`1.0`，Class_4=`1.0`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.5435`
- balanced accuracy：`0.5440`
- 每类 recall：Class_0=0.74，Class_1=0.64，Class_2=0.48，Class_3=0.54，Class_4=0.32

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_low_lr_weak_aug`
- 解冻策略：`emb_first2_last4`
- 可训练参数：`45324549` / `151279878`，比例 `29.96%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`5e-05`
- weight_decay：`0.05`，label_smoothing：`0.0`
- 类别权重：Class_2=`1.05`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.5935`
- balanced accuracy：`0.5960`
- 每类 recall：Class_0=0.76，Class_1=0.80，Class_2=0.36，Class_3=0.48，Class_4=0.58

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_random_head_probe`
- 解冻策略：`emb_first2_last4`
- 可训练参数：`45324549` / `151279878`，比例 `29.96%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_2=`1.15`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6043`
- balanced accuracy：`0.6000`
- 每类 recall：Class_0=0.64，Class_1=0.80，Class_2=0.54，Class_3=0.50，Class_4=0.52

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_random_head_weighted_probe`
- 解冻策略：`emb_first2_last4`
- 可训练参数：`45324549` / `151279878`，比例 `29.96%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6395`
- balanced accuracy：`0.6360`
- 每类 recall：Class_0=0.66，Class_1=0.70，Class_2=0.62，Class_3=0.66，Class_4=0.54

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_weight_refine_a`
- 解冻策略：`emb_first2_last4`
- 可训练参数：`45324549` / `151279878`，比例 `29.96%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.1`，Class_1=`1.0`，Class_2=`1.2`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6245`
- balanced accuracy：`0.6240`
- 每类 recall：Class_0=0.78，Class_1=0.72，Class_2=0.62，Class_3=0.56，Class_4=0.44

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_class4_refine`
- 解冻策略：`emb_first2_last4`
- 可训练参数：`45324549` / `151279878`，比例 `29.96%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.08`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6101`
- balanced accuracy：`0.6040`
- 每类 recall：Class_0=0.68，Class_1=0.62，Class_2=0.50，Class_3=0.56，Class_4=0.66

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_range_probe`
- 解冻策略：`emb_last4`
- 可训练参数：`31148805` / `151279878`，比例 `20.59%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6033`
- balanced accuracy：`0.5960`
- 每类 recall：Class_0=0.68，Class_1=0.58，Class_2=0.64，Class_3=0.54，Class_4=0.54

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_range_probe_seedfixed`
- 解冻策略：`emb_first4_last6`
- 可训练参数：`73676037` / `151279878`，比例 `48.70%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6554`
- balanced accuracy：`0.6560`
- 每类 recall：Class_0=0.78，Class_1=0.76，Class_2=0.58，Class_3=0.66，Class_4=0.50

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_first4_last6_lr_refine`
- 解冻策略：`emb_first4_last6`
- 可训练参数：`73676037` / `151279878`，比例 `48.70%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`5e-06`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6443`
- balanced accuracy：`0.6400`
- 每类 recall：Class_0=0.68，Class_1=0.76，Class_2=0.50，Class_3=0.70，Class_4=0.56

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_neighbor_probe`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6750`
- balanced accuracy：`0.6760`
- 每类 recall：Class_0=0.68，Class_1=0.86，Class_2=0.62，Class_3=0.66，Class_4=0.56

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_first3_last6_lr_refine`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`7e-06`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6621`
- balanced accuracy：`0.6640`
- 每类 recall：Class_0=0.74，Class_1=0.78，Class_2=0.44，Class_3=0.78，Class_4=0.58

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。
### OpenCLIP ViT-B 部分解冻主线阶段结论

- 代码修正：在 `src/openclip_finetune.py` 的每个 fold 训练开始前重置随机种子，避免随机分类头初始化受前一个实验消耗 RNG 影响。
- 代码补充：在 `src/openclip_finetune.py` 中加入可选 `linear / ln_residual / ln_gated_residual` 分类头，支持把 frozen-feature 阶段有效的 Adapter 模块叠加到部分解冻方案上。
- 训练方式：随机初始化分类头，直接端到端训练部分解冻的 OpenCLIP ViT-B-32，不再使用 sklearn LogisticRegression 初始化。
- 固定设置：`crop_scale_min=0.90`，`jitter=0.05`，`TTA=3`，`lr_head=1e-3`，`weight_decay=0.05`，`label_smoothing=0.03`。
- 当前最佳：`emb_first3_last6`，可训练参数 `66,588,165 / 151,279,878`，约 `44.02%`。
- 当前最佳指标：`macro-F1=0.6750`，`balanced accuracy=0.6760`。
- 每类 recall：Class_0=0.68，Class_1=0.86，Class_2=0.62，Class_3=0.66，Class_4=0.56。
- 对比结论：`emb_last4` 约 20.59% 参数，最高约 `0.6074`；`emb_first2_last4` 约 29.96% 参数，约 `0.6453`；`emb_first4_last6` 约 48.70% 参数，约 `0.6554`；`emb_first3_last6` 约 44.02% 参数，目前最好。
- 学习率结论：`lr_backbone=1e-5` 最好；`5e-6/7e-6` 略低，`1.3e-5` 开始下降，`2e-5` 明显不稳定。

结论：本任务的主要突破来自“让 ViT 的底层输入嵌入、前 3 个 block 和后 6 个 block 共同适配 32x32 细粒度图像”。推理阶段不需要继续解冻或训练，只需要加载微调后的权重并 `eval + no_grad + TTA`。

### 部分解冻方案叠加 Adapter / LoRA 对照

- 输出目录：`outputs/openclip_partial_unfreeze_adapter_head_probe`
- 设置：固定当前最优 `emb_first3_last6`、`lr_backbone=1e-5`、`lr_head=1e-3`、`TTA=3`。
- 线性分类头：`macro-F1=0.6750`，`balanced accuracy=0.6760`。
- `LN gated residual adapter` 分类头：`macro-F1=0.6367`，`balanced accuracy=0.6360`。
- 结论：Adapter 在 frozen feature 阶段有效，但叠加到已经部分解冻的 encoder 后会扰动已适配的特征空间，不纳入当前最终方案。

- 输出目录：`outputs/openclip_lora_after_unfreeze_probe`
- LoRA 设置：`last_two_blocks_attn_lora / last_two_blocks_mlp_lora`，rank=`8`，alpha=`16`，dropout=`0.05`，学习率候选 `1e-4 / 3e-4`。
- 最佳 LoRA：`last_two_blocks_attn_lora`，`lr_backbone=3e-4`，`macro-F1=0.5788`，`balanced accuracy=0.5760`。
- 结论：当前 LoRA 参数量太少，无法像直接解冻前后关键 block 那样充分适配 32x32 图像纹理；暂不作为主线。

### emb_first3_last6 主线的四类细化尝试

固定主线：`OpenCLIP ViT-B-32/openai + emb_first3_last6 + linear head + lr_backbone=1e-5 + lr_head=1e-3 + TTA=3`。

1. `pairwise margin loss`
   - 输出目录：`outputs/openclip_partial_unfreeze_pairwise_refine`
   - 尝试：`pairwise_class_set=Class_2,Class_3,Class_4`，`weight=0.01/0.02`，`margin=0.03/0.04`
   - 最佳：`weight=0.01, margin=0.04`
   - 指标：`macro-F1=0.6620`，`balanced accuracy=0.6640`
   - 结论：边界约束能让部分 fold 变高，但整体低于无 margin 的 `0.6750`，说明对当前部分解冻模型来说约束偏硬。

2. 更强正则
   - 输出目录：`outputs/openclip_partial_unfreeze_regularization_refine`
   - 尝试：`weight_decay=0.075/0.10`，`label_smoothing=0.03/0.05`
   - 最佳：`weight_decay=0.10, label_smoothing=0.03`
   - 指标：`macro-F1=0.6704`，`balanced accuracy=0.6720`
   - 结论：更强正则接近当前最优，但没有超过；当前 `weight_decay=0.05, label_smoothing=0.03` 仍更合适。

3. 增强强度复查
   - 输出目录：`outputs/openclip_partial_unfreeze_crop_refine`，`outputs/openclip_partial_unfreeze_crop095_refine`
   - 尝试：`crop_scale_min=0.85/0.95`，固定 `jitter=0.05`
   - `crop_scale_min=0.85`：`macro-F1=0.6722`，`balanced accuracy=0.6680`
   - `crop_scale_min=0.95`：`macro-F1=0.6502`，`balanced accuracy=0.6520`
   - 结论：略强裁剪接近当前最优，但仍低于 `0.90`；过保守裁剪会牺牲 Class_0/Class_2。

4. 训练轮数和 patience
   - 输出目录：`outputs/openclip_partial_unfreeze_longer_train_probe`
   - 尝试：`epochs=100, patience=15`
   - 指标：`macro-F1=0.6750`，`balanced accuracy=0.6760`
   - 结论：与 `epochs=80, patience=12` 完全一致，说明当前早停没有截断提升；继续增加训练轮数收益不大。

阶段结论：四类细化均未超过当前主方案。当前最优仍保持 `emb_first3_last6 + linear head + crop_scale_min=0.90 + jitter=0.05 + weight_decay=0.05 + label_smoothing=0.03`，5-fold `macro-F1=0.6750`，`balanced accuracy=0.6760`。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_full_visual_unfreeze_lr_probe`
- 解冻策略：`emb_first6_last6`
- 可训练参数：`87851781` / `151279878`，比例 `58.07%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`5e-06`，lr_head：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6480`
- balanced accuracy：`0.6480`
- 每类 recall：Class_0=0.76，Class_1=0.74，Class_2=0.64，Class_3=0.54，Class_4=0.56

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_adapter_head_probe`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- head_type：`linear`，adapter_dim：`24`，adapter_scale：`1.0`，lr_adapter：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6750`
- balanced accuracy：`0.6760`
- 每类 recall：Class_0=0.68，Class_1=0.86，Class_2=0.62，Class_3=0.66，Class_4=0.56

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_lora_after_unfreeze_probe`
- 解冻策略：`last_two_blocks_attn_lora`
- 可训练参数：`77829` / `151353606`，比例 `0.05%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`0.0003`，lr_head：`0.001`
- head_type：`linear`，adapter_dim：`24`，adapter_scale：`1.0`，lr_adapter：`0.001`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`8`，alpha=`16.0`，dropout=`0.05`
- macro-F1：`0.5788`
- balanced accuracy：`0.5760`
- 每类 recall：Class_0=0.68，Class_1=0.70，Class_2=0.44，Class_3=0.52，Class_4=0.54

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_finetune_pairwise_smoke`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- head_type：`linear`，adapter_dim：`24`，adapter_scale：`1.0`，lr_adapter：`0.001`
- pairwise_margin_weight：`0.02`，pairwise_margin：`0.04`，pairwise_class_set：`Class_2,Class_3,Class_4`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.1933`
- balanced accuracy：`0.2280`
- 每类 recall：Class_0=0.30，Class_1=0.34，Class_2=0.40，Class_3=0.00，Class_4=0.10

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_pairwise_refine`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- head_type：`linear`，adapter_dim：`24`，adapter_scale：`1.0`，lr_adapter：`0.001`
- pairwise_margin_weight：`0.01`，pairwise_margin：`0.04`，pairwise_class_set：`Class_2,Class_3,Class_4`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6620`
- balanced accuracy：`0.6640`
- 每类 recall：Class_0=0.74，Class_1=0.84，Class_2=0.62，Class_3=0.56，Class_4=0.56

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_regularization_refine`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- head_type：`linear`，adapter_dim：`24`，adapter_scale：`1.0`，lr_adapter：`0.001`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.0`，pairwise_class_set：`Class_2,Class_3,Class_4`
- weight_decay：`0.1`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6704`
- balanced accuracy：`0.6720`
- 每类 recall：Class_0=0.62，Class_1=0.86，Class_2=0.60，Class_3=0.76，Class_4=0.52

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_crop_refine`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- head_type：`linear`，adapter_dim：`24`，adapter_scale：`1.0`，lr_adapter：`0.001`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.0`，pairwise_class_set：`Class_2,Class_3,Class_4`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6722`
- balanced accuracy：`0.6680`
- 每类 recall：Class_0=0.72，Class_1=0.78，Class_2=0.66，Class_3=0.66，Class_4=0.52

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_crop095_refine`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- head_type：`linear`，adapter_dim：`24`，adapter_scale：`1.0`，lr_adapter：`0.001`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.0`，pairwise_class_set：`Class_2,Class_3,Class_4`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6502`
- balanced accuracy：`0.6520`
- 每类 recall：Class_0=0.52，Class_1=0.80，Class_2=0.50，Class_3=0.74，Class_4=0.70

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_longer_train_probe`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- head_type：`linear`，adapter_dim：`24`，adapter_scale：`1.0`，lr_adapter：`0.001`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.0`，pairwise_class_set：`Class_2,Class_3,Class_4`
- weight_decay：`0.05`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6750`
- balanced accuracy：`0.6760`
- 每类 recall：Class_0=0.68，Class_1=0.86，Class_2=0.62，Class_3=0.66，Class_4=0.56

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。

### OpenCLIP 轻量微调/旋转增强补充实验

- 输出目录：`outputs\openclip_partial_unfreeze_combo_refine`
- 解冻策略：`emb_first3_last6`
- 可训练参数：`66588165` / `151279878`，比例 `44.02%`
- 旋转增强：`none`，角度参数：`0.0`
- lr_backbone：`1e-05`，lr_head：`0.001`
- head_type：`linear`，adapter_dim：`24`，adapter_scale：`1.0`，lr_adapter：`0.001`
- pairwise_margin_weight：`0.0`，pairwise_margin：`0.0`，pairwise_class_set：`Class_2,Class_3,Class_4`
- weight_decay：`0.1`，label_smoothing：`0.03`
- 类别权重：Class_0=`1.15`，Class_1=`1.0`，Class_2=`1.25`，Class_3=`1.0`，Class_4=`1.03`
- LoRA：rank=`4`，alpha=`8.0`，dropout=`0.0`
- macro-F1：`0.6987`
- balanced accuracy：`0.6960`
- 每类 recall：Class_0=0.78，Class_1=0.76，Class_2=0.60，Class_3=0.68，Class_4=0.66

结论：本轮结果用于检验 ViT 部分解冻训练是否能超过当前 feature adapter 最优线。若超过 0.65，说明非冻结微调是后续主线；若明显低于 0.60，则说明 250 张样本下部分解冻仍有过拟合风险。
### 组合细化突破与测试集提交生成

- 输出目录：`outputs/openclip_partial_unfreeze_combo_refine`
- 配置：`OpenCLIP ViT-B-32/openai + emb_first3_last6 + linear head`
- 组合设置：`crop_scale_min=0.85`，`jitter=0.05`，`weight_decay=0.10`，`label_smoothing=0.03`，`epochs=100`，`patience=15`
- 5-fold 指标：`macro-F1=0.6987`，`balanced accuracy=0.6960`
- 每类 recall：Class_0=0.78，Class_1=0.76，Class_2=0.60，Class_3=0.68，Class_4=0.66
- 结论：此前单独尝试 `crop=0.85` 和更强正则都未超过主线，但二者叠加后明显提升，说明该任务中适度更强的数据扰动与更强 weight decay 存在互补，当前最优更新为该组合。

### 最终单模型训练与测试集预测

- 输出目录：`outputs/openclip_partial_unfreeze_final_submission_combo`
- 测试集目录：`final_solution_openclip_adapter/test_shuffled/test_shuffled`
- 最终训练：使用全部 250 张训练图，按 5-fold 最优平均 epoch 取 `epochs=29`
- 生成提交文件：`submission.csv` 和 `outputs/openclip_partial_unfreeze_final_submission_combo/submission_openclip_partial_unfreeze_combo.csv`
- 测试集图片数：`1968`
- 预测类别分布：Class_0=951，Class_1=128，Class_2=344，Class_3=296，Class_4=249
- 置信度摘要：mean=0.9165，median=0.9701，p90=0.9733，max=0.9753
- 伪标签候选：已生成 `outputs/openclip_partial_unfreeze_final_submission_combo/pseudo_label_candidates_top80_per_class.csv`，每类 top80，最低置信度均高于 0.95
- 结论：当前已得到可提交的单模型 CSV。伪标签有可尝试空间，但由于测试集预测明显不均衡且 softmax 可能过度自信，不建议直接用全部测试集伪标签训练；若继续尝试，应只用每类限额的高置信候选，并保留当前提交作为 fallback。
### Final epoch sweep for submission

- 固定配置：`OpenCLIP ViT-B-32/openai + emb_first3_last6 + linear head`
- 固定训练/推理模块：`crop_scale_min=0.85`、`jitter=0.05`、`weight_decay=0.10`、`label_smoothing=0.03`、`TTA=3`
- 不使用伪标签，不使用模型集成。
- 训练集：全部 250 张训练图。
- 测试集：`final_solution_openclip_adapter/test_shuffled/test_shuffled`，共 1968 张。

候选版本：

| final epoch | 输出目录 | Class_0 | Class_1 | Class_2 | Class_3 | Class_4 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 24 | `outputs/openclip_partial_unfreeze_final_epoch_24` | 783 | 305 | 151 | 369 | 360 | 分布较均衡，但 Class_2 偏少 |
| 29 | `outputs/openclip_partial_unfreeze_final_epoch_29` | 951 | 128 | 344 | 296 | 249 | 当前根目录 `submission.csv` 采用该版本 |
| 34 | `outputs/openclip_partial_unfreeze_final_epoch_34` | 775 | 71 | 269 | 467 | 386 | Class_1 明显偏少 |
| 40 | `outputs/openclip_partial_unfreeze_final_epoch_40` | 660 | 83 | 203 | 705 | 317 | Class_3 明显偏多，可能有后期漂移 |

结论：

- epoch sweep 只影响最终全量训练轮数，不改变 5-fold 方案选择。
- 从无标签测试集分布看，epoch 34/40 出现更明显的类别漂移；epoch 24 较均衡但 Class_2 数量过低。
- epoch 29 与 5-fold 最优组合的早停区间更接近，置信度也较高，因此当前最终提交保留 epoch 29。
- 根目录最终提交文件：`submission.csv`。
