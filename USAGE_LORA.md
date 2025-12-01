# LoRA微调使用指南

本指南详细介绍如何使用LoRA技术微调大语言模型进行股票价格预测。

## 什么是LoRA？

LoRA (Low-Rank Adaptation) 是一种高效的模型微调技术：
- **只训练少量参数**：通常只需训练<1%的模型参数
- **内存效率高**：大幅降低GPU内存需求
- **训练速度快**：相比全量微调快很多
- **易于切换**：可以为不同任务保存不同的LoRA适配器

## 完整使用流程

### 步骤1：下载预训练模型

推荐使用GPT-2 Medium（345M参数）：

```bash
python download_model.py --model gpt2-medium --save_dir ./models
```

下载完成后，模型将保存在 `./models/gpt2-medium/` 目录。

### 步骤2：下载股票数据

```bash
python download.py --ticker AAPL --period 1y --output sample_test.csv
```

### 步骤3：使用LoRA微调模型

```bash
python train_lora.py \
    --data sample_test.csv \
    --model_path ./models/gpt2-medium \
    --epochs 20 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --save_dir ./checkpoints
```

**参数说明**：
- `--lora_rank`: LoRA的秩，控制适配器大小（推荐8-16）
- `--lora_alpha`: LoRA的缩放因子（通常设为rank的2倍）
- `--batch_size`: 批次大小，根据GPU内存调整（4GB GPU建议用2-4）

### 步骤4：使用微调后的模型预测

```bash
python framework_llm.py \
    --test_csv sample_test.csv \
    --model_path ./models/gpt2-medium \
    --lora_weights ./checkpoints/best_lora_weights.pth \
    --output predictions.csv
```

## LoRA参数调优建议

### Rank（秩）
- **较小值（4-8）**：参数更少，训练更快，但可能表达能力不足
- **中等值（8-16）**：推荐，平衡性能和效率
- **较大值（32+）**：表达能力更强，但参数更多

### Alpha（缩放因子）
- 通常设为 `rank * 2`（如rank=8，alpha=16）
- 控制LoRA适配器对原始权重的影响程度
- 较大的alpha意味着LoRA的影响更大

### 学习率
- LoRA微调通常使用较小的学习率（1e-4 到 5e-4）
- 比全量微调的学习率稍大一些

## 内存优化技巧

1. **减小batch_size**：如果遇到OOM错误，减小batch_size
2. **使用梯度累积**：可以修改代码使用梯度累积来模拟更大的batch
3. **混合精度训练**：可以使用torch.cuda.amp自动混合精度
4. **选择较小的模型**：如果GPU内存不足，使用GPT-2而不是GPT-2 Medium

## 常见问题

### Q: 训练时出现 "CUDA out of memory" 错误
A: 
- 减小batch_size（如改为2或1）
- 减小max_length（如改为256）
- 使用更小的模型（GPT-2而不是GPT-2 Medium）

### Q: LoRA权重文件有多大？
A: 通常只有几MB到几十MB，远小于完整模型（几百MB到几GB）

### Q: 可以同时使用多个LoRA适配器吗？
A: 理论上可以，但需要修改代码实现

### Q: LoRA微调的效果如何？
A: 对于股票预测任务，LoRA微调通常能达到接近全量微调的效果，但训练成本低得多

## 性能对比

| 方法 | 可训练参数 | GPU内存 | 训练时间 | 效果 |
|------|-----------|---------|---------|------|
| 全量微调 | 345M (100%) | ~12GB | 慢 | 最好 |
| LoRA (rank=8) | ~2M (<1%) | ~6GB | 快 | 接近全量 |
| 简单模型 | 64K | ~1GB | 很快 | 一般 |

## 下一步

- 尝试不同的LoRA参数组合
- 实验不同的基础模型（GPT-2, GPT-2 Large等）
- 添加更多特征（技术指标等）
- 实现模型集成

