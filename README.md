# 股票价格预测项目 - 基于LoRA大模型微调

这是一个使用**LoRA（Low-Rank Adaptation）技术微调大语言模型**进行股票价格预测的项目。项目使用PyTorch和Transformers框架，默认采用 **Qwen3-0.6B** 作为基础模型，通过LoRA高效微调，能够预测未来3天的股票收盘价。

## 项目简介

本项目通过**LoRA微调大语言模型（Qwen3-0.6B）**来预测股票价格。主要功能包括：
- 从Yahoo Finance下载股票历史数据
- 自动下载或本地加载预训练大语言模型（默认：Qwen3-0.6B）
- 使用LoRA技术高效微调大模型（只需训练少量参数）
- 将时间序列数据转换为文本格式，利用大语言模型的理解能力
- 支持模型训练和推理
- GPU加速支持

## 项目结构

```
.
├── README.md                  # 项目说明文档
├── QUICKSTART.md             # 快速开始指南
├── requirements.txt          # Python依赖包列表
├── .gitignore               # Git忽略文件配置
├── LICENSE                  # 许可证文件
├── setup.py                 # 项目安装配置
├── download.py              # 股票数据下载脚本
├── download_model.py        # 大模型下载脚本（新增）
├── lora.py                  # LoRA实现模块（新增）
├── llm_model.py             # 基于大模型的预测模型（新增）
├── train.py                 # 简单模型训练脚本（原始）
├── train_lora.py            # LoRA微调训练脚本（新增）
├── framework.py             # 简单模型推理脚本（原始）
├── framework_llm.py         # 大模型推理脚本（新增）
├── Final_Project_export/    # 原始项目文件（参考）
│   ├── download.py
│   ├── framework.py
│   ├── sample_test.csv
│   └── predictions.csv
├── models/                  # 下载的预训练模型（可选，已忽略）
├── Qwen3-0.6B/              # 默认Qwen模型存放目录（已忽略）
├── checkpoints/             # 训练检查点（已忽略）
├── sample_test.csv          # 示例测试数据（由download.py生成）
└── predictions.csv          # 预测结果输出（已忽略）
```

## 环境要求

- Python 3.7+
- CUDA支持的GPU（用于模型训练和推理）
- PyTorch（支持CUDA）

## 安装步骤

1. 克隆项目到本地：
```bash
git clone <your-repo-url>
cd final-work
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

### 方式一：使用LoRA微调大模型（推荐）

#### 1. 准备预训练大模型

默认使用 **Qwen/Qwen3-0.6B**（约0.6B参数，支持中文场景，易于LoRA微调）。

```bash
# 方式一：直接使用 Hugging Face 名称（联网）
python download_model.py --model Qwen/Qwen3-0.6B --save_dir ./Qwen3-0.6B

# 方式二：如果已手动下载，可指定本地目录
# 假设模型位于 ./Qwen3-0.6B
```

> 如果需要其他模型，可将 `--model` 替换为 Hugging Face 上的任何 CausalLM 模型名称，
> 例如 `Qwen/Qwen1.5-0.5B`、`meta-llama/Llama-2-7b-hf` 等（需注意授权）。

#### 2. 下载股票数据

```bash
python download.py --ticker AAPL --period 1y
```

#### 3. 使用LoRA微调模型

```bash
python train_lora.py \
    --data sample_test.csv \
    --model_path ./Qwen3-0.6B \
    --epochs 20 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16.0
```

训练完成后，LoRA权重将保存到 `checkpoints/best_lora_weights.pth`。

#### 4. 使用微调后的模型进行预测

```bash
python framework_llm.py \
    --test_csv sample_test.csv \
    --model_path ./Qwen3-0.6B \
    --lora_weights ./checkpoints/best_lora_weights.pth \
    --output predictions.csv
```

### 方式二：使用简单模型（原始方法）

如果您想使用简单的全连接网络（不需要大模型）：

```bash
# 1. 下载数据
python download.py

# 2. 训练模型
python train.py --data sample_test.csv --epochs 100

# 3. 进行预测
python framework.py --model_path model.pth --test_csv sample_test.csv
```

## 模型架构

### LoRA微调大模型架构（推荐）

项目默认使用**Qwen3-0.6B**作为基础模型，通过**LoRA（Low-Rank Adaptation）**技术进行高效微调：

1. **基础模型**: Qwen/Qwen3-0.6B（约0.6B参数）
   - 支持中英文语料，语义理解能力强
   - 模型体积适中，便于单卡训练

2. **LoRA适配器**:
   - 只训练少量参数（通常<1%的模型参数）
   - 将原始权重矩阵 W 分解为 W + BA（低秩分解）
   - 大幅降低训练成本和内存需求

3. **数据格式转换**:
   - 将时间序列数据转换为文本格式
   - 例如: "Day 1: Open=100.5, High=102.3, Low=99.8, Close=101.2, Volume=1000000"
   - 利用大语言模型对文本的理解能力

4. **预测头**:
   - 将语言模型的隐藏状态映射到价格预测
   - 使用全连接层输出收盘价

### 简单模型架构（备选）

项目也保留了原始的简单全连接神经网络（`SimpleStockPredictor`）：
- 输入层：接收30天的历史数据（Open, High, Low, Close, Volume）
- 隐藏层：64个神经元，使用ReLU激活函数
- 输出层：预测下一个交易日的收盘价

## 参数说明

### download_model.py 参数
- `--model`: 要下载的模型（默认：`Qwen/Qwen3-0.6B`）
- `--save_dir`: 模型保存目录（默认：`./Qwen3-0.6B`）

### train_lora.py 参数（LoRA微调）
- `--data`: 训练数据CSV文件路径
- `--model_path`: 预训练模型路径（默认：`./Qwen3-0.6B`）
- `--epochs`: 训练轮数（默认：20）
- `--batch_size`: 批次大小（默认：4，大模型建议使用小批次）
- `--learning_rate`: 学习率（默认：1e-4）
- `--window_size`: 滑动窗口大小（默认：30）
- `--lora_rank`: LoRA秩（默认：8，控制适配器大小）
- `--lora_alpha`: LoRA alpha参数（默认：16.0，控制适配器缩放）
- `--save_dir`: 模型保存目录（默认：`./checkpoints`）

### framework_llm.py 参数（大模型推理）
- `--test_csv`: 测试数据CSV文件路径
- `--model_path`: 预训练模型路径
- `--lora_weights`: LoRA权重路径（如果使用LoRA微调）
- `--window_size`: 滑动窗口大小（默认：30）
- `--output`: 预测结果输出路径

### train.py 参数（简单模型）
- `--data`: 训练数据CSV文件路径
- `--epochs`: 训练轮数（默认：100）
- `--batch_size`: 批次大小（默认：32）
- `--learning_rate`: 学习率（默认：0.001）
- `--window_size`: 滑动窗口大小（默认：30）

### framework.py 参数（简单模型推理）
- `--test_csv`: 测试数据CSV文件路径
- `--model_path`: 训练好的模型路径（可选）
- `--window_size`: 滑动窗口大小（默认：30）

## 输出说明

运行 `framework.py` 后，会输出：
- 总可训练参数数量
- 峰值GPU内存使用量（MB）
- 未来3天的收盘价预测值
- 预测结果保存到 `predictions.csv`

## 推荐模型

**推荐使用 Qwen/Qwen3-0.6B（约0.6B参数）**，原因：
- ✅ 中英文双向支持，适合中文金融描述
- ✅ 参数量适中（单卡12GB即可运行）
- ✅ 与 LoRA 结合后显存占用低，可快速实验
- ✅ 兼容 `trust_remote_code=True` 的最新特性

其他可选模型：
- **Qwen/Qwen1.5-0.5B**：更轻量，适合显存较小的环境
- **Qwen/Qwen1.5-1.8B**：更高精度，但显存需求更高
- 也可以尝试 Llama / GPT-NeoX / InternLM 等，只需调整 `--model_path`

## 注意事项

1. **GPU要求**：
   - LoRA微调大模型：强烈建议使用CUDA GPU（至少6GB显存）
   - GPT-2 Medium + LoRA：推荐8GB+ GPU内存
   - 如果没有GPU，训练会非常慢（不推荐）

2. **内存优化**：
   - 使用较小的batch_size（如4或8）
   - 可以使用梯度累积来模拟更大的batch
   - LoRA已经大幅降低了内存需求

3. **数据格式**：CSV文件必须包含以下列：`Date`, `Open`, `High`, `Low`, `Close`, `Volume`

4. **预测准确性**：股票价格预测受多种因素影响，本项目的预测结果仅供参考，不构成投资建议。

5. **LoRA优势**：
   - 只需训练<1%的模型参数
   - 训练速度快，内存占用低
   - 可以轻松切换不同的适配器

## 扩展建议

1. **数据增强**：添加技术指标（如移动平均线、RSI等）
2. **模型改进**：使用LSTM、Transformer或微调LLM模型
3. **多股票支持**：扩展为支持多只股票同时预测
4. **实时预测**：集成实时数据源进行在线预测
5. **评估指标**：添加MAE、RMSE等评估指标

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过GitHub Issues联系。

