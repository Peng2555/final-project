# 快速开始指南

本指南将帮助您快速上手股票价格预测项目。

## 前置要求

- Python 3.7 或更高版本
- CUDA支持的GPU（推荐，用于加速训练和推理）
- 至少 2GB 可用磁盘空间

## 安装步骤

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd final-work
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：如果您使用GPU，请确保安装支持CUDA的PyTorch版本。访问 [PyTorch官网](https://pytorch.org/) 获取适合您系统的安装命令。

## 使用流程

### 步骤1：下载股票数据

```bash
# 下载AAPL（苹果公司）过去1年的数据
python download.py --ticker AAPL --period 1y

# 或者下载其他股票
python download.py --ticker TSLA --period 6mo --output tsla_data.csv
```

### 步骤2：训练模型

```bash
# 使用默认参数训练
python train.py --data sample_test.csv --epochs 100

# 自定义参数训练
python train.py --data sample_test.csv --epochs 200 --batch_size 64 --learning_rate 0.0001
```

训练完成后，您会得到：
- `model.pth` - 最终模型
- `best_model.pth` - 验证集上表现最好的模型
- `training_history.csv` - 训练历史记录

### 步骤3：进行预测

```bash
# 使用训练好的模型进行预测
python framework.py --model_path best_model.pth --test_csv sample_test.csv

# 或者使用未训练的模型（仅用于测试）
python framework.py --test_csv sample_test.csv
```

预测结果将保存到 `predictions.csv` 文件中。

## 完整示例

```bash
# 1. 下载数据
python download.py --ticker AAPL --period 1y

# 2. 训练模型（可能需要几分钟到几十分钟，取决于您的硬件）
python train.py --data sample_test.csv --epochs 100 --batch_size 32

# 3. 进行预测
python framework.py --model_path best_model.pth --test_csv sample_test.csv

# 4. 查看预测结果
cat predictions.csv
```

## 常见问题

### Q: 训练时出现 "CUDA out of memory" 错误
A: 减小批次大小：`python train.py --batch_size 16` 或 `--batch_size 8`

### Q: 没有GPU怎么办？
A: 代码会自动使用CPU，但训练速度会慢很多。建议使用较小的数据集和较少的训练轮数。

### Q: 如何提高预测准确性？
A: 
- 增加训练数据量
- 增加训练轮数
- 尝试不同的模型架构（修改 `framework.py` 中的 `SimpleStockPredictor` 类）
- 添加更多特征（技术指标等）

### Q: 预测结果不准确怎么办？
A: 股票价格预测本身具有很高的不确定性。建议：
- 使用更多历史数据
- 尝试不同的模型架构（LSTM、Transformer等）
- 添加更多特征和技术指标
- 进行模型集成

## 下一步

- 查看 `README.md` 了解项目详细信息
- 修改 `framework.py` 中的模型架构
- 添加更多数据预处理步骤
- 实现模型评估指标（MAE、RMSE等）

## 获取帮助

如有问题，请通过GitHub Issues联系。

