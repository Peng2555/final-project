"""
股票价格预测模型框架
支持模型推理和预测
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from torch.cuda import max_memory_allocated, reset_peak_memory_stats

class SimpleStockPredictor(nn.Module):
    """简单的股票价格预测模型"""
    def __init__(self, input_size=5, hidden_size=64, window_size=30):
        super().__init__()
        self.window_size = window_size
        self.fc1 = nn.Linear(input_size * window_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Predict next close

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten: batch_size x (window_size * input_size)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def predict(test_csv='sample_test.csv', model_path=None, window_size=30, output_path='predictions.csv'):
    """
    主函数：加载模型并进行预测
    
    Args:
        test_csv: 测试数据CSV文件路径
        model_path: 训练好的模型路径（可选）
        window_size: 滑动窗口大小
        output_path: 预测结果输出路径
    
    Returns:
        predictions: 预测结果列表
    """
    # 加载数据
    print(f"加载测试数据: {test_csv}")
    df = pd.read_csv(test_csv)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 检查数据格式
    if not all(col in df.columns for col in features):
        raise ValueError(f"CSV文件必须包含以下列: {features}")
    
    # 提取最后window_size天的数据
    if len(df) < window_size:
        raise ValueError(f"数据长度({len(df)})小于窗口大小({window_size})")
    
    last_window = df.iloc[-window_size:][features].values.astype(np.float32)
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，将使用CPU（速度较慢）")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleStockPredictor(input_size=len(features), window_size=window_size)
    
    # 加载模型权重（如果提供）
    if model_path:
        print(f"加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    # 重置GPU统计信息
    if device.type == 'cuda':
        reset_peak_memory_stats(device)
    
    # 迭代预测（滑动窗口，每次预测一天）
    predictions = []
    current_window = torch.from_numpy(last_window).unsqueeze(0).to(device)  # batch_size=1
    
    for day in range(3):
        with torch.no_grad():
            pred_close = model(current_window)
        
        predictions.append(pred_close.item())
        
        # 滑动窗口：创建新行，使用预测的收盘价，其他值使用上一行的值作为占位符
        last_row = current_window[0, -1].clone().cpu().numpy()
        new_row = last_row.copy()
        new_row[3] = pred_close.item()  # 更新Close价格（索引3）
        # 移位：移除第一行，追加新行
        new_window = np.vstack((last_window[1:], new_row))
        last_window = new_window
        current_window = torch.from_numpy(new_window).unsqueeze(0).to(device)
    
    # 计算总可训练参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 峰值GPU内存使用
    if device.type == 'cuda':
        peak_gpu_mem_mb = max_memory_allocated(device) / (1024 ** 2)  # in MB
    else:
        peak_gpu_mem_mb = 0
    
    # 输出结果
    print(f"总可训练参数: {total_params}")
    if device.type == 'cuda':
        print(f"峰值GPU内存使用 (MB): {peak_gpu_mem_mb:.2f}")
    print(f"预测结果: {predictions}")
    
    # 保存预测结果到CSV
    pred_df = pd.DataFrame({
        'Predicted_Close_Day1': [predictions[0]],
        'Predicted_Close_Day2': [predictions[1]],
        'Predicted_Close_Day3': [predictions[2]]
    })
    pred_df.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")
    
    return predictions

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='股票价格预测推理脚本')
    parser.add_argument('--test_csv', type=str, default='sample_test.csv', 
                       help='测试数据CSV文件路径')
    parser.add_argument('--model_path', type=str, default=None,
                       help='训练好的模型路径（可选，如果不提供则使用未训练的模型）')
    parser.add_argument('--window_size', type=int, default=30,
                       help='滑动窗口大小')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='预测结果输出文件路径')
    
    args = parser.parse_args()
    
    predict(
        test_csv=args.test_csv, 
        model_path=args.model_path, 
        window_size=args.window_size,
        output_path=args.output
    )

if __name__ == "__main__":
    main()

