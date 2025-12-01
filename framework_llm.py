"""
基于大语言模型的股票价格预测推理脚本
支持LoRA微调后的模型进行预测
"""
import torch
import pandas as pd
import numpy as np
import argparse
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from llm_model import StockLLMPredictor
import os

def predict_with_llm(
    test_csv='sample_test.csv',
    model_path='Qwen/Qwen3-0.6B',
    lora_weights_path=None,
    full_model_path=None,
    window_size=30,
    output_path='predictions.csv',
    use_lora=True
):
    """
    使用大语言模型进行股票价格预测
    
    Args:
        test_csv: 测试数据CSV文件路径
        model_path: 预训练模型路径
        lora_weights_path: LoRA权重路径（如果使用LoRA微调）
        window_size: 滑动窗口大小
        output_path: 预测结果输出路径
        use_lora: 是否使用LoRA
    
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
    dates = pd.to_datetime(df['Date']).values[-window_size:] if 'Date' in df.columns else None
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，将使用CPU（速度较慢）")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    print(f"使用设备: {device}")
    
    # 创建模型
    print(f"\n加载模型: {model_path}")
    model = StockLLMPredictor(
        model_name_or_path=model_path,
        window_size=window_size,
        use_lora=use_lora,
        use_fp16=False  # 推理时使用float32更稳定
    )
    model.to(device)
    
    # 优先加载完整模型（包含预测头）
    if full_model_path and os.path.exists(full_model_path):
        print(f"加载完整模型状态: {full_model_path}")
        checkpoint = torch.load(full_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  已加载 epoch {checkpoint.get('epoch', '?')} 的模型")
        print(f"  验证损失: {checkpoint.get('val_loss', '?'):.6f}")
    # 否则加载LoRA权重（如果提供）
    elif lora_weights_path and os.path.exists(lora_weights_path):
        print(f"加载LoRA权重: {lora_weights_path}")
        model.load_lora_weights(lora_weights_path)
    elif use_lora:
        print("警告: 未提供模型权重路径，将使用未微调的模型")
    
    model.eval()
    
    # 重置GPU统计信息
    if device.type == 'cuda':
        reset_peak_memory_stats(device)
    
    # 迭代预测（滑动窗口，每次预测一天）
    predictions = []
    current_window = last_window.copy()
    
    print(f"\n开始预测...")
    for day in range(3):
        # 使用模型预测
        pred_close = model.predict_from_data(current_window, dates)
        predictions.append(pred_close)
        
        print(f"第 {day+1} 天预测收盘价: {pred_close:.2f}")
        
        # 滑动窗口：创建新行，使用预测的收盘价
        # 对于其他值，我们使用简单的估计（例如使用前一天的Open作为新一天的Open）
        last_row = current_window[-1].copy()
        new_row = last_row.copy()
        new_row[3] = pred_close  # 更新Close价格
        
        # 简单估计：假设Open价格接近前一天的Close
        new_row[0] = pred_close * 0.99  # Open略低于Close
        new_row[1] = pred_close * 1.02  # High略高于Close
        new_row[2] = pred_close * 0.98  # Low略低于Close
        # Volume保持与前一天相同（或可以添加更复杂的估计）
        
        # 移位：移除第一行，追加新行
        current_window = np.vstack((current_window[1:], new_row))
        
        # 更新日期（如果可用）
        if dates is not None:
            # 简单增加一天（实际应该考虑交易日）
            from datetime import timedelta
            # 确保日期是 datetime 类型
            if isinstance(dates[-1], np.datetime64):
                next_date = pd.Timestamp(dates[-1]) + timedelta(days=1)
            else:
                next_date = dates[-1] + timedelta(days=1)
            dates = np.append(dates[1:], next_date)
    
    # 计算总可训练参数
    if use_lora:
        trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # 峰值GPU内存使用
    if device.type == 'cuda':
        peak_gpu_mem_mb = max_memory_allocated(device) / (1024 ** 2)  # in MB
    else:
        peak_gpu_mem_mb = 0
    
    # 输出结果
    print(f"\n模型统计:")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    if use_lora:
        print(f"  可训练参数（LoRA）: {trainable_params / 1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
    if device.type == 'cuda':
        print(f"  峰值GPU内存使用 (MB): {peak_gpu_mem_mb:.2f}")
    print(f"\n预测结果:")
    for i, pred in enumerate(predictions, 1):
        print(f"  第 {i} 天收盘价: {pred:.2f}")
    
    # 保存预测结果到CSV
    pred_df = pd.DataFrame({
        'Predicted_Close_Day1': [predictions[0]],
        'Predicted_Close_Day2': [predictions[1]],
        'Predicted_Close_Day3': [predictions[2]]
    })
    pred_df.to_csv(output_path, index=False)
    print(f"\n预测结果已保存到: {output_path}")
    
    return predictions

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='使用大语言模型进行股票价格预测')
    parser.add_argument('--test_csv', type=str, default='sample_test.csv', 
                       help='测试数据CSV文件路径')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-0.6B',
                       help='预训练模型路径（可以是本地路径或Hugging Face模型名称）')
    parser.add_argument('--lora_weights', type=str, default=None,
                       help='LoRA权重路径（如果使用LoRA微调）')
    parser.add_argument('--full_model', type=str, default=None,
                       help='完整模型路径（包含预测头，优先使用，如 ./checkpoints/best_model_full.pth）')
    parser.add_argument('--window_size', type=int, default=30,
                       help='滑动窗口大小')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='预测结果输出文件路径')
    parser.add_argument('--no_lora', action='store_true',
                       help='不使用LoRA（使用原始模型）')
    
    args = parser.parse_args()
    
    predict_with_llm(
        test_csv=args.test_csv,
        model_path=args.model_path,
        lora_weights_path=args.lora_weights,
        full_model_path=args.full_model,
        window_size=args.window_size,
        output_path=args.output,
        use_lora=not args.no_lora
    )

if __name__ == "__main__":
    main()

