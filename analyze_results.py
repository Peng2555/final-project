"""
模型预测结果分析和可视化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
import os
from datetime import datetime, timedelta

# 设置字体（使用默认字体，避免中文乱码）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history_path='./checkpoints/training_history.csv', save_path='./training_history.png'):
    """绘制训练历史曲线"""
    df = pd.read_csv(history_path)
    
    # 过滤掉空值
    df = df.dropna()
    
    if len(df) == 0:
        print("警告: 训练历史数据为空")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2, markersize=4)
    ax.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Model Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 标记最佳模型
    best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    best_val_loss = df['val_loss'].min()
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Model (Epoch {best_epoch})')
    ax.plot(best_epoch, best_val_loss, 'ro', markersize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图已保存到: {save_path}")
    plt.close()

def plot_price_predictions(test_csv='sample_test.csv', predictions_csv='predictions.csv', 
                          window_size=30, save_path='./price_predictions.png'):
    """绘制历史价格和预测价格"""
    # 加载历史数据
    df = pd.read_csv(test_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 加载预测结果
    pred_df = pd.read_csv(predictions_csv)
    
    # 获取最后window_size天的数据用于显示
    recent_data = df.tail(window_size + 10).copy()
    
    # 创建预测日期（从最后一天开始）
    last_date = recent_data['Date'].iloc[-1]
    pred_dates = [last_date + timedelta(days=i+1) for i in range(3)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上图：完整视图
    ax1.plot(recent_data['Date'], recent_data['Close'], label='Historical Close Price', 
             marker='o', linewidth=2, markersize=4, color='#2E86AB')
    ax1.plot(pred_dates, [pred_df['Predicted_Close_Day1'].iloc[0], 
                          pred_df['Predicted_Close_Day2'].iloc[0],
                          pred_df['Predicted_Close_Day3'].iloc[0]], 
             label='Predicted Close Price', marker='s', linewidth=2, markersize=6, 
             color='#A23B72', linestyle='--')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Close Price ($)', fontsize=12)
    ax1.set_title('Stock Price Prediction (History + Next 3 Days)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 下图：最近30天 + 预测
    recent_30 = recent_data.tail(30)
    ax2.plot(recent_30['Date'], recent_30['Close'], label='Last 30 Days Close Price', 
             marker='o', linewidth=2, markersize=5, color='#2E86AB')
    ax2.plot(pred_dates, [pred_df['Predicted_Close_Day1'].iloc[0], 
                          pred_df['Predicted_Close_Day2'].iloc[0],
                          pred_df['Predicted_Close_Day3'].iloc[0]], 
             label='Predicted Close Price', marker='s', linewidth=2.5, markersize=8, 
             color='#A23B72', linestyle='--', markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Close Price ($)', fontsize=12)
    ax2.set_title('Last 30 Days Trend + Next 3 Days Prediction', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"价格预测图已保存到: {save_path}")
    plt.close()

def calculate_statistics(test_csv='sample_test.csv', predictions_csv='predictions.csv'):
    """计算统计指标"""
    df = pd.read_csv(test_csv)
    pred_df = pd.read_csv(predictions_csv)
    
    # 历史统计
    recent_close = df['Close'].tail(30).values
    last_close = df['Close'].iloc[-1]
    
    predictions = [
        pred_df['Predicted_Close_Day1'].iloc[0],
        pred_df['Predicted_Close_Day2'].iloc[0],
        pred_df['Predicted_Close_Day3'].iloc[0]
    ]
    
    stats = {
        '历史数据统计（最近30天）': {
            '平均收盘价': np.mean(recent_close),
            '最高收盘价': np.max(recent_close),
            '最低收盘价': np.min(recent_close),
            '标准差': np.std(recent_close),
            '最后收盘价': last_close
        },
        '预测统计': {
            '第1天预测': predictions[0],
            '第2天预测': predictions[1],
            '第3天预测': predictions[2],
            '平均预测': np.mean(predictions),
            '预测变化率（相对最后收盘价）': ((predictions[-1] - last_close) / last_close) * 100
        },
        '趋势分析': {
            '预测趋势': '上涨' if predictions[-1] > predictions[0] else '下跌',
            '预测波动': np.std(predictions),
            '预测范围': f"{min(predictions):.2f} - {max(predictions):.2f}"
        }
    }
    
    return stats

def generate_report(stats, save_path='./analysis_report.txt'):
    """生成分析报告"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("股票价格预测分析报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for section, data in stats.items():
            f.write(f"\n【{section}】\n")
            f.write("-" * 60 + "\n")
            for key, value in data.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.2f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("报告结束\n")
    
    print(f"分析报告已保存到: {save_path}")

def main():
    """主函数"""
    print("开始生成分析和可视化...")
    
    # 检查文件是否存在
    if not os.path.exists('./checkpoints/training_history.csv'):
        print("警告: 训练历史文件不存在")
    else:
        plot_training_history()
    
    if not os.path.exists('predictions.csv'):
        print("警告: 预测结果文件不存在")
    else:
        plot_price_predictions()
        stats = calculate_statistics()
        generate_report(stats)
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("统计摘要")
        print("=" * 60)
        for section, data in stats.items():
            print(f"\n【{section}】")
            for key, value in data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    
    print("\n分析和可视化完成！")

if __name__ == "__main__":
    main()

