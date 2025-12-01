"""
模型训练脚本
用于训练股票价格预测模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
from framework import SimpleStockPredictor
from sklearn.preprocessing import StandardScaler
import os

class StockDataset(Dataset):
    """股票数据数据集类"""
    def __init__(self, data, window_size=30, features=None):
        self.window_size = window_size
        self.features = features or ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 提取特征
        self.data = data[self.features].values.astype(np.float32)
        
        # 数据标准化
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        
        # 创建滑动窗口
        self.X, self.y = self._create_windows()
    
    def _create_windows(self):
        """创建滑动窗口数据"""
        X, y = [], []
        for i in range(len(self.data) - self.window_size):
            X.append(self.data[i:i+self.window_size])
            y.append(self.data[i+self.window_size, 3])  # Close price (index 3)
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array([self.y[idx]]))

def train_model(model, train_loader, val_loader, epochs, device, learning_rate=0.001):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description='训练股票价格预测模型')
    parser.add_argument('--data', type=str, default='sample_test.csv', help='训练数据CSV文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--window_size', type=int, default=30, help='滑动窗口大小')
    parser.add_argument('--output', type=str, default='model.pth', help='模型保存路径')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    
    args = parser.parse_args()
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，将使用CPU训练（速度较慢）")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"使用设备: {device}")
    
    # 加载数据
    print(f"加载数据: {args.data}")
    df = pd.read_csv(args.data)
    
    # 划分训练集和验证集
    split_idx = int(len(df) * (1 - args.val_split))
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    # 创建数据集
    train_dataset = StockDataset(train_df, window_size=args.window_size)
    val_dataset = StockDataset(val_df, window_size=args.window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    model = SimpleStockPredictor(input_size=len(features), hidden_size=64, window_size=args.window_size)
    model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, args.epochs, device, args.learning_rate
    )
    
    # 保存最终模型
    torch.save(model.state_dict(), args.output)
    print(f"模型已保存到: {args.output}")
    print(f"最佳模型已保存到: best_model.pth")
    
    # 保存训练历史（可选）
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_df.to_csv('training_history.csv', index=False)
    print("训练历史已保存到: training_history.csv")

if __name__ == "__main__":
    main()

