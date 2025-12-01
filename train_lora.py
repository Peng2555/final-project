"""
使用LoRA微调大语言模型进行股票价格预测
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import os
from llm_model import StockLLMPredictor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class StockLLMDataset(Dataset):
    """用于大语言模型的股票数据集"""
    def __init__(self, data, window_size=30, features=None, tokenizer=None, max_length=512):
        self.window_size = window_size
        self.features = features or ['Open', 'High', 'Low', 'Close', 'Volume']
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 检查并验证特征列
        missing_features = [f for f in self.features if f not in data.columns]
        if missing_features:
            raise ValueError(f"CSV文件中缺少以下必需的特征列: {missing_features}")
        
        # 提取特征数据，只选择数值类型的列
        feature_data = data[self.features].copy()
        
        # 尝试将非数值列转换为数值（如果可能）
        for col in feature_data.columns:
            if feature_data[col].dtype == 'object':
                # 保存原始数据用于错误提示
                original_data = data[col].copy()
                # 尝试转换为数值
                feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
                # 检查是否还有非数值值
                if feature_data[col].isna().any():
                    invalid_indices = feature_data[col][feature_data[col].isna()].index.tolist()[:5]
                    invalid_values = [original_data.iloc[idx] for idx in invalid_indices]
                    raise ValueError(
                        f"列 '{col}' 包含无法转换为数值的数据。\n"
                        f"请检查CSV文件，确保该列只包含数值。\n"
                        f"问题数据示例（前5个）:\n"
                        f"  索引: {invalid_indices}\n"
                        f"  值: {invalid_values}\n"
                        f"提示: 如果CSV文件包含股票代码、名称等非数值列，请确保它们不在特征列表中。"
                    )
        
        # 提取特征和日期
        self.dates = pd.to_datetime(data['Date']).values if 'Date' in data.columns else None
        self.data = feature_data.values.astype(np.float32)
        
        # 数据标准化
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(self.data)
        
        # 创建滑动窗口
        self.X, self.y = self._create_windows()
    
    def _create_windows(self):
        """创建滑动窗口数据"""
        X, y = [], []
        for i in range(len(self.data_scaled) - self.window_size):
            # 使用原始数据（未标准化）用于格式化文本
            window_data = self.data[i:i+self.window_size]
            window_dates = self.dates[i:i+self.window_size] if self.dates is not None else None
            
            # 目标值（下一个交易日的收盘价，使用原始值）
            target = self.data[i+self.window_size, 3]  # Close price (index 3)
            
            X.append((window_data, window_dates))
            y.append(target)
        
        return X, np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        window_data, window_dates = self.X[idx]
        target = self.y[idx]
        
        # 格式化为文本
        text_parts = []
        for i, row in enumerate(window_data):
            date_str = f"Day {i+1}" if window_dates is None else str(window_dates[i])
            text = f"{date_str}: Open={row[0]:.2f}, High={row[1]:.2f}, Low={row[2]:.2f}, Close={row[3]:.2f}, Volume={int(row[4])}"
            text_parts.append(text)
        
        text = "\n".join(text_parts) + "\nNext day Close price: <PRICE>"
        
        # Tokenize
        if self.tokenizer is not None:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(target, dtype=torch.float32)
            }
        else:
            return {
                'text': text,
                'labels': torch.tensor(target, dtype=torch.float32)
            }

def train_lora_model(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    learning_rate=1e-4,
    save_dir='./checkpoints'
):
    """训练LoRA模型"""
    # 只优化可训练参数（LoRA参数 + 预测头）
    trainable_params = model.get_trainable_parameters()
    
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    nan_encountered = False
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)
            
            # 确保labels类型与模型匹配（模型可能使用float16）
            model_dtype = next(model.model.parameters()).dtype
            labels = labels.to(dtype=model_dtype)
            
            optimizer.zero_grad()
            pred, loss = model(input_ids, attention_mask, labels)
            
            # 检查损失是否为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n警告: 检测到无效损失值 (NaN/Inf) 在 Epoch {epoch+1}")
                nan_encountered = True
                break
            
            loss.backward()
            
            # 检查梯度是否为NaN
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\n警告: 检测到无效梯度 (NaN/Inf) 在 Epoch {epoch+1}")
                nan_encountered = True
                break
            
            optimizer.step()
            
            loss_value = loss.item()
            if not (np.isnan(loss_value) or np.isinf(loss_value)):
                train_loss += loss_value
                train_bar.set_postfix({'loss': f'{loss_value:.6f}'})
            else:
                print(f"\n警告: 损失值为无效值: {loss_value}")
                nan_encountered = True
                break
        
        if nan_encountered:
            print(f"\n训练在第 {epoch+1} 轮因 NaN/Inf 而停止")
            break
        
        train_loss /= len(train_loader)
        
        # 再次检查平均损失
        if np.isnan(train_loss) or np.isinf(train_loss):
            print(f"\n警告: 平均训练损失为无效值: {train_loss}")
            nan_encountered = True
            break
        
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_count = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].unsqueeze(1).to(device)
                
                # 确保labels类型与模型匹配（模型可能使用float16）
                model_dtype = next(model.model.parameters()).dtype
                labels = labels.to(dtype=model_dtype)
                
                pred, loss = model(input_ids, attention_mask, labels)
                
                loss_value = loss.item()
                if not (np.isnan(loss_value) or np.isinf(loss_value)):
                    val_loss += loss_value
                    val_count += 1
                    val_bar.set_postfix({'loss': f'{loss_value:.6f}'})
                else:
                    print(f"\n警告: 验证损失值为无效值: {loss_value}")
                    nan_encountered = True
                    break
        
        if nan_encountered:
            print(f"\n训练在第 {epoch+1} 轮因 NaN/Inf 而停止")
            break
        
        if val_count > 0:
            val_loss /= val_count
        else:
            val_loss = float('inf')
        
        # 检查验证损失
        if np.isnan(val_loss) or np.isinf(val_loss):
            print(f"\n警告: 平均验证损失为无效值: {val_loss}")
            nan_encountered = True
            break
        
        val_losses.append(val_loss)
        scheduler.step()
        
        # 保存最佳模型（只保存有效值）
        if not (np.isnan(val_loss) or np.isinf(val_loss)) and val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存LoRA权重
            lora_path = os.path.join(save_dir, 'best_lora_weights.pth')
            model.save_lora_weights(lora_path)
            # 保存完整模型状态（包括预测头）
            full_path = os.path.join(save_dir, 'best_model_full.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss
            }, full_path)
            print(f"\n✓ 保存最佳模型 (Val Loss: {val_loss:.6f}, Train Loss: {train_loss:.6f})")
        
        if (epoch + 1) % 5 == 0:
            print(f'\nEpoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
    
    # 如果遇到NaN，打印警告
    if nan_encountered:
        print(f"\n警告: 训练过程中检测到 NaN/Inf 值，训练已提前停止")
        print(f"已保存 {len(train_losses)} 个有效epoch的训练历史")
        if best_val_loss != float('inf'):
            print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description='使用LoRA微调大语言模型进行股票预测')
    parser.add_argument('--data', type=str, default='sample_test.csv', help='训练数据CSV文件路径')
    parser.add_argument('--model_path', type=str, default='./Qwen3-0.6B',
                       help='预训练模型路径（如果不存在会自动下载）')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小（大模型建议使用小批次）')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率（默认5e-5，更稳定）')
    parser.add_argument('--window_size', type=int, default=30, help='滑动窗口大小')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA秩')
    parser.add_argument('--lora_alpha', type=float, default=16.0, help='LoRA alpha参数')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--use_fp16', action='store_true', help='使用float16（可能不稳定，默认使用float32）')
    
    args = parser.parse_args()
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，将使用CPU训练（速度非常慢，不推荐）")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"使用设备: {device}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    print(f"\n加载数据: {args.data}")
    df = pd.read_csv(args.data)
    
    # 划分训练集和验证集
    split_idx = int(len(df) * (1 - args.val_split))
    train_df = df[:split_idx].copy()
    val_df = df[split_idx:].copy()
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    # 创建模型
    print(f"\n初始化模型...")
    model = StockLLMPredictor(
        model_name_or_path=args.model_path,
        window_size=args.window_size,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        use_fp16=args.use_fp16
    )
    model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数（LoRA）: {trainable_params / 1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
    
    # 创建数据集
    print(f"\n创建数据集...")
    train_dataset = StockLLMDataset(
        train_df,
        window_size=args.window_size,
        tokenizer=model.tokenizer,
        max_length=args.max_length
    )
    val_dataset = StockLLMDataset(
        val_df,
        window_size=args.window_size,
        tokenizer=model.tokenizer,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 训练模型
    print(f"\n开始训练...")
    train_losses, val_losses = train_lora_model(
        model,
        train_loader,
        val_loader,
        args.epochs,
        device,
        args.learning_rate,
        args.save_dir
    )
    
    # 保存训练历史（只保存有效值）
    if len(train_losses) > 0:
        history_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        history_path = os.path.join(args.save_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"\n训练历史已保存到: {history_path}")
        
        # 检查是否有保存的模型
        lora_path = os.path.join(args.save_dir, 'best_lora_weights.pth')
        if os.path.exists(lora_path):
            print(f"最佳模型已保存到: {lora_path}")
        else:
            print(f"警告: 没有保存任何模型（可能因为所有损失值都无效）")
    else:
        print(f"\n警告: 没有有效的训练历史可保存")

if __name__ == "__main__":
    main()

