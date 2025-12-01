"""
基于大语言模型的股票价格预测模型
使用Qwen3-0.6B作为基础模型，通过LoRA进行微调
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from lora import apply_lora_to_model, get_lora_parameters, save_lora_weights, load_lora_weights
import pandas as pd
import numpy as np
from typing import Optional, Tuple

class StockLLMPredictor(nn.Module):
    """
    基于大语言模型的股票价格预测器
    将时间序列数据转换为文本格式，使用Qwen3-0.6B进行预测
    """
    def __init__(
        self,
        model_name_or_path: str = 'Qwen/Qwen3-0.6B',
        window_size: int = 30,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        max_length: int = 512,
        use_fp16: bool = False  # 默认使用float32以提高训练稳定性
    ):
        super().__init__()
        self.window_size = window_size
        self.max_length = max_length
        self.use_lora = use_lora
        
        # 加载预训练模型和tokenizer
        print(f"加载模型: {model_name_or_path}")
        # 根据use_fp16参数决定使用float16还是float32
        model_dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32
        print(f"使用数据类型: {model_dtype}")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=model_dtype,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"无法从路径加载模型: {e}")
            print("尝试使用默认模型...")
            default_model = './Qwen3-0.6B'
            self.model = AutoModelForCausalLM.from_pretrained(
                default_model,
                torch_dtype=model_dtype,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                default_model,
                trust_remote_code=True
            )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 添加特殊token用于数值预测
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<PRICE>', '</PRICE>']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 应用LoRA（如果启用）
        if use_lora:
            print("应用LoRA适配器...")
            # Qwen模型的注意力层名称（通常使用q_proj, k_proj, v_proj, o_proj）
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            self.model = apply_lora_to_model(
                self.model,
                target_modules=target_modules,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
        
        # 预测头：将语言模型输出映射到价格预测
        # Qwen使用hidden_size而不是n_embd
        hidden_size = getattr(self.model.config, 'hidden_size', getattr(self.model.config, 'n_embd', 512))
        
        # 获取模型的dtype，确保预测头使用相同的类型
        model_dtype = next(self.model.parameters()).dtype
        
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        # 将预测头转换为与模型相同的dtype
        self.price_head = self.price_head.to(dtype=model_dtype)
    
    def format_stock_data_as_text(self, data: np.ndarray, dates: Optional[np.ndarray] = None) -> str:
        """
        将股票数据格式化为文本
        
        Args:
            data: 股票数据数组 (window_size, 5) - [Open, High, Low, Close, Volume]
            dates: 日期数组（可选）
        
        Returns:
            格式化的文本字符串
        """
        text_parts = []
        for i, row in enumerate(data):
            date_str = f"Day {i+1}" if dates is None else str(dates[i])
            text = f"{date_str}: Open={row[0]:.2f}, High={row[1]:.2f}, Low={row[2]:.2f}, Close={row[3]:.2f}, Volume={int(row[4])}"
            text_parts.append(text)
        
        # 添加预测提示
        text = "\n".join(text_parts) + "\nNext day Close price: <PRICE>"
        return text
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签（用于训练）
        
        Returns:
            (预测值, 损失)
        """
        # 获取语言模型输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最后一个隐藏状态（[CLS]位置或最后一个token）
        # 对于Qwen模型，我们使用最后一个token的隐藏状态
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        last_token_hidden = last_hidden_state[:, -1, :]  # (batch_size, hidden_size)
        
        # 确保隐藏状态类型与预测头匹配
        model_dtype = next(self.model.parameters()).dtype
        last_token_hidden = last_token_hidden.to(dtype=model_dtype)
        
        # 通过预测头得到价格预测
        price_pred = self.price_head(last_token_hidden)  # (batch_size, 1)
        
        loss = None
        if labels is not None:
            # 确保labels类型与预测值匹配
            labels = labels.to(dtype=price_pred.dtype)
            # 计算MSE损失
            loss_fn = nn.MSELoss()
            loss = loss_fn(price_pred, labels)
        
        return price_pred, loss
    
    def predict_from_data(self, data: np.ndarray, dates: Optional[np.ndarray] = None) -> float:
        """
        从原始数据预测价格
        
        Args:
            data: 股票数据数组 (window_size, 5)
            dates: 日期数组（可选）
        
        Returns:
            预测的收盘价
        """
        self.eval()
        
        # 格式化数据为文本
        text = self.format_stock_data_as_text(data, dates)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # 预测
        with torch.no_grad():
            pred, _ = self.forward(input_ids, attention_mask)
        
        return pred.item()
    
    def save_lora_weights(self, path: str):
        """保存LoRA权重"""
        if self.use_lora:
            save_lora_weights(self.model, path)
        else:
            print("警告: 模型未使用LoRA，无法保存LoRA权重")
    
    def load_lora_weights(self, path: str):
        """加载LoRA权重"""
        if self.use_lora:
            load_lora_weights(self.model, path)
        else:
            print("警告: 模型未使用LoRA，无法加载LoRA权重")
    
    def get_trainable_parameters(self):
        """获取可训练参数（LoRA参数）"""
        if self.use_lora:
            lora_params = get_lora_parameters(self.model)
            # 加上预测头的参数
            head_params = list(self.price_head.parameters())
            return lora_params + head_params
        else:
            return list(self.parameters())

