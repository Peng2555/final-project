"""
LoRA (Low-Rank Adaptation) 实现
用于高效微调大语言模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LoRALayer(nn.Module):
    """
    LoRA层实现
    将原始权重矩阵 W 分解为 W + BA，其中 B 和 A 是低秩矩阵
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        scale: float = 1.0,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = scale / rank
        
        # LoRA的A和B矩阵，使用指定的dtype或默认float32
        dtype = dtype or torch.float32
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, dtype=dtype) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        
        # Dropout层
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x @ W^T + x @ A^T @ B^T * scale
        """
        # 计算LoRA适配: x @ A^T @ B^T
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_output * self.scale * self.alpha

class LoRALinear(nn.Module):
    """
    带LoRA适配器的线性层
    包装原始线性层并添加LoRA适配
    """
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear_layer
        
        # 获取原始层的dtype（从权重参数中获取）
        original_dtype = next(linear_layer.parameters()).dtype
        
        self.lora = LoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=original_dtype
        )
        
        # 冻结原始权重
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：原始输出 + LoRA适配
        """
        # 确保LoRA输出与原始输出类型匹配
        linear_output = self.linear(x)
        lora_output = self.lora(x)
        
        # 如果类型不匹配，转换LoRA输出类型
        if linear_output.dtype != lora_output.dtype:
            lora_output = lora_output.to(linear_output.dtype)
        
        return linear_output + lora_output

def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[list] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> nn.Module:
    """
    将LoRA适配器应用到模型的指定模块
    
    Args:
        model: 要应用LoRA的模型
        target_modules: 目标模块名称列表（如 ['c_attn', 'c_proj']），如果为None则应用到所有Linear层
        rank: LoRA的秩（低秩矩阵的维度）
        alpha: LoRA的缩放因子
        dropout: Dropout率
    
    Returns:
        应用了LoRA的模型
    """
    if target_modules is None:
        # 默认应用到所有Linear层
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
                target_modules.append(name)
    
    # 应用LoRA到目标模块
    for name, module in model.named_modules():
        # 检查是否是目标模块且是Linear层
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            # 获取父模块和子模块名
            parts = name.split('.')
            parent_name = '.'.join(parts[:-1])
            child_name = parts[-1]
            
            # 获取父模块
            if parent_name:
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
            else:
                parent_module = model
            
            # 替换为LoRALinear
            lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent_module, child_name, lora_linear)
            print(f"已为 {name} 应用LoRA适配器 (rank={rank}, alpha={alpha})")
    
    return model

def get_lora_parameters(model: nn.Module):
    """
    获取所有LoRA参数（用于优化器）
    
    Returns:
        需要训练的参数列表
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params.append(param)
    return lora_params

def save_lora_weights(model: nn.Module, path: str):
    """
    保存LoRA权重（只保存LoRA适配器，不保存基础模型）
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_state_dict[name] = param.cpu()
    
    torch.save(lora_state_dict, path)
    print(f"LoRA权重已保存到: {path}")

def load_lora_weights(model: nn.Module, path: str, strict: bool = True):
    """
    加载LoRA权重到模型
    """
    lora_state_dict = torch.load(path, map_location='cpu')
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(lora_state_dict, strict=False)
    
    if strict:
        if missing_keys:
            print(f"警告: 缺少以下键: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 意外的键: {unexpected_keys}")
    else:
        if missing_keys:
            print(f"已忽略缺少的键: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"已忽略意外的键: {len(unexpected_keys)} 个")
    
    print(f"LoRA权重已从 {path} 加载")

