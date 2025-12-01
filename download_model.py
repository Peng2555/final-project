"""
大模型下载脚本
从Hugging Face下载预训练模型（支持国内镜像）
"""
import os
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch

# 国内镜像配置
MIRROR_CONFIGS = {
    'hf-mirror': {
        'endpoint': 'https://hf-mirror.com',
        'description': 'Hugging Face 镜像站 (hf-mirror.com)'
    },
    'modelscope': {
        'endpoint': None,  # ModelScope 使用不同的 API
        'description': '魔塔社区 (ModelScope)'
    }
}

def setup_mirror(mirror_type='hf-mirror'):
    """
    设置镜像环境变量
    
    Args:
        mirror_type: 镜像类型 ('hf-mirror', 'modelscope', 'none')
    """
    if mirror_type == 'none' or mirror_type is None:
        # 清除镜像设置，使用原始 Hugging Face
        if 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        print("使用原始 Hugging Face 源")
        return
    
    if mirror_type in MIRROR_CONFIGS:
        config = MIRROR_CONFIGS[mirror_type]
        if config['endpoint']:
            os.environ['HF_ENDPOINT'] = config['endpoint']
            print(f"已设置镜像: {config['description']}")
            print(f"镜像地址: {config['endpoint']}")
        else:
            print(f"使用 {config['description']} (需要安装 modelscope 库)")
    else:
        print(f"警告: 未知的镜像类型 {mirror_type}，使用原始源")

def download_gpt2_model(model_name='gpt2-medium', save_dir='./models', mirror='hf-mirror'):
    """
    下载GPT-2模型
    
    Args:
        model_name: 模型名称 ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        save_dir: 保存目录
        mirror: 镜像类型 ('hf-mirror', 'modelscope', 'none')
    """
    # 设置镜像
    setup_mirror(mirror)
    
    print(f"正在下载 {model_name} 模型...")
    print("这可能需要几分钟时间，取决于您的网络速度...")
    
    # 创建保存目录
    model_path = os.path.join(save_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # 下载tokenizer
        print("下载tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        print(f"Tokenizer已保存到: {model_path}")
        
        # 下载模型
        print("下载模型权重...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.save_pretrained(model_path)
        print(f"模型已保存到: {model_path}")
        
        # 打印模型信息
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型信息:")
        print(f"  模型名称: {model_name}")
        print(f"  参数量: {num_params / 1e6:.2f}M")
        print(f"  保存路径: {model_path}")
        
        return model_path
    
    except Exception as e:
        print(f"下载模型时出错: {str(e)}")
        if mirror == 'hf-mirror':
            print("\n提示: 如果镜像下载失败，可以尝试:")
            print("  1. 检查网络连接")
            print("  2. 尝试使用 --mirror modelscope")
            print("  3. 或使用 --mirror none 直接连接 Hugging Face")
        raise

def download_llama_model(model_name='meta-llama/Llama-2-7b-hf', save_dir='./models', use_auth_token=None, mirror='hf-mirror'):
    """
    下载LLaMA模型（需要Hugging Face认证）
    
    Args:
        model_name: 模型名称
        save_dir: 保存目录
        use_auth_token: Hugging Face访问令牌
        mirror: 镜像类型 ('hf-mirror', 'modelscope', 'none')
    """
    # 设置镜像
    setup_mirror(mirror)
    
    print(f"正在下载 {model_name} 模型...")
    print("注意: LLaMA模型需要Hugging Face访问权限")
    
    model_path = os.path.join(save_dir, model_name.replace('/', '_'))
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # 下载tokenizer
        print("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        tokenizer.save_pretrained(model_path)
        
        # 下载模型
        print("下载模型权重（这可能需要很长时间）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_auth_token=use_auth_token
        )
        model.save_pretrained(model_path)
        
        print(f"模型已保存到: {model_path}")
        return model_path
    
    except Exception as e:
        print(f"下载模型时出错: {str(e)}")
        print("提示: 如果您要下载LLaMA模型，请确保:")
        print("  1. 已申请Hugging Face访问权限")
        print("  2. 提供了正确的访问令牌")
        if mirror == 'hf-mirror':
            print("  3. 如果镜像下载失败，可以尝试 --mirror modelscope 或 --mirror none")
        raise

def main():
    parser = argparse.ArgumentParser(description='下载预训练大语言模型（支持国内镜像）')
    parser.add_argument('--model', type=str, default='gpt2-medium',
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama2-7b'],
                       help='要下载的模型')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='模型保存目录')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='Hugging Face访问令牌（用于LLaMA模型）')
    parser.add_argument('--mirror', type=str, default='hf-mirror',
                       choices=['hf-mirror', 'modelscope', 'none'],
                       help='镜像源选择: hf-mirror (Hugging Face镜像站，推荐), modelscope (魔塔社区), none (原始源)')
    
    args = parser.parse_args()
    
    print(f"使用镜像源: {args.mirror}")
    if args.mirror in MIRROR_CONFIGS:
        print(f"  {MIRROR_CONFIGS[args.mirror]['description']}")
    print()
    
    if args.model.startswith('llama'):
        # LLaMA模型
        if args.model == 'llama2-7b':
            model_name = 'meta-llama/Llama-2-7b-hf'
        else:
            model_name = args.model
        
        download_llama_model(
            model_name=model_name,
            save_dir=args.save_dir,
            use_auth_token=args.hf_token,
            mirror=args.mirror
        )
    else:
        # GPT-2模型
        download_gpt2_model(
            model_name=args.model,
            save_dir=args.save_dir,
            mirror=args.mirror
        )
    
    print("\n模型下载完成！")
    print(f"您可以在 {args.save_dir} 目录中找到下载的模型")

if __name__ == "__main__":
    main()

