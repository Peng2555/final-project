"""
项目安装配置文件
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stock-price-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="基于大模型微调的股票价格预测项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-price-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "stock-download=download:main",
            "stock-train=train:main",
            "stock-predict=framework:main",
        ],
    },
)

