"""
股票数据下载脚本
从Yahoo Finance下载股票历史数据
"""
import yfinance as yf
import pandas as pd
import argparse
from datetime import datetime, timedelta

def download_stock_data(ticker='AAPL', period='1y', output='sample_test.csv', 
                        start_date=None, end_date=None):
    """
    下载股票历史数据
    
    Args:
        ticker: 股票代码（如 'AAPL' 表示苹果公司）
        period: 数据周期（'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'）
        output: 输出CSV文件路径
        start_date: 开始日期（格式：'YYYY-MM-DD'），如果提供则忽略period
        end_date: 结束日期（格式：'YYYY-MM-DD'），如果提供则忽略period
    """
    print(f"正在下载 {ticker} 的股票数据...")
    
    try:
        # 下载数据
        if start_date and end_date:
            print(f"日期范围: {start_date} 到 {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date)
        else:
            print(f"周期: {period}")
            data = yf.download(ticker, period=period)
        
        if data.empty:
            raise ValueError(f"未能下载到 {ticker} 的数据，请检查股票代码是否正确")
        
        # 重置索引以包含Date列
        data.reset_index(inplace=True)
        
        # 选择相关列（匹配格式要求）
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns[1:]):
            raise ValueError("下载的数据缺少必要的列")
        
        data = data[required_columns]
        
        # 保存到CSV
        data.to_csv(output, index=False)
        print(f"数据已下载并保存到: {output}")
        print(f"数据行数: {len(data)}")
        print(f"日期范围: {data['Date'].min()} 到 {data['Date'].max()}")
        
        return data
    
    except Exception as e:
        print(f"下载数据时出错: {str(e)}")
        raise

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='下载股票历史数据')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='股票代码（如 AAPL, TSLA, MSFT 等）')
    parser.add_argument('--period', type=str, default='1y',
                       choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
                       help='数据周期')
    parser.add_argument('--start', type=str, default=None,
                       help='开始日期（格式：YYYY-MM-DD），如果提供则忽略period')
    parser.add_argument('--end', type=str, default=None,
                       help='结束日期（格式：YYYY-MM-DD），如果提供则忽略period')
    parser.add_argument('--output', type=str, default='sample_test.csv',
                       help='输出CSV文件路径')
    
    args = parser.parse_args()
    
    download_stock_data(
        ticker=args.ticker,
        period=args.period,
        output=args.output,
        start_date=args.start,
        end_date=args.end
    )

if __name__ == "__main__":
    main()
