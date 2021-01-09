#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is a small project that implements the Sharpe ratio
(https://en.wikipedia.org/wiki/Sharpe_ratio) to a csv Dataframe and return
the stock with the best value. Data from the homonimous Datacamp project.
"""

# Importing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def computeSharpe(filename='datasets/stock_data.csv'):
    stock_data = pd.read_csv(filename, parse_dates=True, index_col='Date').dropna()
    stock_returns = stock_data.pct_change()
    excess_returns = stock_returns.sub(sp_returns, axis=0)
    avg_excess_return = excess_returns.mean()
    sd_excess_return = excess_returns.std()

    daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)
    annual_sharpe_ratio = daily_sharpe_ratio.mul(stock_data.shape[0])
    return annual_sharpe_ratio.idxmax()


if __name__ == "__main__":
    filename = input('Insert the name of the stock database or ENTER for the default value "stock_data.csv": ')    
    if filename == '': 
        filename = 'datasets/stock_data.csv'
    # Plot the most important features
    stock_data = pd.read_csv(filename, parse_dates=True, index_col='Date').dropna()
    benchmark_data = pd.read_csv('datasets/benchmark_data.csv', parse_dates=True, index_col='Date').dropna()
    stock_data.plot(subplots=1, title='Stock Data')
    benchmark_data.plot(subplots=1, title='Benchmark Data')

    # Computing and plotting daily price change
    stock_returns = stock_data.pct_change()
    sp_returns = benchmark_data['S&P 500'].pct_change()
    
    stock_returns.plot()
    Legend = legend=list(stock_data.columns)+list(benchmark_data.columns)
    sp_returns.plot(title='Daily price changes', legend=Legend)
    excess_returns = stock_returns.sub(sp_returns, axis=0)
    excess_returns.plot(title='Excess returns')    
    print('The stock to buy is: {}'.format(computeSharpe(filename)))
    


