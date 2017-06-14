#coding=utf-8
#author=godpgf
import pandas as pd
import numpy as np
from stdb import *
from portfolio import MaxSharpePortfolio, MinVariancePortfolio
import matplotlib.pyplot as plt #绘图

def get_stock_data(code, dataProxy, dt='2015-12-31', bar_count = 230):
    st = dataProxy.history(code, pd.Timestamp(dt),bar_count,'1d','close')
    st.name = code
    return st

if __name__ == '__main__':
    stock = ['1000413', '1000063', '1002007', '1000001', '1000002']
    dataProxy = LocalDataProxy()
    stock_list = [(st, get_stock_data(st, dataProxy)) for st in stock]
    msp = MaxSharpePortfolio()
    mvp = MinVariancePortfolio()
    for st in stock_list:
        msp.add_stock(st[0], st[1])
        mvp.add_stock(st[0], st[1])
    msp_weights = msp.create()
    mvp_weights = mvp.create()

    # 用蒙特卡洛模拟产生大量随机组合----------------------------------------
    returns = msp.get_returns()
    port_returns = []
    port_variance = []
    for p in range(4000):
        weights = np.random.random(len(stock))
        weights /= np.sum(weights)
        port_returns.append(np.sum(returns.mean() * 252 * weights))
        port_variance.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))

    port_returns = np.array(port_returns)
    port_variance = np.array(port_variance)

    def statistics(weights):
        weights = np.array(weights)
        port_returns = np.sum(returns.mean() * weights) * 252
        port_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return np.array([port_returns, port_variance, port_returns / port_variance])


    plt.figure(figsize=(8, 4))
    # 圆圈：蒙特卡洛随机产生的组合分布
    plt.scatter(port_variance, port_returns, c=port_returns / port_variance, marker='o')
    # 叉号：有效前沿
    #plt.scatter(target_variance, target_returns, c=target_returns / target_variance, marker='x')
    # 红星：标记最高sharpe组合
    plt.plot(statistics(msp_weights)[1], statistics(msp_weights)[0], 'r*', markersize=15.0)
    # 黄星：标记最小方差组合
    plt.plot(statistics(mvp_weights)[1], statistics(mvp_weights)[0], 'y*', markersize=15.0)
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    print 'finish'
