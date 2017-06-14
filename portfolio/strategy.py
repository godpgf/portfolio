#coding=utf-8
#author=godpgf
import pandas as pd
import numpy as np
#最优化投资组合的推导是一个约束最优化问题
import scipy.optimize as sco


class BasePortfolio(object):

    def __init__(self):
        self.stock_pool = list()

    def add_stock(self, code, close, expect_score = None):
        close.name = code
        st_des = {
            'code':code,
            'close':close,
            'expect_score':expect_score
        }
        self.stock_pool.append(st_des)

    def get_returns(self):

        d = pd.DataFrame([st_des['close'] for st_des in self.stock_pool])
        ##转置
        data = d.T
        returns = np.log(data / data.shift(1))
        return returns


class MinVariancePortfolio(BasePortfolio):

    def create(self):
        returns = self.get_returns()

        noa = len(returns.T)
        # 约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # 我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
        bnds = tuple((0, 1) for x in range(noa))

        returns_cov = returns.cov()
        def min_variance(weights):
            weights = np.array(weights)
            port_variance = np.sqrt(np.dot(weights.T, np.dot(returns_cov * 252, weights)))
            return port_variance

        optv = sco.minimize(min_variance, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)

        return optv['x']


class MaxSharpePortfolio(BasePortfolio):

    def create(self):
        returns = self.get_returns()

        noa = len(returns.T)
        # 约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # 我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
        bnds = tuple((0, 1) for x in range(noa))

        returns_cov = returns.cov()
        returns_mean = returns.mean()
        returns_expect = np.array([self.stock_pool[i]['expect_score'] if self.stock_pool[i]['expect_score'] else returns_mean[i] for i in range(noa)])

        def min_sharpe(weights):
            weights = np.array(weights)
            port_returns = np.sum(returns_expect * weights) * 252
            port_variance = np.sqrt(np.dot(weights.T, np.dot(returns_cov * 252, weights)))
            return -port_returns / port_variance

        opts = sco.minimize(min_sharpe, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)

        return opts['x']