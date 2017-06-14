#coding=utf-8
#author=godpgf
import pandas as pd
import numpy as np
#最优化投资组合的推导是一个约束最优化问题
import scipy.optimize as sco


class BasePortfolio(object):

    def __init__(self):
        self.stock_pool = list()

    def add_stock(self, code, data, min_expect_return, max_expect_return):
        st_des = {
            'code':code,
            'data':data,
            'min_expect_return':min_expect_return,
            'max_expect_return':max_expect_return
        }
        self.stock_pool.append(st_des)

    def get_returns(self):
        def filter_stock_data(code, st):
            st = st['Close']
            st.name = code
            return st

        d = pd.DataFrame([filter_stock_data(st_des['code'], st_des['data']) for st_des in self.stock_pool])
        ##转置
        data = d.T
        returns = np.log(data / data.shift(1))
        return returns

    def get_expect_returns(self):
        return [(st_des['min_expect_return'], st_des['max_expect_return']) for st_des in self.stock_pool]


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
        #returns_expect = returns.mean()
        returns_expect = self.get_expect_returns()
        returns_expect = np.array([(re[0] + re[1]) * 0.5 / (re[1] - re[0]) for re in returns_expect])

        def min_sharpe(weights):
            weights = np.array(weights)
            port_returns = np.sum(returns_expect * weights) * 252
            port_variance = np.sqrt(np.dot(weights.T, np.dot(returns_cov * 252, weights)))
            return -port_returns / port_variance

        opts = sco.minimize(min_sharpe, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)

        return opts['x']