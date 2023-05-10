import numpy as np
# import math
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

# 生成100个数据,随机种子seed=5
np.random.seed(5)
X = np.random.uniform(0,1,100)
e = np.random.normal(0,0.2,100)
Y = np.sin(2*np.pi*X) + e
plt.scatter(X,Y)

'''问题1：Fit the data using KNN with k=1'''
#定义KNN算法
def KNN(X, Y, X_new, k):
    # 遍历所有点
    Y_ba = []
    for i in range(len(X_new)):
        Dis = np.sqrt((X_new[i]-X)**2)
        # 获取最近k个数据点的索引
        df_dis = pd.DataFrame(Dis, columns=['Dis'])
        index = df_dis.sort_values(by='Dis', ascending=True).head(k).index
        # 计算xi点的估计值yi
        yi_ba = np.average(Y[index])
        Y_ba.append(yi_ba)
    Y_ba = np.array(Y_ba)
    return Y_ba

Y_1 = KNN(X, Y, X, 1)
plt.scatter(X,Y,marker='^',color='chocolate')
df_1 = pd.DataFrame({'X':X,
                    'Y_估计':Y_1})
df_sort1 = df_1.sort_values(by='X', ascending=True)
plt.plot(df_sort1['X'],df_sort1['Y_估计'], color='steelblue')
plt.xlabel('X')
plt.ylabel('Y')

'''问题2：Fit the data using KNN with k=100'''
Y_2 = KNN(X, Y, X, 100)
plt.scatter(X,Y,marker='^',color='chocolate')
df_2 = pd.DataFrame({'X':X,
                    'Y_估计':Y_2})
df_sort2 = df_2.sort_values(by='X', ascending=True)
plt.plot(df_sort2['X'],df_sort2['Y_估计'], color='steelblue')
plt.xlabel('X')
plt.ylabel('Y')

'''Fit the data using KNN with optimal k 
Using 10-fold cross validation to find optimal k'''
#重新定义KNN算法
def KNNRegressor(train_X, train_Y, test_X, test_Y, k):
    # 遍历所有点
    Y_ba = []
    for i in range(len(test_X)):
        Dis = np.sqrt((train_X-test_X[i])**2)
        # 获取最近k个数据点的索引
        df_dis = pd.DataFrame(Dis, columns=['Dis'])
        index = df_dis.sort_values(by='Dis', ascending=True).head(k).index
        # 计算xi点的估计值yi
        yi_ba = np.average(train_Y[index])
        Y_ba.append(yi_ba)
    Y_ba = np.array(Y_ba)
    TestError = np.average((test_Y-Y_ba)**2)
    return Y_ba, TestError

# select optimal k
def selectk(X, Y, max_k):
    # use 10-fold cross validation
    best_Error = float('inf')
    best_k = 0
    Error = []
    for k in range(1, max_k+1):
        kf = KFold(n_splits = 10, shuffle=True, random_state=5)
        index = np.arange(0,100)
        total_Error = 0
        for Trindex, Tsindex in kf.split(index):
            X_train = X[Trindex]
            X_test = X[Tsindex]
            Y_train = Y[Trindex]
            Y_test = Y[Tsindex]
            Y_ba, TestError = KNNRegressor(X_train, Y_train, X_test, Y_test, k)
            total_Error += TestError
        Error.append(total_Error)
        if best_Error>total_Error:
            best_k = k
            best_Error = total_Error
    plt.plot(np.arange(1,max_k+1), Error)
    print('The optimal k is {}'.format(best_k))
    return best_k

selectk(X, Y, 100)

# 定义plot函数
def fitplot(X, Y, Y_ba, k):
    plt.scatter(X,Y,marker='^',color='chocolate')
    df = pd.DataFrame({'X':X,
                         'Y_估计':Y_ba})
    df_sort = df.sort_values(by='X', ascending=True)
    plt.plot(df_sort['X'],df_sort['Y_估计'], color='steelblue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('k={}'.format(k))
    plt.savefig('fig_k={}.jpg'.format(k))

Y_3 = KNN(X, Y, X, 8)
fitplot(X, Y, Y_3, 8)


'''问题4：Using linear smoother estimator with wieght 
10-fold cross validation to find optimal bandwith h'''
# kernel regression using Gaussian kernel
def KernelRegressor(X, Y, X_new, Y_new, h):
    Y_ba = []
    for i in range(len(X_new)):
        w = np.exp(-((X_new[i]-X)**2)/(2*h))
        w_sum = sum(w)
        w_guass = w/sum(w)
        Yi_ba = np.dot(w_guass, Y)
        Y_ba.append(Yi_ba)
    TestError = np.average((Y_new-Y_ba)**2)
    return Y_ba, TestError

# select optimal bandwidth h
def selecth(X, Y, max_h):
    # use 10-fold cross validation
    best_Error = float('inf')
    best_h = 0
    Error = []
    h_range = np.arange(0.0005, max_h, 0.0001)
    for h in h_range:
        kf = KFold(n_splits = 10, shuffle=True, random_state=5)
        index = np.arange(0,100)
        total_Error = 0
        for Trindex, Tsindex in kf.split(index):
            X_train = X[Trindex]
            X_test = X[Tsindex]
            Y_train = Y[Trindex]
            Y_test = Y[Tsindex]
            Y_ba, TestError = KernelRegressor(X_train, Y_train, X_test, Y_test, h)
            total_Error += TestError
        Error.append(total_Error)
        if best_Error>total_Error:
            best_h = h
            best_Error = total_Error
    plt.plot(h_range, Error)
    print('The optimal h is {}'.format(best_h))
    return best_h

selecth(X,Y,0.01)

# 重新定义plot函数，更具普适性
def fitplot2(X, Y, X_new, Y_new, Regressor, a):
    if Regressor == 'KNNRegressor':
        Y_ba, Error = KNNRegressor(X, Y, X_new, Y_new, a)
        a_index = 'k'
    elif Regressor == 'KernelRegressor':
        Y_ba, Error = KernelRegressor(X, Y, X_new, Y_new, a)
        a_index = 'h'
    else:
        return 'The regressor does not exist.'
    plt.scatter(X,Y,marker='^',color='chocolate')
    df = pd.DataFrame({'X':X,'Y_估计':Y_ba})
    df_sort = df.sort_values(by='X', ascending=True)
    plt.plot(df_sort['X'],df_sort['Y_估计'], color='steelblue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('{}  {}={}'.format(Regressor, a_index, a))
    plt.savefig('{}_{}={}.jpg'.format(Regressor, a_index, a))

fitplot2(X,Y,X,Y,'KernelRegressor',0.0014)