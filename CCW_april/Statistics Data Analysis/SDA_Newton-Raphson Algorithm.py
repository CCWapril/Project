from math import *
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

'''Newton-Raphson Algorithm'''
# define the function
def func(x):
    return x - 2*sp.sin(x) + 1
# define the algrithm
def  Newton(fun, x0=1):
    e = abs(fun(x0))
    x = sp.symbols('x')
    dy = sp.diff(fun(x),x)
    while e > 0.0001:
        x1 = float(x0 - fun(x0)/dy.subs(x,x0))
        x0 = x1
        e = abs(fun(x1))
    return x1
# run
if __name__ == '__main__':
    x_sol = Newton(func)
    abs_error = abs(func(x_sol))
    print('The solution is {}, where the absolute error is {}.'.format(x_sol, abs_error))

'''Newton-Raphson Algorithm:gamma distribution'''

'''3.1 Generate 5000 observations from the Gamma distribution with α=β=5'''
n = 5000
np.random.seed(5)
data = np.random.gamma(5, 5, n)
plt.hist(data, 50, linewidth=2)

# calculate the likelihood function
sum_logy = np.sum(np.log(data))
sum_y = np.sum(data)

# define likelihood function
def likelihood(alpha,beta):
    l = -n*sp.log(sp.gamma(alpha))-n*alpha*sp.log(beta)+(alpha-1)*sum_logy-sum_y/beta
    return l

# define the algrithm
def  NewtonMultiple(fun):
    e = float('inf')
    a0,b0= [1,1] # define the initial value
    a,b = sp.symbols('alpha beta')
    U = np.array([sp.diff(fun(a,b),a), sp.diff(fun(a,b),b)])
    H = np.array([[sp.diff(U[0],a),sp.diff(U[0],b)],
                 [sp.diff(U[1],a),sp.diff(U[1],b)]])
    while e > 0.00000001:
        # 赋值
        U_0 = np.zeros(2)
        H_0 = np.zeros([2,2])
        for i in range(2):
            U_0[i] = float(U[i].subs([(a,a0),(b,b0)]))
            for j in range(2):
                H_0[i][j] = float(H[i][j].subs([(a,a0),(b,b0)]))
        a1, b1 = np.array([a0,b0])-np.dot(np.linalg.inv(H_0),U_0)
        e = abs((a1-a0)/a0)+abs((b1-b0)/b0)
        a0, b0 =  a1, b1
    return a1, b1

NewtonMultiple(likelihood)

'''Simulation and Estimation of Poisson Regression'''

'''4.(1) Data Generating Process'''
# 生成数据x
n = 300
rou = 0.5
p = 3
cov = np.zeros([p,p])
# 生成方差协方差矩阵
for i in range(p):
    for j in range(p):
        cov[i][j] = rou**abs(i-j)
mean = np.zeros(p)
np.random.seed(5)
x = np.random.multivariate_normal(mean,cov,n)
# 生成数据y
y = []
beta = np.array([1,2,-1.5])
np.random.seed(5)
for i in range(n):
    y.append(np.random.poisson(np.exp(np.dot(x[i],beta))))
y = np.array(y)

'''4.(2) Estimation'''
def NewtonRaphson(U,H):
    e = float('inf')
    b0 = np.ones(3)
    while e > 0.001:
        # 赋值
        U_0 = np.zeros(3)
        H_0 = np.zeros([3,3])
        for i in range(3):
            U_0[i] = float(U[i].subs([(b[0],b0[0]),(b[1],b0[1]),(b[2],b0[2])]))
            for j in range(3):
                H_0[i][j] = float(H[i][j].subs([(b[0],b0[0]),(b[1],b0[1]),(b[2],b0[2])]))
        b1 = b0-np.dot(np.linalg.inv(H_0),U_0)
        e = sum(abs((b1-b0)/b0))
        b0 = b1
    return b1
b = sp.symbols('beta1 beta2 beta3')
l = 0 #likelihood fucntion
for i in range(n):
    l += -sp.exp(np.dot(b,x[i]))+np.dot(b,x[i])*y[i]-sp.log(factorial(y[i]))

U = [] # score funciton
H = [] # hessian matrix
for i in range(p):
    U.append(sp.diff(l,b[i]))
    H_i = []
    for j in range(p):
        H_i.append(sp.diff(U[i],b[j]))
    H.append(H_i)
U = np.array(U)
H = np.array(H)
result = NewtonRaphson(U,H)

'''4.(3) Deviance and deviance residuals'''
def deviance(y,y_pre):
    D = []
    for i in range(len(y)):
        D.append(2*(y[i]*np.log(y[i]/y_pre[i])-(y[i]-y_pre[i])))
    D = np.array(D)
    return D, sum(D)
def residual(y,y_pre):
    r = []
    D,sum_D = deviance(y,y_pre)
    for i in range(len(y)):
        r.append(np.sign(y[i]-y_pre[i])*np.sqrt(D[i]))
    r = np.array(r)
    return r
if __name__ == '__main__':
    y_pre = np.exp(np.dot(x,result)) # yi预测值
    #非零化处理
    y_nonzero = y[y!=0]
    y_pre_nonzero = y_pre[y!=0]
    D, D_sum = deviance(y_nonzero,y_pre_nonzero)
    r = residual(y_nonzero,y_pre_nonzero)
    print('The deviance is {:.4f}'.format(D_sum))
    plt.plot(r,'o-')

'''4.(4) Omitted Variable'''
def NewtonRaphson2(U,H):
    e = float('inf')
    b0 = np.ones(2)
    while e > 0.001:
        # 赋值
        U_0 = np.zeros(2)
        H_0 = np.zeros([2,2])
        for i in range(2):
            U_0[i] = float(U[i].subs([(b[0],b0[0]),(b[1],b0[1])]))
            for j in range(2):
                H_0[i][j] = float(H[i][j].subs([(b[0],b0[0]),(b[1],b0[1])]))
        b1 = b0-np.dot(np.linalg.inv(H_0),U_0)
        e = sum(abs((b1-b0)/b0))
        b0 = b1
    return b1

b = sp.symbols('beta1 beta2')
x_new = x[:,:2]
l2 = 0 #likelihood fucntion
for i in range(n):
    l2 += -sp.exp(np.dot(b,x_new[i]))+np.dot(b,x_new[i])*y[i]-sp.log(factorial(y[i]))

U2 = [] # score funciton
H2 = [] # hessian matrix
p = 2
for i in range(p):
    U2.append(sp.diff(l2,b[i]))
    H2_i = []
    for j in range(p):
        H2_i.append(sp.diff(U2[i],b[j]))
    H2.append(H2_i)
U2 = np.array(U2)
H2 = np.array(H2)
result2 = NewtonRaphson2(U2,H2)
if __name__ == '__main__':
    y_pre = np.exp(np.dot(x_new,result2)) # yi预测值
    #非零化处理
    y_nonzero = y[y!=0]
    y_pre_nonzero = y_pre[y!=0]
    D, D_sum = deviance(y_nonzero,y_pre_nonzero)
    r = residual(y_nonzero,y_pre_nonzero)
    print('The deviance is {:.4f}'.format(D_sum))
    plt.plot(r,'o-')
    print('Since the model exists specification error, the residual is larger than 4.(3).')