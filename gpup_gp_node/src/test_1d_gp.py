import numpy as np
from gp import GaussianProcess
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import GPy

np.random.seed(145453)

def func(x, v = 0.3):
    # return x*np.sin(x)+np.random.normal(0, v)
    return 3*x+4+np.random.normal(0, v)

var = 0.08

# x_data = np.linspace(0, 4, 5).reshape(-1,1)
x_data = np.random.uniform(0, 6, 1000).reshape(-1,1)
y_data = np.array([func(i, np.sqrt(var)) for i in x_data]) #


x_real = np.linspace(0, 6, 100).reshape(-1,1)
y_real = np.array([func(i, 0) for i in x_real]) 

gp_est = GaussianProcess(x_data, y_data.reshape((-1,)), optimize = True, theta=None, algorithm = 'Matlab')

x_n = np.array([3.0])
m, s = gp_est.predict(x_n)

# print(m,s)

x_new = np.linspace(0, 6, 100).reshape(-1,1)
means = np.empty(100)
variances = np.empty(100)
for i in range(100):
    means[i], variances[i] = gp_est.predict(x_new[i])
# print(np.mean(variances))

plt.plot(x_data, y_data, '+k')
plt.plot(x_real, y_real, '--k')
plt.plot(x_n, m, '*y')
plt.errorbar(x_n, m, yerr=np.sqrt(s), ecolor='y')
msl = (means.reshape(1,-1)[0]-np.sqrt(variances))#.reshape(-1,1)
msu = (means.reshape(1,-1)[0]+np.sqrt(variances))#.reshape(-1,1)[0]
plt.plot(x_new, means,'-r')
plt.fill_between(x_new.reshape(1,-1)[0], msl, msu)
plt.ylabel('f(x)')
plt.title('Noise variance: %f, Avg. prediction variance: %f'%(var, np.mean(variances)))

plt.show()