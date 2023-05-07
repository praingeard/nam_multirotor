import matplotlib.pyplot as plt
import numpy as np 

V = np.linspace(0,3,100)
beta1min = 0.0
beta1bar = 0.25
beta1max = 0.5

a1 = (beta1max - beta1min - 2*beta1bar)/2
a2 = (beta1max + beta1min - 2*beta1bar)/2

beta_current = []
poly_current = []
alpha = 3*beta1max
beta = beta1min

def q_approx(x):
    a = 0.339
    b = 5.510
    #poly = 1/((1-a)*x + a*np.sqrt(x**2 + b)) * np.exp(-(x**2/2))/np.sqrt(2*np.pi)
    poly = 1/12*np.exp(-x**2/2) + 1/4*np.exp(-2*x**2/3)
    return poly

for v in V:
    beta_current.append(beta1bar + a2 + v*a1)
    #poly_current.append(q_approx(v))
    poly_current.append(alpha*q_approx(v) + beta)
print(V)
fig, ax = plt.subplots()
ax.plot(V,poly_current)
ax.set(xlabel='x', ylabel='CBS(x)')
plt.show()