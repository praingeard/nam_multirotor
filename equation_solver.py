import sympy
from sympy import Symbol, solve, simplify
from sympy.solvers.inequalities import reduce_inequalities
from numpy import sqrt
beta_0 = Symbol('beta0')
beta_1 = Symbol('beta1')
beta_2 = Symbol('beta2')
beta_3 = Symbol('beta3')
lambda_R = Symbol('lambdaR')
lambda_I = Symbol('lambdaI')
gamma_0 = Symbol('gamma_0')
gamma_1 = Symbol('gamma_1')

beta_0 = 1
beta_1 = 100
beta_2 = 20
beta_3 = 3
lambda_R = 3
lambda_I = 0

delta_1 = beta_1*beta_2-beta_0*beta_3
delta_2 = beta_2*beta_3*lambda_R - beta_1
delta_3 = beta_3**2*lambda_R-beta_2
delta_4 = (beta_1**2 - beta_0*beta_3**2)*lambda_R**2 - beta_0*delta_3*lambda_I**2

equation1 = (beta_1*beta_2*beta_3-beta_0*beta_3**2)*lambda_R-beta_1**2
equation2 = delta_3
equation3 = beta_2*delta_2 - delta_1
equation4 = delta_4
equation5 = delta_1*(delta_1*(delta_3*lambda_I**2 + beta_3**2*lambda_R**3) - beta_1**2*beta_3*lambda_R**2) - beta_0*beta_3**2*delta_4

equation1_2 = (gamma_1**2*lambda_R - gamma_0) * lambda_I**2 + gamma_1**2*lambda_R**3

print(reduce_inequalities([equation1 > 0],[beta_0]))

# beta_0 = 2
# beta_1 = 1
# beta_2 =0.3
# beta_3 =0.1
beta0 = 1
beta1 = 5
beta2 = 15
beta3 = 3
lambdaRlist = [3]
lambdaIlist = [0]
lambda_R = 3
lambda_I = 0
gamma_0 = 2.9
gamma_1 = 3.9
beta0list = []
beta1list = []
beta2list = []
beta3list = []
gamma0list = []
gamma1list = []

delta_1 = beta_1*beta_2-beta_0*beta_3
delta_2 = beta_2*beta_3*lambda_R - beta_1
delta_3 = beta_3**2*lambda_R-beta_2
delta_4 = (beta_1**2 - beta_0*beta_3**2)*lambda_R**2 - beta_0*delta_3*lambda_I**2

equation1 = (beta_1*beta_2*beta_3-beta_0*beta_3**2)*lambda_R-beta_1**2
equation2 = delta_3
equation3 = beta_2*delta_2 - delta_1
equation4 = delta_4
equation5 = delta_1*(delta_1*(delta_3*lambda_I**2 + beta_3**2*lambda_R**3) - beta_1**2*beta_3*lambda_R**2) - beta_0*beta_3**2*delta_4

equation1_2 = (gamma_1**2*lambda_R - gamma_0) * lambda_I**2 + gamma_1**2*lambda_R**3

print(equation1, equation2, equation3, equation4, equation5)


# for lambdaR in lambdaRlist:
#     for lambdaI in lambdaIlist:
#         beta0list.append(beta1*(-beta1 + beta2*beta3*lambdaR)/(beta3**2*lambdaR))
#         beta0list.append(beta2*(2*beta1 - beta2*beta3*lambdaR)/beta3)
#         beta0list.append(beta1**2*lambdaR**2/(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**2))
#         beta0list.append(beta1*(beta2*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3) - sqrt(-beta2*(2*beta1*beta2*beta3*lambdaI**2*lambdaR**2 - 2*beta1*beta3**3*lambdaI**2*lambdaR**3 - beta1*beta3**3*lambdaR**5 - beta1*beta3**3*lambdaR**4 + beta2**3*lambdaI**4 - 2*beta2**2*beta3**2*lambdaI**4*lambdaR - beta2**2*beta3**2*lambdaI**2*lambdaR**3 - beta2**2*beta3**2*lambdaI**2*lambdaR**2 + beta2*beta3**4*lambdaI**4*lambdaR**2 + beta2*beta3**4*lambdaI**2*lambdaR**4 + beta2*beta3**4*lambdaI**2*lambdaR**3 + beta2*beta3**4*lambdaR**5)))/(beta3*(-2*beta2*lambdaI**2 + 2*beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3 + beta3**2*lambdaR**2)))
#         beta1list.append(beta3*(beta2*lambdaR - sqrt(lambdaR*(-4*beta0 + beta2**2*lambdaR)))/2)
#         beta1list.append(beta3*(beta0 + beta2**2*lambdaR)/(2*beta2))
#         beta1list.append(-sqrt(beta0*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**2))/lambdaR)
#         beta1list.append((2*beta2*beta3*lambdaR**2*((27*beta0**2*beta3**4*lambdaR**4*(2*beta2*lambdaI**2 - 2*beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3 - beta3**2*lambdaR**2) + 18*beta0*beta2**2*beta3**2*lambdaR**2*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3)**2 - 2*beta2**4*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3)**3 + beta2*beta3**3*lambdaR**6*sqrt((-4*beta2**2*(6*beta0*beta3**2*lambdaR**2 + beta2**2*(beta2*lambdaI**2 - beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3))**3*(beta2*lambdaI**2 - beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3)**3 + (27*beta0**2*beta3**4*lambdaR**4*(2*beta2*lambdaI**2 - 2*beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3 - beta3**2*lambdaR**2) + 18*beta0*beta2**2*beta3**2*lambdaR**2*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3)**2 + 2*beta2**4*(beta2*lambdaI**2 - beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3)**3)**2)/(beta2**2*beta3**6*lambdaR**12)))/(beta2*beta3**3*lambdaR**6))**(1/3)*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3) - 2**(2/3)*beta3**2*lambdaR**4*((27*beta0**2*beta3**4*lambdaR**4*(2*beta2*lambdaI**2 - 2*beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3 - beta3**2*lambdaR**2) + 18*beta0*beta2**2*beta3**2*lambdaR**2*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3)**2 - 2*beta2**4*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3)**3 + beta2*beta3**3*lambdaR**6*sqrt((-4*beta2**2*(6*beta0*beta3**2*lambdaR**2 + beta2**2*(beta2*lambdaI**2 - beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3))**3*(beta2*lambdaI**2 - beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3)**3 + (27*beta0**2*beta3**4*lambdaR**4*(2*beta2*lambdaI**2 - 2*beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3 - beta3**2*lambdaR**2) + 18*beta0*beta2**2*beta3**2*lambdaR**2*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3)**2 + 2*beta2**4*(beta2*lambdaI**2 - beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3)**3)**2)/(beta2**2*beta3**6*lambdaR**12)))/(beta2*beta3**3*lambdaR**6))**(2/3) + 2*2**(1/3)*(6*beta0*beta3**2*lambdaR**2 - beta2**2*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3))*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3))/(6*beta3**2*lambdaR**4*((27*beta0**2*beta3**4*lambdaR**4*(2*beta2*lambdaI**2 - 2*beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3 - beta3**2*lambdaR**2) + 18*beta0*beta2**2*beta3**2*lambdaR**2*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3)**2 - 2*beta2**4*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3)**3 + beta2*beta3**3*lambdaR**6*sqrt((-4*beta2**2*(6*beta0*beta3**2*lambdaR**2 + beta2**2*(beta2*lambdaI**2 - beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3))**3*(beta2*lambdaI**2 - beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3)**3 + (27*beta0**2*beta3**4*lambdaR**4*(2*beta2*lambdaI**2 - 2*beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3 - beta3**2*lambdaR**2) + 18*beta0*beta2**2*beta3**2*lambdaR**2*(-beta2*lambdaI**2 + beta3**2*lambdaI**2*lambdaR + beta3**2*lambdaR**3)**2 + 2*beta2**4*(beta2*lambdaI**2 - beta3**2*lambdaI**2*lambdaR - beta3**2*lambdaR**3)**3)**2)/(beta2**2*beta3**6*lambdaR**12)))/(beta2*beta3**3*lambdaR**6))**(1/3)))
#         beta2list.append(beta0*beta3/beta1 + beta1/(beta3*lambdaR))
#         beta2list.append(beta3**2*lambdaR)
#         beta2list.append((beta1 - sqrt(-beta0*beta3**2*lambdaR + beta1**2))/(beta3*lambdaR))
#         #beta2list.append(lambdaR*(beta0*beta3**2*lambdaI**2 + beta0*beta3**2*lambdaR - beta1**2*lambdaR)/(beta0*lambdaI**2))
#         #beta2list.append((-2**(2/3)*beta1**2*lambdaI**4*(-27*beta0**2*beta3**4*lambdaR*(2*lambdaI**2 + lambdaR**2 + lambdaR)/(beta1**2*lambdaI**2) + sqrt(beta3**3*(beta3*(-27*beta0**2*beta1*beta3**2*lambdaI**4*lambdaR*(2*lambdaI**2 + lambdaR**2 + lambdaR) - 2*beta3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**3 + 9*lambdaI**2*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2))**2 - 4*(beta3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**2 - 3*lambdaI**2*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2))**3)/(beta1**6*lambdaI**12)) - 2*beta3**3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**3/(beta1**3*lambdaI**6) + 9*beta3**2*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2)/(beta1**3*lambdaI**4))**(2/3) + 2*beta1*beta3*lambdaI**2*((-27*beta0**2*beta1*beta3**4*lambdaI**4*lambdaR*(2*lambdaI**2 + lambdaR**2 + lambdaR) + beta1**3*lambdaI**6*sqrt(beta3**3*(beta3*(-27*beta0**2*beta1*beta3**2*lambdaI**4*lambdaR*(2*lambdaI**2 + lambdaR**2 + lambdaR) - 2*beta3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**3 + 9*lambdaI**2*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2))**2 - 4*(beta3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**2 - 3*lambdaI**2*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2))**3)/(beta1**6*lambdaI**12)) - 2*beta3**3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**3 + 9*beta3**2*lambdaI**2*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2))/(beta1**3*lambdaI**6))**(1/3)*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3) + 2*2**(1/3)*beta3*(-beta3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**2 + 3*lambdaI**2*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2)))/(6*beta1**2*lambdaI**4*((-27*beta0**2*beta1*beta3**4*lambdaI**4*lambdaR*(2*lambdaI**2 + lambdaR**2 + lambdaR) + beta1**3*lambdaI**6*sqrt(beta3**3*(beta3*(-27*beta0**2*beta1*beta3**2*lambdaI**4*lambdaR*(2*lambdaI**2 + lambdaR**2 + lambdaR) - 2*beta3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**3 + 9*lambdaI**2*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2))**2 - 4*(beta3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**2 - 3*lambdaI**2*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2))**3)/(beta1**6*lambdaI**12)) - 2*beta3**3*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)**3 + 9*beta3**2*lambdaI**2*(2*beta0*lambdaI**2 + beta1*beta3*lambdaI**2*lambdaR + beta1*beta3*lambdaR**3)*(2*beta0**2*beta3*lambdaI**2 + 2*beta0*beta1*beta3**2*lambdaI**2*lambdaR + 2*beta0*beta1*beta3**2*lambdaR**3 + beta1**3*lambdaR**2))/(beta1**3*lambdaI**6))**(1/3)))
#         beta3list.append(beta1*(beta2*lambdaR - sqrt(lambdaR*(-4*beta0 + beta2**2*lambdaR)))/(2*beta0*lambdaR))
#         beta3list.append(-sqrt(beta2/lambdaR))
#         beta3list.append(2*beta1*beta2/(beta0 + beta2**2*lambdaR))
#         beta3list.append(-sqrt((beta0*beta2*lambdaI**2 + beta1**2*lambdaR**2)/(beta0*lambdaR*(lambdaI**2 + lambdaR))))
#         #gamma0list.append(gamma_1**2*lambdaR*(lambdaI**2 + lambdaR**2)/lambdaI**2)
#         gamma1list.append(-lambdaI*sqrt(gamma_0/(lambdaR*(lambdaI**2 + lambdaR**2))))

#print(beta0list, beta1list, beta2list, beta3list, gamma0list, gamma1list)