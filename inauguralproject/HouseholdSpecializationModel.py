
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF
        

        # b. home production
        
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H= np.minimum(HM, HF)
        elif par.sigma == 0.1: #new 
            H= (1-par.alpha)*HM + par.alpha*HF + HM*HF
        else: 
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        
            



        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt


    def solve(self,do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        def objective(x):
            LM, LF, HF, HM = x
            value = self.calc_utility(*x) # * unpacks
            return value
        
    
        obj = lambda x: - 100*objective(x) #We make a positive monotone transformation, in order to ensure the stability of the SLSQP method, which is sensitive to starting values. we call a minimizer later, but we want to maximize so minus in front of obj. func. 
        guess = [(3,5.0,5.5,4.0)]
        bounds = [(0,24),(0,24),(0,24),(0,24)] 
        def con1(x):
            LM, HM= x
            return 24 - (LM + HM) # TM = LM + HM constraint - not to be broken
        def con2(x):
            LF, HF = x
            return 24 - (LF + HF) # TF = LF + HF contraint - not to be broken
        
        contraints = ({'type':'ineq', 'fun': con1},{'type':'ineq', 'fun': con2})

        #optimizer
        res = optimize.minimize(obj,
                                 guess,
                                 method='SLSQP',
                                 bounds=bounds 
                                 )
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        return opt


    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol
        for it, w in enumerate(par.wF_vec):
            par.wF = w 
            if discrete == True:
                res = self.solve_discrete()
            else:
                res = self.solve()
            sol.LM_vec[it] = res.LM
            sol.LF_vec[it] = res.LF
            sol.HM_vec[it] = res.HM
            sol.HF_vec[it] = res.HF

   
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None,do_print=False):
        """ estimate alpha and sigma """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        model = HouseholdSpecializationModelClass()
        model.par.alpha = alpha
        model.par.sigma = sigma

        # values from study
        beta0_study= 0.4
        beta1_study= -0.1

        # some initial guess for alpha and sigma values that minimize obj function
        alpha_guess = 0.5
        sigma_guess = 0.5
        guess = (alpha_guess, sigma_guess)

        # obj to be minimized
        def objective(x):
            par.alpha, par.sigma = x
            self.solve_wF_vec()
            self.run_regression()
            value = (beta0_study-sol.beta0)**2 + (beta1_study-sol.beta1)**2
            return value

        obj = lambda x: objective(x)
        
        # call minimizer and store results
        res = optimize.minimize(obj,
                                x0=guess,
                                method='Nelder-Mead'
                                )
        opt.alpha = res.x[0]
        opt.sigma = res.x[1]

        if do_print:
            for k,v in opt.__dict__.items():
                print(f'optimal {k} = {v:6.4f}')

        return opt
    