# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 13:51:14 2024

@author: anbarani
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.linalg import inv
from scipy.special import digamma, gamma, gammaln
from GMM import GMM_custom, _mahal, plot_ellipsoid

def ln_Dir_norm(alpha):
    """
    Calulates ln(C(alpha)), where C is normalization for Dirichlet distribution
    """  
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

def ln_Wishart_norm(W, v):
    """
    Calulates ln(B(W, v)), where B is normalization for Wishart distribution
    """  
    
    D = W.shape[0]       
    A = 2**(v*D/2) * np.pi**(D*(D-1)/4) 
    B = np.sum( [gammaln(0.5*(v + 1 - i)) for i in range(D)] )
        
    return - 0.5*v*np.log(np.linalg.det(W)) - np.log(A) - B

class GMM_VB:
    
    def __init__(self, K = 6, max_iter = 100, tol = 1e-10):
              
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        
    def init(self, X, m_0 = None):

        """
        priors initialization, random 
        """        
        self.n_samples, self.n_features = X.shape
        
        # 0 < r_nk < 1, uniform distribution but sum(r_nk) = 1
        # r_nk: (n_samples, K)
        self.r_nk = np.array( [np.random.dirichlet(np.ones(self.K)) for _ in range(self.n_samples)] ) 
        # degrees of freedom for cov Wishart prior 
        self.v_0 = 1
        # mean prior for p(mu | cov)
        if m_0 is None:
            self.m_0 = np.zeros((self.K, self.n_features)) 
        else:
            self.m_0 = m_0
        # noise prior for p(mu | cov) 
        self.beta_0 = 1e-20
        # Dirichlet prior for pi
        self.alpha_0 = 0.1
        # Wishart W_0 prior
        self.W_0 = np.eye(self.n_features)
                      
    def statistics(self, X):
        
        """
        Input: X(n_samples, n_features)
        Outputs:
        - effective number of points for clusters N_k: (K, )
        - means x_k weighted with r_nk: (K, n_features)
        - cov S_k: np.array([S_1, ..., S_k])
        """
        N_k = np.sum(self.r_nk, axis = 0)
        x_k = self.r_nk.T.dot(X)
              
        S_k = []
        for k in range(self.K):
            
            if N_k[k] != 0:
                x_k[k, :] = x_k[k, :] / N_k[k]
            
            X_shift = X - x_k[k, :]
            r_nk_k = self.r_nk[:, k]
            cov = (X_shift.T * r_nk_k).dot(X_shift)
            
            if N_k[k] != 0:
                cov /= N_k[k]
            S_k.append(cov)
            
        return  N_k, x_k, S_k 
   
    def _posterior(self, X):
        
        N_k, x_k, S_k = self.statistics(X)
        
        assert(np.shape(N_k) == (self.K, ))
        assert(np.shape(x_k) == (self.K, self.n_features))
        assert(len(S_k) == self.K)
        assert(np.shape(S_k[0]) == (self.n_features, self.n_features))
        
        # posterior Dirichlet alpha, (K, )
        self.alpha_k = self.alpha_0 + N_k 
        
        # posterior p(mu | cov) noise, (K, )
        self.beta_k = self.beta_0 + N_k
        
        # posterior p(mu | cov) mean, (K, )
        self.m_k = self.m_0 * self.beta_0 / self.beta_k.reshape(-1, 1) + x_k * N_k.reshape(-1, 1) / self.beta_k.reshape(-1, 1)

        # posterior p(cov), Wishart matrix parameter W_k: np.array([W_1, ..., W_k])
        self.W_k = []
        self.W_k_inv = []
        for k in range(self.K):
            
            x_k_reshape = (x_k[k, :] - self.m_0[k, :]).reshape(-1, 1)
            W_inv = inv(self.W_0) + N_k[k]*S_k[k] + self.beta_0*N_k[k] / (self.beta_0 + N_k[k]) * x_k_reshape.dot(x_k_reshape.T)
            self.W_k.append(inv(W_inv))        
            self.W_k_inv.append(W_inv)  
            
        # posterior p(cov), Wishart degrees of freedom parameter
        self.v_k = self.v_0 + N_k
        
    def _E_step(self, X):
        """
        E-like step, Calculate expectations over mu_k, cov_k and pi_k  
        Input: X(n_samples, n_features)
        Outputs:
        - E_mu_cov: (n_samples, K)
        - E_cov: (K, )
        - E_pi_k: (K, )
        """
        # expectation over (mu_k, cov_k) 
        E_mu_cov = np.zeros((self.n_samples, self.K))
        E_cov = np.zeros(self.K)
        
        for k in range(self.K):
            
            E_mu_cov[:, k] = self.n_features * (self.beta_k[k])**(-1)
            X_shifted = X - self.m_k[k]
            M1 =  X_shifted.dot(self.W_k[k])
            M2 = np.sum(M1 * X_shifted, axis = 1)
            E_mu_cov[:, k] += self.v_k[k] * M2
            
            E_cov[k] = sum([digamma((self.v_k[k] + 1 - i) * 0.5) for i in range(self.n_features)])
            E_cov[k] += self.n_features*np.log(2) + np.log(np.linalg.det(self.W_k[k]))
        
        E_pi_k = digamma(self.alpha_k) - digamma(sum(self.alpha_k))
        
        return E_mu_cov, E_cov, E_pi_k
    
    def update_r_nk(self, E_mu_cov, E_cov):
        
        ln_ro_nk = -0.5 * E_mu_cov
        ln_ro_nk += 0.5 * E_cov
        ln_ro_nk += -0.5 * self.n_features * np.log(2*np.pi)
        ln_ro_nk += self.E_pi_k
        
        # normalization
        self.r_nk = np.exp(ln_ro_nk) / np.sum(np.exp(ln_ro_nk), axis = 1).reshape(-1, 1)
    
    def ELBO(self, X, E_cov):
        
        N_k, x_k, S_k = self.statistics(X)
        ELBO = 0
        H_q_cov = 0
        for k in range(self.K):
            
            ELBO_k = 0 
            x_k_shifted = x_k[k, :] - self.m_k[k, :]
            m_k_shifted = self.m_k[k, :] - self.m_0[k, :]
            
            # E [ln{ p(X | Z, mu, Cov) }]
            ELBO_k += E_cov[k] - self.n_features * (self.beta_k[k])**(-1) - self.v_k[k]*np.trace(S_k[k].dot(self.W_k[k])) 
            ELBO_k += - self.v_k[k] * np.sum(self.W_k[k].dot(x_k_shifted) * x_k_shifted)
            ELBO_k += - self.n_features * np.log(2*np.pi)
            ELBO_k *= 0.5*N_k[k]
            
            # E [ln{ p(mu , Cov) }]
            ELBO_k += 0.5*(self.n_features * np.log(self.beta_0/(2*np.pi)) + E_cov[k] - self.n_features * self.beta_0/self.beta_k[k])
            ELBO_k += - 0.5*(self.beta_0 * self.v_k[k] * np.sum(self.W_k[k].dot(m_k_shifted) * m_k_shifted))
            ELBO_k += ln_Wishart_norm(self.W_0, self.v_0) 
            ELBO_k += (self.v_0 - self.n_features - 1)/2 * E_cov[k]
            ELBO_k += - 0.5 * self.v_k[k] * np.trace(inv(self.W_0).dot(self.W_k[k]))
            
            ELBO += ELBO_k
            
            # Entropy of q(Cov)
            H_q_cov += - ln_Wishart_norm(self.W_k[k], self.v_k[k]) - (self.v_k[k] - self.n_features - 1)/2*E_cov[k] + self.v_k[k]*self.n_features/2
            
            
        # E [ln{ p(Z | pi) }]
        ELBO += np.sum(self.r_nk * self.E_pi_k)
       
        # E [ln{ p( pi ) }]
        ELBO += ln_Dir_norm( [self.alpha_0]*self.K ) + (self.alpha_0 - 1)*np.sum(self.E_pi_k)
        
        # E [ln{ q(Z) }]         
        ELBO -= np.sum(self.r_nk) 
       
        # E [ln{ q(pi) }]         
        ELBO -= np.sum(self.E_pi_k * (self.alpha_k - 1)) + np.log(ln_Dir_norm(self.alpha_k))
        
        # E [ln{ q(mu, Cov) }]    
        ELBO -= np.sum(0.5*E_cov + 0.5*self.n_features*np.log(self.beta_k/(2*np.pi))) - 0.5*self.K*self.n_features - H_q_cov
       
        return ELBO
    
    def run(self, X, m_0 = None, plot = False):
        """
        Performs Variational Inference gradient ascent steps, based on X. 
        The final parameters are recorded in the class properties
        """
        # initialize priors and r_nk
        self.init(X, m_0 = m_0)        
        j = 0    
        if plot: # plot iterations
            m = 1
            n = 1
            fig = plt.figure()               
        J = []
        self.weights = np.empty((0, self.K))
        
        for it in range(self.max_iter):

            # update posterior
            self._posterior(X)
                                             
            #E-like step
            E_mu_cov, E_cov, self.E_pi_k = self._E_step(X)
            
            if it == 0:
                J = [self.ELBO(X, E_cov)]
                 
            # update expectations of pi_k   
            self.weights = np.vstack((self.weights, self.alpha_k / np.sum(self.alpha_k)))
                               
            #update r_nk
            self.update_r_nk(E_mu_cov, E_cov)
            
            # append cost function aka likelyhood 
            J.append(self.ELBO(X, E_cov))
            flag = False
            if abs((J[it + 1] - J[it]) / J[it]) < self.tol:
                print('Reached tol threshold')
                flag = True
                
            elif it == self.max_iter - 1: 
                print('Reached max iteration')
                flag = True
            
            if plot and X.shape[1] == 2: # plot iterations
                m = 2
                n = 3
                if (it == 0) or (it == 10) or (it == 30) or (it == 50) or (flag):
                    
                    ax = fig.add_subplot(m, n, j + 1)
                    ax.scatter(X[:, 0], X[:, 1])
                    for k in range(self.K):
                        if self.weights[it, k] > 1e-3:
                            plot_ellipsoid(ax, self.m_k[k, :], self.W_k_inv[k] / self.v_k[k], 'r')                                
                    j += 1
                    ax.set_title('Iteration' + str(it))
            if flag:
                break
            
        if plot:               
            ax1 = fig.add_subplot(m, n, m*n)
            ax1.scatter(range(len(J)), J)
            ax1.set_title('ELBO')
            ax1.set_xlabel('Iteration')
            
if __name__== '__main__':
    
    """
    functions validation: generate data from multivariate gaussian and 
    check marginal_p_x, responsibility and _M_step outcome (should give true results)
    """
    
    np.random.seed(751)
    K = 3
    n_samples = 200
    n_features = 2

    mean = np.array([[1, 2], [-1, -1], [-1.5, 2]])
    cov = [np.array([[0.5, -0.5], [-0.5, 1]]), np.array([[0.5, 0.2], [0.2, 0.5]]), np.array([[0.7, 0.35], [0.35, 0.3]])]
    X = []
    
    for i in range(K):
        X.append(np.random.multivariate_normal(mean[i, :], cov[i], size = n_samples))

    X = np.concatenate(X)
    
    gmm = GMM_VB(K = 10, max_iter = 200)
    
    gmm_EM = GMM_custom(K = 10)
    gmm_EM.EM(X, plot = True)
    print('Infered Mean_k EM {}'.format(gmm_EM.mean))
    
    # algo validation
    # gmm.run(X, m_0 = gmm_EM.mean, plot = True)
    gmm.run(X, plot = True)
    print('Infered Mean_k {}'.format(gmm.m_k))
    # expected value of Wishart distribution
    # print('Infered Cov_k {}'.format([gmm.W_k_inv[i] / gmm.v_k[i] for i in range(gmm.K)] ))
    # expected value of Dirichlet distribution
    print('Infered Pi_k {}'.format(gmm.weights[-1,:]))
    N_k, x_k, S_k = gmm.statistics(X)
    
    # weights dynamics
    plt.figure()
    for k in range(gmm.K):
        plt.plot(gmm.weights[:, k])
    
    plt.xlabel('Iterations')
    plt.ylabel('Weights (expectations)')