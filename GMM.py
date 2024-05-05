# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:49:57 2023

@author: anbarani
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import chi2

def plot_ellipsoid(ax, mean, cov, color):
    
    """
    plot gaussian ellipsoids, h and w represents sigma
    pl is plt.gca() or ax object
    """
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi # convert to degrees
    # ellispoid axes as sigma
    ell = mpl.patches.Ellipse (mean, 2*np.sqrt(v[0]), 2*np.sqrt(v[1]), angle = 180 + angle, color=color)
    # ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)

def likelyhood(X, mean, cov, pi):
    """
    GMM: x - observable, z - latent 
    calculates total log-likelyhood
    sum_over_samples{ ln(marginal_p_x) }   
    
    X = 2D data (n_samples, n_features)
    mean = 2D data (K, n_features)
    cov = square matrix (n_features, n_features)
    pi = row-vector (1, K)
    
    """
    
    return np.mean(np.log(marginal_p_x(X, mean, cov, pi)))
    
def marginal_p_x(X, mean, cov, pi):
    
    """
    GMM: x - observable, z - latent 
    calculates marginal p(x) of GM:
    p(x) = sum_over_k{ p(x|z = k)*p(z = k) }
    
    X = 2D data (n_samples, n_features)
    mean = 2D data (K, n_features)
    cov = square matrix (n_features, n_features)
    pi = row-vector (1, K)
    Output: 1D array with length of n_samples
    """
    assert(X.shape[1] == mean.shape[1])
    assert(len(cov) == len(pi))
    assert(mean.shape[0] == len(pi))
    
    n_samples, n_features = np.shape(X)
    K = len(pi)
    prob_matrix = np.zeros((n_samples, K))
    
    for k, pi_k in enumerate(pi):        
            
        prob_matrix[:, k] = pi_k * _gaussian(X, mean[k, :], cov[k])
            
    return np.sum(prob_matrix, axis = 1)

def responsibility(X, mean, cov, pi):
     """
     calculates p(z = k | X) = p(X | z = k)/marginal_p_x for all k
     
     X = 2D data (n_samples, n_features)
     mean = 2D data (K, n_features)
     cov = square matrix (n_features, n_features)
     pi = row-vector (1, K)
     Output: 2D array (n_samples, K)
     """
     n_samples, n_features = np.shape(X)
     K = len(pi)
     
     prob_matrix = np.zeros((n_samples, K))
     p_x = marginal_p_x(X, mean, cov, pi)
     
     for k, pi_k in enumerate(pi):      
         
         prob_matrix[:, k] = pi_k * _gaussian(X, mean[k, :], cov[k]) / p_x
         
     return prob_matrix            
         
def _gaussian(X, mean, cov):
    
    """
    Calculates gaussian for all samples and particular cluster k
    X = 2D data (n_samples, n_features)
    mean = row-vector (1, n_features)
    cov = square matrix (n_features, n_features)
    Output: 1D array with length of n_samples
    """
    assert(cov.shape[0] == cov.shape[1])
    
    n_features = X.shape[1]
    A = 1/((2*np.pi)**(n_features/2))*1/(np.sqrt(abs(np.linalg.det(cov))))

    return A*np.exp(-0.5*_mahal(X, mean, cov))                                                                              

def _mahal(X, mean, cov):
    
    """
    Calculates squared mahalanobis for all samples and particular cluster k
    X = 2D data (n_samples, n_features)
    mean = row-vector (1, n_features)
    cov = square matrix (n_features, n_features)
    Output: 1D array with length of n_samples
    """
    M1 = (X-mean).dot(np.linalg.inv(cov))
    return np.sum(M1*(X-mean), axis = 1)

def Outliers(X, gmm):
    """
    Identify outliers based on trained gmm
    Output: 1D array with outlier indexes
    """
    n_samples, n_features = np.shape(X)
    K = len(gmm.pi)
    cdfs = np.zeros((n_samples, K))
    
    for i in range(K):
        mahal = _mahal(X, gmm.mean[i, :], gmm.cov[i]) # mahal for all X w.r.t. cluster i
        cdfs[:, i] = chi2.cdf(mahal, df = n_features)
           
    cdf = np.sum((1 - cdfs) * gmm.pi, axis = 1) # CDF for all X
    cdf = cdf/np.max(gmm.pi)

    # resp = gmm.Compute_resp(X)
    # idx = np.argmax(resp, axis = 1)
    # cdf = 1 - cdfs[range(resp.shape[0]), idx]
    
    # cdf = np.sum(cdfs*resp, axis = 1) # CDF for all X
    
    idx = np.where(cdf < 0.05)[0]    
                    
    return idx
        
class GMM_custom:
    
    def __init__(self, K = 2, max_iter = 50, tol = 1e-3, init = 'random'):
              
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        
    def random_init(self, X):
        """
        mu, cov and pi initialization, random 
        """        
        _, n_features = X.shape
        self.mean = np.zeros((self.K, n_features))
        self.cov = [np.identity(n_features)] * self.K
        # self.pi = np.random.uniform(low = 0, high = 1, size = (self.K - 1, 1))
        self.pi = np.repeat(1 / self.K, self.K)

    
        for i in range(n_features):
            
            self.mean[:, i] = np.random.uniform(low = np.min(X[:,i]), high = np.max(X[:,i]), size = self.K)
            
    def kmeans_init(self, X):
        """
        mu, cov and pi initialization, k-means
        """  
        self.random_init(X)
        
        iter = 10
        J = np.zeros((iter, 1))

        for i in range(iter):

            # E-step
            #pairwise distances between X and mu array
            d = euclidean_distances(X, self.mean) # matrix n_samples x K
            R_nk = d == np.min(d, axis = 1).reshape(np.shape(d)[0], 1) # sorting
            J[i] = np.sum(d*R_nk)
            
            # M-step            
            self.mean = (R_nk.T @ np.array(X)) / np.sum(R_nk, axis = 0).reshape(self.K, 1)
               
        
    def _E_step(self, X):
        
        return responsibility(X, self.mean, self.cov, self.pi)
    
    def _M_step(self, X, resp):
        """
        M-step of EM algo
        
        X = 2D data (n_samples, n_features)
        resp = 2D data (n_samples, K)
        
        Output: 1D array with length of n_samples
        """       
        n_samples, _ = X.shape
        
        N_k = np.sum(resp, axis = 0) # effective number per cluster
        N = X.T.dot(resp)  # obtain n_features x K matrix
        mean_k = (N/N_k).T
        
        cov_k = []
        for k in range(self.K):
            
            X_shift = X - mean_k[k, :]
            resp_k = resp[:, k]
             
            cov_k.append( (X_shift.T * resp_k).dot(X_shift) / N_k[k])
            
        pi_k = N_k / n_samples    
        
        return mean_k, cov_k, pi_k
    
    def EM(self, X, plot = False):
        """
        Performs GMM based on X. 
        The final parameters are recorded in the class properties
        """
        if self.init == 'random':           
            self.random_init(X)
        elif self.init == 'kmeans':
            self.kmeans_init(X)
            
        J = [likelyhood(X, self.mean, self.cov, self.pi)]
        j = 0    
        if plot: # plot iterations
            m = 1
            n = 1
            fig = plt.figure()
            
        for it in range(self.max_iter):
                       
            if plot and X.shape[1] == 2: # plot iterations
                m = 2
                n = 3
                if (it == 0) or (it == 1) or (it == 3) or (it == 5) or (it == 19):
                    
                    ax = fig.add_subplot(m, n, j + 1)
                    ax.scatter(X[:, 0], X[:, 1])
                    for i in range(self.K):
                        plot_ellipsoid(ax, self.mean[i, :], self.cov[i], 'r')                                
                    j += 1
                    ax.set_title('Iteration' + str(it))
                    
                    
            #calculate responsibilities
            resp = self._E_step(X)
            #update parameters
            self.mean, self.cov, self.pi = self._M_step(X, resp)
            # append cost function aka likelyhood 
            J.append(likelyhood(X, self.mean, self.cov, self.pi))
            if abs((J[it + 1] - J[it]) / J[it]) < self.tol:
                print('Reached tol threshold')
                break
            
        if it == self.max_iter - 1:    
            print('Reached max iteration')   
        
        if plot:               
            ax1 = fig.add_subplot(m, n, m*n)
            ax1.scatter(range(len(J)), J)
            ax1.set_title('Log-likelyhood')
            ax1.set_xlabel('Iteration')
        
    def Compute_p_x(self, X):
        """
        Output: 1D array with length of n_samples
        """
        return marginal_p_x(X, self.mean, self.cov, self.pi)    

    def Compute_resp(self, X):
        """
        Output: 2D array (n_samples, K)
        """
        return responsibility(X, self.mean, self.cov, self.pi)
    
    def Compute_likelyhood(self, X):
        
        return likelyhood(X, self.mean, self.cov, self.pi)

if __name__== '__main__':
    
    """
    functions validation: generate data from multivariate gaussian and 
    check marginal_p_x, responsibility and _M_step outcome (should give true results)
    """
    K = 3
    n_samples = 1000
    n_features = 2
    
    pi = np.array([1/3, 1/3, 1/3])
    mean = np.array([[1, 2], [-1, -1], [-1.5, 2]])
    cov = [np.array([[0.5, -0.5], [-0.5, 1]]), np.array([[0.5, 0.2], [0.2, 0.5]]), np.array([[0.7, 0.35], [0.35, 0.3]])]
    X = []
    
    for i in range(K):
        X.append(np.random.multivariate_normal(mean[i, :], cov[i], size = n_samples))

    X = np.concatenate(X)
    
    # marginal probability
    plt.figure()
    ax = plt.scatter(X[:,0], X[:,1], c = marginal_p_x(X, mean, cov, pi))
    
    for i in range(K):       
        ax = plt.gca()   
        plot_ellipsoid(ax, mean[i, :], cov[i], 'r')
    
    # responsibilities
    plt.figure()
    ax = plt.scatter(X[:,0], X[:,1], c = responsibility(X, mean, cov, pi)[:, 1], cmap = 'jet')
    
    print('Likelyhood {}'.format(likelyhood(X, mean, cov, pi)))
    # make the M-step
    gmm = GMM_custom(K = K)
    check_mean, check_cov, check_pi = gmm._M_step(X, responsibility(X, mean, cov, pi))
    print('Mean_k {}'.format(check_mean))
    print('Cov_k {}'.format(check_cov))
    print('Pi_k {}'.format(check_pi))
    
    # algo validation
    gmm.EM(X, plot = True)
    print('Infered Mean_k {}'.format(gmm.mean))
    print('Infered Cov_k {}'.format(gmm.cov))
    print('Infered Pi_k {}'.format(check_pi))
    
        
