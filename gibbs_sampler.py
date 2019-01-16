######################################################
#                    gibbs_sampler.py                #
#                                                    #
# Contains functions to run the Gibbs sampler MCMC   #
# described in Kelly (2007) for linear regression    #
#                                                    #
# Author: Ryan Rubenzahl                             #
# Last Update: 1/16/2019                             #
######################################################


import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

from IPython.display import clear_output



def gibbs(x, y, xy_err, init, hypers, num_iters):
    '''
    Gibbs sampler to estimate the posterior distribution
    given the data x, y and the measurement errors xy_err

    Params:
    x, y (N,): The data
    xy_err (2,2): Error matrix for the data ([[sigma_x, sigma_xy], [sigma_xy, sigma_y]])
    init (array) The initial guesses for the parameters
    hypers (array) The initial guesses for the hyperparameters
    num_iters (int) Number of iterations to run the Gibbs sampler for

    Returns:
    Chains (num_iters, nparams) A dataframe containing the Gibbs samples at each iteration
    '''
        
    assert (len(y) == len(x))
    assert (len(y) == xy_err.shape[-1]) and (len(x) == xy_err.shape[-1])
    
    # Unpack initial guesses
    theta, psi, G, eta = init
            
    # Unpack hypers
    mu0, u2, w2 = hypers
    
    # Initialize array to store the results of each iteration
    chains = []
    
    # Iterate the sampler
    for it in range(num_iters):
        
        if it % 100 == 0:
            clear_output(wait=True)
            sys.stdout.write('%d / %d' % (it+1, num_iters))
            
        # Unpack parameters so we can keep updating them
        alpha, beta, sigma2 = theta
        pi, mu, tau2 = psi.T

        ######### 1. For non-detections, simulate missing data ########
        # I'm going to skip this for now, but to implement use
        # step 2 in Kelly's gibbs sampler outline (TODO)
        
        # Next, we draw values of xi and eta from the current model
        xi = draw_xi(x, y, xy_err, eta, G, theta, psi)

        # Similarly, draw values of eta
        eta = draw_eta(x, y, xy_err, xi, theta)
        
        # Next, if multiple Gaussians, draw new values of the Gaussian labels, G
        if psi.ndim > 1:
            G = draw_G(xi, psi)

        ######## 2. Draw values for theta ########
        # First, draw alpha and beta
        alpha, beta = draw_alpha_beta(xi, eta, theta)
        theta = np.asarray([alpha, beta, sigma2])
            
        # Then draw a new value for sigma^2
        sigma2 = draw_sigma2(xi, eta, theta)
        theta = np.asarray([alpha, beta, sigma2])
            
            
        ######## 3. Draw values for psi ########
        # First, draw new values for the group proportions, pi
        pi = draw_pi(G)
        psi = np.asarray([pi, mu, tau2]).T
            
        # Next, draw new values for mu_k
        mu = draw_mu(xi, G, psi, mu0, u2)
        psi = np.asarray([pi, mu, tau2]).T
            
        # Finally, draw a new value for tau^2
        tau2 = draw_tau2(xi, G, psi, w2)
        psi = np.asarray([pi, mu, tau2]).T
            
            
        ######## 4. Use new theta, psi to update priors ########
        # First up, the mean mu0
        mu0 = draw_mu0(psi, u2)
            
        # Next, u^2
        u2 = draw_u2(mu0, psi, w2)
            
        # And last, but not least, w^2
        w2 = draw_w2(u2, psi)
    
        # Save the results of this iteration in the chains
        chains.append([alpha, beta, sigma2, *pi, *mu, *tau2])
        
    chains = pd.DataFrame(chains)
    chains.columns = np.array(['alpha', 'beta', 'sigma2', # theta
                               # psi
                               *['pi%d'  %k for k in range(len(psi))], 
                               *['mu%d'  %k for k in range(len(psi))], 
                               *['tau2%d'%k for k in range(len(psi))]])
    
    return chains


def scaled_inv_chi2(w2, dof=1):
    '''
    Returns a draw from a scaled inverse chi^2
    distribution with dof degrees of freedom and
    scale parameter w^2
    '''
    return (w2*dof) / np.random.chisquare(dof)


def draw_xi(x, y, xy_err, eta, G, theta, psi):
    '''
    Draws values of xi from p(xi | x, y, eta, G, theta, psi)
    Assumes one independent variable.
    
    Parameters:
    x: (N,) the measured independent variable
    y: (N,) the measured dependent variable
    xy_err: (2,2,N) the error matrix for the each data point
            xy_err[0,0] = sigma_x, xy_err[0,1] = sigma_xy, ...
    eta: (N,) a list of dependent variable draws
    G:  (N,K) latent variable for class membership
    theta: list - regression parameters (alpha, beta, sigma^2)
    psi:   list - mixture model parameters (pi, mu, tau^2)
            pi (K,), mu (K,), tau2 (K,)
            
    Returns: 
    xi: (N,)  Random draws of xi for each of the data points
    '''
    
    # Unpack error matrix
    sigma_x  = xy_err[0,0]
    sigma_y  = xy_err[1,1]
    sigma_xy = xy_err[0,1]
    
    # Unpack model parameters
    alpha, beta, sigma2 = theta
    
    psi = np.asarray([psi for i in range(len(x))]) # simplify np math
    pi, mu, tau2 = psi.T

    # Compute correlation coefficient
    rho_xy = sigma_xy / (sigma_x * sigma_y)

    # Estimate the variances for each Gaussian for our draws
    sigma_xi_ik = 1/np.sqrt((1/(sigma_x**2*(1 - rho_xy**2)))
                           + (beta**2 / sigma2) + (1 / tau2))

    # Sum over k to get which Gaussian xi comes from
    sigma_xi_i = np.sqrt(np.sum(G * sigma_xi_ik.T**2, axis=1))

    # Now estimate the mean for the draw
    xi_xy_i = x + (sigma_xy/sigma_y**2)*(eta - y)
    xi_ik = sigma_xi_i**2 * (xi_xy_i/(sigma_x**2*(1-rho_xy**2))
                            + beta*(eta - alpha)/sigma2
                            + mu/tau2)
    # Sum over k for this too
    xi_i_hat = np.sum(G * xi_ik.T, axis=1)

    # And make the draws
    xi_i = np.random.normal(loc=xi_i_hat, scale=sigma_xi_i)
    
    return xi_i


def draw_eta(x, y, xy_err, xi, theta):
    '''
    Draws values of eta from p(xi | x, y, xi, theta)
    Assumes one independent variable.
    
    Parameters:
    x: (N,) the measured independent variable
    y: (N,) the measured dependent variable
    xy_err: (2,2,N) the error matrix for the each data point
            xy_err[0,0] = sigma_x, xy_err[0,1] = sigma_xy, ...
    xi: (N,) a list of independent variable draws
    theta: list - regression parameters (alpha, beta, sigma^2)
            
    Returns: 
    eta: (N,)  Random draws of eta for each of the data points
    '''
    
    # Unpack error matrix
    sigma_x  = xy_err[0,0]
    sigma_y  = xy_err[1,1]
    sigma_xy = xy_err[0,1]
    
    # Unpack model parameters
    alpha, beta, sigma2 = theta

    # Compute correlation coefficient
    rho_xy = sigma_xy / (sigma_x * sigma_y)

    # Estimate the variances for each Gaussian for our draws
    sigma_eta_i = 1/np.sqrt((1/(sigma_y**2*(1 - rho_xy**2)))+(1 / sigma2))

    # Now estimate the mean for the draw
    term1 = (y + sigma_xy*(xi-x)/sigma_x**2) / (sigma_y**2*(1-rho_xy**2))
    term2 = (alpha + beta*xi) / sigma2
    eta_i_hat = sigma_eta_i**2 * (term1 + term2)

    # And make the draws
    eta_i = np.random.normal(loc=eta_i_hat, scale=sigma_eta_i)
    
    return eta_i


def draw_G(xi, psi):
    '''
    Draws values of G from a multinomial distribution
    with m=1 and group probabilities q_k = p(G_ik=1 | xi_i, psi)
    Assumes one independent variable.
    
    Parameters:
    xi: (N,) a list of independent variable draws
    psi: list - mixture model parameters (pi, mu, sigma^2)
            
    Returns: 
    G: (N,K)  Random draws of G for each of the data points
    '''
    
    # Unpack model parameters
    pi, mu, tau2 = psi.T

    # Compute the group probabilities
    N = stats.norm.pdf
    numerator = np.asarray([pi*N(xi[i], loc=mu, scale=np.sqrt(tau2))
                                    for i in range(len(xi))])
    denominator = np.sum(numerator, axis=1)
    qk = [numerator[i] / denominator[i] for i in range(len(xi))]
    
    # Then draw the labels from the multinomial distribution
    G_i=np.array([np.random.multinomial(1,qk[i])for i in range(len(qk))])
       
    return G_i


def draw_alpha_beta(xi, eta, theta):
    '''
    Draws values of alpha and beta from p(alpha, beta | xi, eta, sigma^2)
    Assumes one independent variable.
    
    Parameters:
    xi:  (N,) a list of independent variable draws
    eta: (N,) a list of dependent variable draws
    theta: list - regression parameters (alpha, beta, sigma^2)
            (only need sigma^2 for this function)
            
    Returns: 
    alpha, beta: (float)  Random draw of alpha and beta from regression
    '''
    
    # Unpack model parameters
    alpha, beta, sigma2 = theta
    
    # Construct the matrix X, with 1st column of 1s, 2nd of xi values
    X = np.asarray([np.ones_like(xi), xi]).T
    
    # Variance and mean of normal to draw parameters from
    Sigma_c = np.linalg.inv(X.T @ X) * sigma2
    c = (Sigma_c/sigma2) @ X.T @ eta
    
    # Draw alpha and beta
    alpha, beta = np.random.multivariate_normal(c, Sigma_c)
    
    return alpha, beta


def draw_sigma2(xi, eta, theta):
    '''
    Draws values of sigma2 from p(sigma2 | xi, eta, alpha, beta)
    Assumes one independent variable.
    
    Parameters:
    xi:  (N,) a list of independent variable draws
    eta: (N,) a list of dependent variable draws
    theta: list - regression parameters (alpha, beta, sigma^2)
            (only need alpha, beta for this function)
            
    Returns: 
    sigma2: (float) Random draw of sigma^2 from the current model
    '''
    
    # Unpack model parameters
    alpha, beta, sigma2 = theta
    
    N = len(xi)
    
    # Scale and Ndof for sigma^2 distribution
    s2 = 1/(N-2) * np.sum((eta - alpha - beta * xi)**2)
    nu = N  - 2
    
    return scaled_inv_chi2(s2, dof=nu)


def draw_pi(G):
    '''
    Draws values of sigma2 from p(pi | G)
    
    Parameters:
    G: (N,K)  Random draws of G for each of the data points
            
    Returns: 
    pi: (K,) Random draws of pi from the current model
    '''

    # Number of points belonging to each Gaussian
    nk = np.sum(G, axis=0)
    
    return np.random.dirichlet(nk + 1)


def draw_mu(xi, G, psi, mu0, u2):
    '''
    Draws values of mu from p(mu | xi, G, tau2, mu0, u2)
    Assumes one independent variable.
    
    Parameters:
    xi and G should be familiar
    psi:   list - mixture model parameters (pi, mu, tau^2)
        (only need tau2 from this)
    mu0: (float) - The hyperparameter mu0
    u2: (float) - The hyperparameter u^2
            
    Returns: 
    mu: (K,)  Random draws of xi for the current model
    '''
    # Unpack
    pi, mu, tau2 = psi.T

    # Number of points belonging to each Gaussian
    nk = np.sum(G, axis=0)
    
    # Variance for the distribution of mu
    sigma_mu_k = 1/np.sqrt(1/u2 + nk/tau2)
    
    # Mean of the distribution
    # Paper has this as 1/nk, but nk can be zero if no points are in one of the Gaussians (ERROR?)
    xi_k_bar = (1/(nk+1))*np.sum([G[i]*xi[i] for i in range(len(xi))], axis=0)
    mu_k_hat = (mu0/u2 + nk*xi_k_bar/tau2) / (1/u2 + nk/tau2)
    
    # Make the draw
    mu_k = np.random.normal(loc=mu_k_hat, scale=sigma_mu_k)
    
    return mu_k


def draw_tau2(xi, G, psi, w2):
    '''
    Draws values of tau2 from p(tau2 | xi, G, mu, w2)
    Assumes one independent variable.
    
    Parameters:
    xi, G should be familiar
    psi:   list - mixture model parameters (pi, mu, tau^2)
        (only need mu from this)
    w2: (float) - The hyperparameter w^2
        
    Returns: 
    tau2: (K,)  Random draws of tau^2 for the current model
    '''
    # Unpack
    pi, mu, tau2 = psi.T

    # Number of points belonging to each Gaussian
    nk = np.sum(G, axis=0)

    # Scale and Ndof for distribution
    t2_k = 1/(nk+1) * (w2 
          + np.sum([G[i]*(xi[i]-mu)**2 for i in range(len(xi))], axis=0))
    nu_k = nk + 1
    
    # Make the draw
    tau2_k = scaled_inv_chi2(t2_k, dof=nu_k)
    
    return tau2_k


def draw_mu0(psi, u2):
    '''
    Draws values of mu0 from p(mu0 | mu, u2)
    Assumes one independent variable.
    
    Parameters:
    psi:   list - mixture model parameters (pi, mu, tau^2)
        (only need mu from this)
    u2: (float) - The hyperparameter u^2
        
    Returns: 
    mu0: (float)  Random draws of mu0 for the current model
    '''
    # Unpack
    pi, mu, tau2 = psi.T
    K = len(pi)
    
    # This one is straightforward
    mu_bar = (1/K) * np.sum(mu)
    
    return np.random.normal(mu_bar, np.sqrt(u2/K))
    

def draw_u2(mu0, psi, w2):
    '''
    Draws values of u^2 from p(u^2 | mu0, mu, w2)
    Assumes one independent variable.
    
    Parameters:
    m0: (float) - The hyperparameter m0
    psi:   list - mixture model parameters (pi, mu, tau^2)
        (only need mu from this)
    w2: (float) - The hyperparameter w^2
        
    Returns: 
    u2: (float)  Random draws of u^2 for the current model
    '''
    
    # Unpack
    pi, mu, tau2 = psi.T
    K = len(pi)
    
    # Ndof and scale parameter
    nu_u = K + 1
    uhat2 = (1/nu_u) * (w2 + np.sum((mu - mu0)**2))
    
    return scaled_inv_chi2(uhat2, dof=nu_u)


def draw_w2(u2, psi):
    '''
    Draws values of w^2 from p(u^2 | u^2, tau^2)
    Assumes one independent variable.
    
    Parameters:
    u2: (float) - The hyperparameter u^2
    psi:   list - mixture model parameters (pi, mu, tau^2)
        (only need tau2 from this)
        
    Returns: 
    w2: (float)  Random draws of w^2 for the current model
    '''
    
    # Unpack
    pi, mu, tau2 = psi.T
    K = len(pi)
    
    # Straightforward
    a = 0.5*(K + 3)
    b = 0.5*((1/u2) + np.sum(1/tau2))
    
    return np.random.gamma(a, scale=b)
