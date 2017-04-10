# -*- coding: utf-8 -*-

#MIT License

#Copyright (c) 2017 Marton Kelemen

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
#import numpy.linalg.inv
import collections
#import numpy.linalg.det
from scipy.stats import chisqprob


def REML_GWAS (y, K,  X = None, ngrids=100, llim=-10, ulim=10, esp=1e-10, eigenSummary = None) :
    
  # I) Data QC: determine if data passed in is valid to run algo
    if X == None:  # This algo will NOT run if there are no Fixed effects, so if there were none passed in, then we add an intercept ( a column of 1s)
        print("no fixed effects specified, adding an intercept")
        X =  np.ones( (y.shape[0], 1) ) # make sure that this is a 2D array, so that we can refer to shape[1]
        
    n = len(y)  # number of individuals in study
    t = K.shape[0]  # the row/col of the kinship matrix (should be same as n)
    q = X.shape[1]  # the number of fixed effect predictors

    if K.shape[1] != t : raise ValueError('malformed Kinship matrix (rows/cols dont equal)')   
    if X.shape[0] != n : raise ValueError('malformed fixed effects matrix (no enough observations)')  
   
 
  
    if  np.linalg.det( np.dot(X.T,X) ) == 0  :raise ValueError('X is singular')   # if the determinant of XtX is 0, then matrix is singular
        
  
  # II) main REML algo: NR with Grid search, performed on the likelihood expressed in terms of the eigen summary of the data to find Delta (Ve/Vg)
  # if no strain incidence matrix was passed in (IE it is a GWAS)
    if  eigenSummary == None :  # if cached Eigen values were not passed in
        eigenSummary = dataEigenSummary(K,X) # compute eigen summary of Kinship/Fixed effects: returns list with 2 elements, [0]  n-1 eigen valus, [1]:  n-1 eigenvectors, each with length of n
    
    etas = np.dot(eigenSummary.vectors.T, y)  # (n-1):1 column vector: matrix product of the t(EigenVctors) * Y, this is kindof like 'XtY', as we replaced the K and X with their eigen vectors

    

    logdelta = np.asarray(list(range(ngrids+1)) ) /ngrids*(ulim-llim)+llim # create a Grid for for the possible values for the ratio of genetic/random variance: ie with lower/upper bound of -10 to 10, this will go from -10 to +10, with steps of 0.2 (IE:  -10.0  -9.8  -9.6  -9.4 ... 9.2   9.4   9.6   9.8  10.0)
    delta = np.exp(logdelta)# bring back the delta onto non-log scale
    m = len(logdelta) # how many grid search points we have (determines the length of the main loop)
    # pre-compute ALL the scores for function, this is NOT the same as in paper, as it has an extra factor of delta
    dLL = logLikelihoodDerivative_all(logdelta,eigenSummary.values,etas) 
    
    optlogdelta = list()  # initialise results for the 'optimal Delta' values
    optLL = list()  # initialise an array for the likelihood for the above
               
    # Handle edge cases, where the first/last likelihoods are too small
    if  dLL[1] < esp  : # if the 1st element of the derivative of the log likelihood is less than convergence threshold
        optlogdelta.append(llim) # add Lower bound of log ratio of two variance components
        optLL.append(deltaLogLikelihood(llim,eigenSummary.values,etas))
    
    if  dLL[m-1] > (0-esp) : # if the last element of the derivative of the log likelihood is greater than neg convergence threshold
        optlogdelta.append(ulim) # add Upper bound of log ratio of two variance components
        optLL.append(deltaLogLikelihood(ulim,eigenSummary.values,etas))
      # normally neither of these should be true, this is just handling the edge cases, if there wouldn't be any valid solution within the upper-lower bounds: IE this will mean that the delta is then either -10 or +10, corresponding to 100% Vg, and 0% Vg
   
   
    for i in range(m-1) : # search Entire Grid (m-1 steps only, as we always search between current and current+1)
        #   also check if the diff between them is greater than threshold
        if   dLL[i]*dLL[i+1] < (0-esp*esp)  and  dLL[i] > 0  and  dLL[i+1] < 0  : # only attempt to find roots where sign flips between the 2 points (IE the function crosses the axis), : http://stackoverflow.com/questions/38961221/uniroot-solution-in-r
            # Find Root within range of current grid points (this is the NR step)
            r_root = uniroot_manual(deltaLogLikelihoodDerivative, lower=logdelta[i], upper=logdelta[i+1], lmbda=eigenSummary.values, etas=etas) # this is not a true NR, this simply 'zooms in' at the root via a simple bisection method
            optlogdelta.append(r_root) # save the root (IE which is the maxima as the function is a derivative, as derivative is 0 at stationary point, so their signs must differ at before/after) 
            optLL.append(deltaLogLikelihood(r_root,eigenSummary.values, etas)) # calculate the precise loglik for the solution, will be used to find the best if there are more than 1

        # this will normally only find a single solution

    
    # III) producing results for genetic/random variance componenets

    # handle edge case, if we couldn't find a single solution, then we make the assumption that this was due to no Vg (IE delta=Max, and likelihood = lowest)
    if(len(optLL) == 0) :
        maxdelta = np.max(delta) 
        maxLL = np.min(dLL)
        print("could not find solution to function, assume no Vg, and minimum likelihood")
    else :
        maxdelta = np.exp(optlogdelta[ np.argmax(optLL) ]) # find the delta (genetic/random variance ratio), which had the max likelihood
        maxLL = np.max(optLL)  # save the likelihood of the above as well

    maxva = np.sum(etas*etas/(eigenSummary.values+maxdelta))/(n-q)  # get the Genetic variance component sum eta^2 / eigenvalues + maxdelta over the DF: this is equivalent to R/DF = SSG/DF (IE the mean sqared error after Vg)
    maxve = maxva*maxdelta # calculate the radom variance (this is just a simple formula triangle, IE Ve = Vg * Ve/Vg, as Delta = Ve/Vg)
  

    return ( {"REML" : maxLL, "delta" : maxdelta, "ve" :maxve, "vg" :maxva } ) # return results


# Eigen decomposition summary of the kinship matrix + Fixed effects ( if X is 0, then this reduces to eigen(K,symmetric=TRUE) )
def dataEigenSummary(K, X) :
    n = X.shape[0]
    q = X.shape[1]
    XtX_inv = np.linalg.inv( np.dot(X.T , X) )
    X_XtX_inv = np.dot(X, XtX_inv)
    S = np.diag(np.ones(n)) - np.dot(X_XtX_inv,X.T)   # diag(X rows) - X * (XtX)^-1 * Xt   # this simplifies: I - 1 * (1t1)^-1 * 1t
    SKS = np.dot(S, ( K + np.diag(np.ones(n)) ) )  # add a small offset to the diagonals of K, to ensure positive definiteness
    SKS = np.dot(SKS, S)
    eig = np.linalg.eigh(SKS)  # get eigen pairs of: S * (K + I) * S, 'eigh', means symmetric Hermitian 
    
    # Need to reverse order of these, as these have the LARGEST eigen values first                    
    values = np.flipud(eig[0])   # identical to R
    values = values[0:(n-q)]-1  # grab the top eigen values, and subtract the same offset, (as the eigen values would be offset by the same amount, where as the eigenvectors are unchanged)
    
    vectors =  np.fliplr(eig[1])     # not identical to R, some of the signs are flipped
    vectors = vectors[:,0:(n-q)]

    eigenSummary = collections.namedtuple('values', 'vectors')
    eigenSummary.values = values
    eigenSummary.vectors =vectors
 
    
    # stop if there are no real eigenvalues, not needed in python, as eigh throws error for complex eigen results
    return (eigenSummary ) # only return n-q of the top eigen pairs, IE don't return as many eigen pairs as we have fixed effects?? is this the bit that 'regresses' out fixed effects???
    

# calculates log likelihod of ratio of prams, (without Z)
def deltaLogLikelihood(logdelta, lmbda, etas) :
    nq = etas.shape[0]
    delta =  np.exp(logdelta) # ratio between genetic and random Varance Component
    return( 0.5*(nq*(np.log(nq/(2*np.pi)) -1 -np.log(np.sum(etas*etas/(lmbda+delta)))) - np.sum(np.log(lmbda+delta))) )
    # the above is the same as this, for a single delta
    #  LL  <- 0.5* ((n-q)*(log((n-q)/(2*pi))-1-log(colSums(Etasq/Lambdas)))      -colSums(log(Lambdas))) # Log Likelihood



# calculates the 1st derivative of the log likelihood (score function), same as equation (9) in paper
def deltaLogLikelihoodDerivative(logdelta, lmbda, etas) :
    nq = etas.shape[0]
    delta = np.exp(logdelta)
    etasq = etas*etas
    ldelta = lmbda+delta
    return( 0.5*(nq*np.sum(etasq/(ldelta*ldelta))/np.sum(etasq/ldelta) - np.sum(1/ldelta)) )
    # the above is the same as this, for a single delta
    # dLL <- 0.5*delta*((n-q)*colSums(Etasq/(Lambdas*Lambdas))/colSums(Etasq/Lambdas)-colSums(1/Lambdas))  # 1st derivative of Log Likelihood






# returns ALL the derivative likelihoods, this is NOT the same as in the paper, as it has an extra 'delta' factored in? but that is a mistake, if I take it out then it still works with identical results
def logLikelihoodDerivative_all(logdelta, lmbda, etas) :
    dLL = np.zeros(len(logdelta))
    #nq = etas.shape[0]
    #etasq = etas*etas
    for i in range(0, len(logdelta)): # go through all deltas
        # delta = np.exp(logdelta[i])
        #ldelta = lmbda+delta
        # dLL[i] = 0.5*delta*(nq*np.sum(etasq/(ldelta*ldelta))  /np.sum(etasq/ldelta) - np.sum(1/ldelta)) 
        dLL[i] = deltaLogLikelihoodDerivative(logdelta[i],lmbda, etas) # * delta # calculate the dLL, and multipyl it by delta...
     
    return(dLL)
    
def uniroot_manual(f, lower, upper, lmbda, etas) :
    maxIt = 1000
    DBL_EPSILON =  2.2204460492503131e-16 # th
    c = lower # init c being a
    for i in range(0,maxIt):
        c_last = c  # save last iteration's result(used for comparison)
        c = (lower+upper)/2  # get next iteration's value at which function is evaluated at
        
        f_c = f(c,lmbda,etas) # evaluate function at new point, (save this as we reuse it 2x)
        
        if   f(lower,lmbda,etas) * f_c < 0 : # if their product's sign is negative,  this can only happen if their signs are opposite (2+ or 2- will give + result)
            upper = c   
        else : # then right boundary is moved closer
            lower = c   # otherwise left boundary is moved closer
        
    
        # check to see if we converged
        if np.abs( c_last - c) < DBL_EPSILON or np.abs(f_c) < DBL_EPSILON : # if either the change between iterations is too small, OR th 
            #print(c_last) 
            #print(c)
            #print("breaking loop")
            break 
        
    
    #print(i) # how many iterations it took
      
    return(c)


##################################################################################################
# significance testing & confidence intervals for REML
##################################################################################################

def OLS_LL(y, X) :
    n = len(y)
    p = X.shape[1]
    DF = n - p
    Xt = X.T
    XtX_inv = np.linalg.inv( np.dot(Xt , X) )
    Xty = np.dot(Xt,y)
    betaHat = np.dot(XtX_inv, Xty)
    yHat = np.dot(X,betaHat)
    e = y - yHat
    SSE = np.dot(e.T, e)
    sigmaSQHat_MSE = SSE / DF
    # ResidualStdError = np.sqrt(sigmaSQHat_MSE)
    
    LL = 0.5 * ( -n* ( np.log(2*np.pi) + np.log(sigmaSQHat_MSE) ) - 1/sigmaSQHat_MSE * SSE )
    return(LL)


# Perform Likelihood ratio Test, comparing the full model ()
def significanceTest_REML(REML_LL, y, X) :
  OLSLL = OLS_LL(y, X) # calculate the loglikelihood of a fixed effects only model
  D_Chisq = -2 * (OLSLL - REML_LL) # twice the  difference in
  DF_diff = 1 # we assume that there is only 1 degree of freedom difference, as we are testing the effect of the Vg term
  
  # REML hypothesis testing (Vischer):    the p value is from a mixture of 2 ChiSQ distributions: one with 0 and the other with 1 Degrees of freedom
  #p_value = chisqprob(D_Chisq, 0) * 0.5 + chisqprob(D_Chisq, DF_diff) * 0.5
  p_value = chisqprob(D_Chisq, DF_diff) * 0.5  # halving the 'higher' Likelihood's P value is the same as adding up the 2 mixture halves
  return(p_value)

def h2_SE_approx2(y, eigenValues_of_K) :
    N = len(y)
    E_a = N * np.var(eigenValues_of_K)
    SE = np.sqrt ( 2 /   E_a )
    return(SE)

# calculating the SE of h2, as suggested by (Visscher et al, 2015)
#def h2_SE_approx(y, eigenValues_of_K) :
#    N = len(y)
#    E_a = N * np.var(eigenValues_of_K)
#    SE = np.sqrt ( 2 / (  E_a * (1 - 1/N) ) )
#    return(SE)
    

#def h2_SE_precise(y, h2, eigenValues_of_K) :
#    N = len(y)
#    a = np.sum( (eigenValues_of_K-1)**2 / (1 + h2 *(eigenValues_of_K -1) )**2 )
#    b = np.sum( (eigenValues_of_K-1)   / (1 + h2 *(eigenValues_of_K -1) )   )
#    SE = np.sqrt ( 2/ (a - b^2/N)  )
#    return(SE)
    
##################################################################################################
# Computing BLUPs for training Data, and back calculating marker effects / predicting phenotypes for validation data
##################################################################################################

# Kernel ridge via 'regular' Ridge formula:   Beta =(XtX +DI)^-1 XtY (THIS IS THE MAIN ONE I USE)
def computeBLUPs_RidgeBLUP(y,X, delta) :  
  
    Xt = X.T 
    n = Xt.shape[0]
    I = np.identity(n)
    XtY = np.dot(Xt,y)
    XtXD_inv = np.linalg.inv( np.dot(Xt , X) +delta*I )
      
    Beta_BLUP = np.dot(XtXD_inv , XtY)
    blup = np.dot(X , Beta_BLUP)
      
    result = collections.namedtuple('BLUP', 'BETA')
    result.BLUP = blup
    result.BETA =Beta_BLUP
    return(result)

# usage:
#BLUP_hat = np.dot(X , Bresults.BETA)
#np.corrcoef(BLUP_hat, y_validation)