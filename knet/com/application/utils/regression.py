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
from scipy import stats

# local test
#from numpy import genfromtxt
#y = genfromtxt('y.txt', delimiter='\t')
# X = genfromtxt('X.txt', delimiter='\t')

#X = X_design
# BOTH y and X must be numpy 2d arrays


def genericReg(y, X, bonfCorrect = True) :
    if X.shape[0] <= X.shape[1] +1 : return (  uniReg(y, X, bonfCorrect = bonfCorrect) ) # if n < p, +1 for the intercept
    else : return (  multiReg(y, X, bonfCorrect = bonfCorrect) )


#X_orig = X
#y= y_train
#X = X_train
#X = X_orig
#X = np.ones((3,3))
def multiReg(y, X, bonfCorrect = True, addIntercept = True) :
    #print("multiReg, X is: " , X.dtype)
    # add a column f 1s to the intercept
    if addIntercept :
        origDtype = X.dtype
        intercept =  np.ones( (X.shape[0], 1), dtype = origDtype ) # make sure that this is a 2D array, so that we can refer to shape[1]
        X = np.column_stack( (intercept, X) )
        #X = X.astype(origDtype)
     ###############
     # orig
    #print("X is: " , X.dtype)
    Xt =X.T
    #print("Xt is: " , Xt.dtype)
    XtX_inv = np.linalg.inv(Xt.dot(X) )
    #print("XtX_inv is: " , XtX_inv.dtype)
    XtY = Xt.dot(y)
    ##############
    
#   temp = X.T.dot(X)
#    XtX_inv = np.linalg.inv(temp )
#    del temp
#    XtY = X.T.dot(y)
    
    # Cholesky decomposition inversion:
#    print("X is: " , X.dtype)
#    temp = X.T.dot(X)
#    print("temp is: " , temp.dtype)
#    c = np.linalg.inv(np.linalg.cholesky(temp))
#    print("c is: " , c.dtype)
#    del temp
#    XtX_inv = np.dot(c.T,c)
#    del c     
#    XtY = X.T.dot(y) 
    
    #######################
    
    
    Beta = XtX_inv.dot(XtY)

    yhat = X.dot(Beta)
    Residuals = (y - yhat)**2
    DF = X.shape[0] - X.shape[1]
      
    MSE = np.sum(Residuals) / DF
    Beta_Coefs = MSE * XtX_inv
    Beta_SE = np.sqrt( np.diag(Beta_Coefs) )
    Beta_SE = Beta_SE.reshape(len(Beta_SE),-1) # enforce that it is the same shape, ie if Beta was (3,1) and this is (3,) then we would get a (3,3) result which is not what we want, but a (3,1) one too... IE a t value associated with each Beta
    t_values = Beta / Beta_SE
        
    P_values = stats.t.sf(np.abs(t_values), DF)*2
    if bonfCorrect : P_values = P_values *len(P_values) # if we are to bonferroni correct it multiply each p value by the number of p values
        
    
    return (  {"beta":Beta, "sigma":MSE, "se":Beta_SE, "df": DF, "p_values": P_values, "sse":np.sum(Residuals), "res":Residuals } )


#X_valid = X_valid_interactions 
#model = model_interactions
#onlySignificantParams = True
#bonfCorrect = False

def predictYhat(X_valid, model, onlySignificantParams = True, bonfCorrect = False, alpha =0.05 ) :
    intercept =  np.ones( (X_valid.shape[0], 1) ) # make sure that this is a 2D array, so that we can refer to shape[1]
    X_valid = np.column_stack( (intercept, X_valid) ) 
    
    
    if onlySignificantParams : 
        p_values = model['p_values']
        if bonfCorrect : 
            #print("bonfcorrect!")
            significantIndices = np.where(p_values<=alpha)[0]
            #print("num significantIndices BEFORE bonf correction: " + str( len(significantIndices)))
            p_values = p_values *len(p_values) # if we are to bonferroni correct it multiply each p value by the number of p values
        
        # delete non-significant columns from X and betas
        significantIndices = np.where(p_values<= alpha)[0]
        #print("num significantIndices AFTER bonf correction: " + str( len(significantIndices)))
            
        Beta_sig = model['beta'][significantIndices]
        X_valid = X_valid[:,significantIndices]
    else : Beta_sig  = model['beta']
    
    return( X_valid @ Beta_sig )  # =  XB

#X = X_train
#y = y_train

# compare p values and betas 
# np.corrcoef( betas, p_values, rowvar=0)[1,0]**2  # 0.00016620417882577391
# np.corrcoef( np.abs(betas), p_values, rowvar=0)[1,0]**2  # 0.89331050397928102
# CONCLUSION: absolute values for Betas is similar to P-values


# performs a series of univariate regressions
def uniReg(y, X, bonfCorrect = True) :
    #print("uniReg")
    results = list()
    betas = list()
    p_values = list()
    Beta_SE = list()
    
    # for the first one we add one for the intercept
    intercept = np.empty((X.shape[0],0), dtype = X.dtype)
    results.append( multiReg(y,intercept, bonfCorrect = False) )
    betas.append(results[0]["beta"][0])
    p_values.append(results[0]["p_values"][0])  
    
    for i in range(X.shape[1]) :
        model = multiReg(y,  np.array( X[:,i]  ), bonfCorrect = False   )
        results.append( model )
        betas.append(model["beta"][1])
        Beta_SE.append(model["se"][1])
        p_values.append(model["p_values"][1])
    
    p_values = np.array(p_values)
    
    if bonfCorrect : p_values *= len(p_values) # if we are to bonferroni correct it multiply each p value by the number of p values
     

    betas = np.array(betas)
    return (  {"beta":betas, "p_values":p_values, "results": results, "se":Beta_SE } )


# k is the number of parameters estimated ( and we add +1, for the noise)
def getBIC (n, SSE, k) :#  SSE :  sum of squared residuals
   return( n + n  *np.log(2*np.pi) + n * np.log(SSE / n) + np.log(n) * (k+1)  ) 

  # where do we penalise for the DFs ??
