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


# BOTH y and X must be numpy 2d arrays
def multiReg(y, X) :
    # add a column f 1s to the intercept

    intercept =  np.ones( (X.shape[0], 1) ) # make sure that this is a 2D array, so that we can refer to shape[1]
    X = np.column_stack( (intercept, X) )
      
    Xt =X.T
    XtX_inv = np.linalg.inv(Xt.dot(X) )
    XtY = Xt.dot(y)
    Beta = XtX_inv.dot(XtY)
     
    yhat = X.dot(Beta)
    Residuals = (y - yhat)**2
    DF = X.shape[0] - X.shape[1]
      
    MSE = np.sum(Residuals) / DF
    Beta_Coefs = MSE * XtX_inv
    Beta_SE = np.sqrt( np.diag(Beta_Coefs) )
    t_values = Beta / Beta_SE
        
    P_values = stats.t.sf(np.abs(t_values), DF)*2
    
    return (  {"beta":Beta, "sigma":MSE, "se":Beta_SE, "df": DF, "p_values": P_values, "sse":np.sum(Residuals), "res":Residuals } )


# k is the number of parameters estimated ( and we add +1, for the noise)
def getBIC (n, SSE, k) :#  SSE :  sum of squared residuals
   return( n + n  *np.log(2*np.pi) + n * np.log(SSE / n) + np.log(n) * (k+1)  ) 

  # where do we penalise for the DFs ??




#y = y[0] 
#X = X_QC_


#result = multiReg(y,  np.array( X[:,0] )   )

# performs a series of univariate regressions
def uniReg(y, X) :

    results = list()
    betas = list()
    p_values = list()
    for i in range(X.shape[1]) :
        results.append( multiReg(y,  np.array( X[:,i] )   ) )
        betas.append(results[i]["beta"][1])
        p_values.append(results[i]["p_values"][1])
    
    return (  {"beta":betas, "p_values":p_values, "results": results } )

