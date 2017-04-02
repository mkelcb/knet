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



emma.REMLE_GWAS <- function(y, K,  X = NULL, ngrids=100, llim=-10, ulim=10, esp=1e-10, eig.L = NULL, eig.R = NULL) {
  
  # I) Data QC: determine if data passed in is valid to run algo
  if( is.null(X)) { # This algo will NOT run if there are no Fixed effects, so if there were none passed in, then we add an intercept ( a column of 1s)
    # print("no fixed effects specified, adding an intercept")
    X = matrix(1, length(y), ncol= 1)
  }
  
  n <- length(y)  # number of individuals in study
  t <- nrow(K)  # the row/col of the kinship matrix (should be same as n)
  q <- ncol(X)  # the number of fixed effect predictors
  
  stopifnot(ncol(K) == t)
  stopifnot(nrow(X) == n)
  
  
  if ( det(crossprod(X,X)) == 0 ) { # if the determinant of XtX is 0, then matrix is singular
    warning("X is singular")
    return (list(REML=0,delta=0,ve=0,vg=0))  # REML: the loglik, delta; DF, 
  }
  
  # II) main REML algo: NR with Grid search, performed on the likelihood expressed in terms of the eigen summary of the data to find Delta (Ve/Vg)
  # if no strain incidence matrix was passed in (IE it is a GWAS)
    if ( is.null(eig.R) ) { # if cached Eigen values were not passed in
      eig.R <- emma.eigen.R.wo.Z(K,X) # compute eigen summary of Kinship/Fixed effects: returns list with 2 elements, [0]  n-1 eigen valus, [1]:  n-1 eigenvectors, each with length of n
    }
    etas <- crossprod(eig.R$vectors,y) # (n-1):1 column vector: matrix product of the t(EigenVctors) * Y, this is kindof like 'XtY', as we replaced the K and X with their eigen vectors
    
    logdelta <- (0:ngrids)/ngrids*(ulim-llim)+llim # create a Grid for for the possible values for the ratio of genetic/random variance: ie with lower/upper bound of -10 to 10, this will go from -10 to +10, with steps of 0.2 (IE:  -10.0  -9.8  -9.6  -9.4 ... 9.2   9.4   9.6   9.8  10.0)
    m <- length(logdelta) # how many grid search points we have (determines the length of the main loop)
    delta <- exp(logdelta)# bring back the delta onto non-log scale
    Lambdas <- matrix(eig.R$values,n-q,m) + matrix(delta,n-q,m,byrow=TRUE) # matrix of eigenValues + Deltas
    Etasq <- matrix(etas*etas,n-q,m) # square Etas (  IE (XtY)^2  )
    
    # Pre-compute the Derivative loglik for ALL possible deltas
    # LL  <- 0.5*      ((n-q)*(log((n-q)/(2*pi))-1-log(colSums(Etasq/Lambdas)))      -colSums(log(Lambdas))) # cache all Log Likelihood: these are NEVER used... duh
    #dLL <- 0.5*delta*((n-q)*colSums(Etasq/(Lambdas*Lambdas))/colSums(Etasq/Lambdas)-colSums(1/Lambdas))  # cache all 1st derivative of Log Likelihood
    dLL <- 0.5*       ((n-q)*colSums(Etasq/(Lambdas*Lambdas))/colSums(Etasq/Lambdas)-colSums(1/Lambdas))  # taken out the 'delta*' factor as it does not do anything, 
    # both of the above store a logLikelihood/derivative of loglikelihoods, for ALL grid points
    optlogdelta <- vector(length=0)  # initialise results for the 'optimal Delta' values
    optLL <- vector(length=0)  # initialise an array for the likelihood for the above
    
    # Handle edge cases, where the first/last likelihoods are too small
    if ( dLL[1] < esp ) { # if the 1st element of the derivative of the log likelihood is less than convergence threshold
      optlogdelta <- append(optlogdelta, llim) # add Lower bound of log ratio of two variance components
      optLL <- append(optLL, emma.delta.REML.LL.wo.Z(llim,eig.R$values,etas))
    }
    if ( dLL[m-1] > 0-esp ) { # if the last element of the derivative of the log likelihood is greater than neg convergence threshold
      optlogdelta <- append(optlogdelta, ulim) # add Upper bound of log ratio of two variance components
      optLL <- append(optLL, emma.delta.REML.LL.wo.Z(ulim,eig.R$values,etas))
    }  # normally neither of these should be true, this is just handling the edge cases, if there wouldn't be any valid solution within the upper-lower bounds: IE this will mean that the delta is then either -10 or +10, corresponding to 100% Vg, and 0% Vg
   
    
    for( i in 1:(m-1) ) # search Entire Grid (m-1 steps only, as we always search between current and current+1)
    {   #   also check if the diff between them is greater than threshold
      if ( ( dLL[i]*dLL[i+1] < 0-esp*esp ) && ( dLL[i] > 0 ) && ( dLL[i+1] < 0 ) ) # only attempt to find roots where sign flips between the 2 points (IE the function crosses the axis), : http://stackoverflow.com/questions/38961221/uniroot-solution-in-r
      { 
        # Find Root within range of current grid points (this is the NR step)
        r <- uniroot_manual(emma.delta.REML.dLL.wo.Z, lower=logdelta[i], upper=logdelta[i+1], lambda=eig.R$values, etas=etas) # this is not a true NR, this simply 'zooms in' at the root via a simple bisection method
        optlogdelta <- append(optlogdelta, r) # save the root (IE which is the maxima as the function is a derivative, as derivative is 0 at stationary point, so their signs must differ at before/after) 
        optLL <- append(optLL, emma.delta.REML.LL.wo.Z(r,eig.R$values, etas)) # calculate the precise loglik for the solution, will be used to find the best if there are more than 1
      }
    }  # this will normally only find a single solution

  
  # III) producing results for genetic/random variance componenets
  maxdelta <- exp(optlogdelta[which.max(optLL)]) # find the delta (genetic/random variance ratio), which had the max likelihood
  maxLL <- max(optLL)  # save the likelihood of the above as well

  maxva <- sum(etas*etas/(eig.R$values+maxdelta))/(n-q)  # get the Genetic variance component sum eta^2 / eigenvalues + maxdelta over the DF: this is equivalent to R/DF = SSG/DF (IE the mean sqared error after Vg)
  maxve <- maxva*maxdelta # calculate the radom variance (this is just a simple formula triangle, IE Ve = Vg * Ve/Vg, as Delta = Ve/Vg)
#
  blups = computeBLUPs(y, K, X, U = eig.R$vectors, lamda = eig.R$values, maxdelta)
  
 # blups2 = computeBLUPs_kerelRidge(y, K, X, U = eig.R$vectors, lamda = eig.R$values, maxdelta)
 # blups3 = computeBLUPs_kerelRidge2(y, K, X, U = eig.R$vectors, lamda = eig.R$values, maxdelta)
  
  
  #return (list(REML=maxLL,delta=maxdelta,ve=maxve,vg=maxva,BLUPS=blups,BLUPS2=blups2, BLUPS3=blups3)) # return results
  return (list(REML=maxLL,delta=maxdelta,ve=maxve,vg=maxva)) # return results
  
  }



#?eigen
#KI_2 = diag(3)
#KI_3 = diag(1,3)
#SKS = S%*%(K+diag(1,n))%*%S

#values= matrix(eig$values)
#vectors=eig$vectors
#vectors2=eig$vectors[,1:(n-q)]
#values2= matrix(eig$values[1:(n-q)]-1)

# Eigen decomposition summary of the kinship matrix + Fixed effects ( if X is 0, then this reduces to eigen(K,symmetric=TRUE) )
emma.eigen.R.wo.Z <- function(K, X) {
  n <- nrow(X)
  q <- ncol(X)
  S <- diag(n)-X%*%solve(crossprod(X,X))%*%t(X)   # diag(X rows) - X * (XtX)^-1 * Xt   # this simplifies: I - 1 * (1t1)^-1 * 1t
  eig <- eigen(S%*%(K + diag(1,n) )%*%S,symmetric=TRUE)  # get eigen pairs of: S * (K + I) * S   # add a small offset to the diagonals of K, to ensure positive definiteness
  stopifnot(!is.complex(eig$values)) # stop if there are no real eigenvalues
  return(list(values=eig$values[1:(n-q)]-1,vectors=eig$vectors[,1:(n-q)])) # only return n-q of the top eigen pairs, IE don't return as many eigen pairs as we have fixed effects?? is this the bit that 'regresses' out fixed effects???
                                       # subtract the same offset, (as the eigen values would be offset by the same amount, where as the eigenvectors are unchanged)
}


# calculates log likelihod of ratio of prams, (without Z)
emma.delta.REML.LL.wo.Z <- function(logdelta, lambda, etas) {
  nq <- length(etas)
  delta <-  exp(logdelta) # ratio between genetic and random Varance Component
  return( 0.5*(nq*(log(nq/(2*pi))-1-log(sum(etas*etas/(lambda+delta))))-sum(log(lambda+delta))) )
  # the above is the same as this, for a single delta
#  LL  <- 0.5* ((n-q)*(log((n-q)/(2*pi))-1-log(colSums(Etasq/Lambdas)))      -colSums(log(Lambdas))) # Log Likelihood
}


# calculates the 1st derivative of the log likelihood (score function)
emma.delta.REML.dLL.wo.Z <- function(logdelta, lambda, etas) {
  nq <- length(etas)
  delta <- exp(logdelta)
  etasq <- etas*etas
  ldelta <- lambda+delta
  return( 0.5*(nq*sum(etasq/(ldelta*ldelta))/sum(etasq/ldelta)-sum(1/ldelta)) )
  # the above is NOT the same as this, for a single delta, but that is actually probably a mistake in the paper
  # dLL <- 0.5*delta*((n-q)*colSums(Etasq/(Lambdas*Lambdas))/colSums(Etasq/Lambdas)-colSums(1/Lambdas))  # 1st derivative of Log Likelihood
}




uniroot_manual = function(f, lower, upper, lambda, etas) {
  maxIt = 1000
  DBL_EPSILON =  2.2204460492503131e-16 # th
  c = lower # init c being a
  for (i in 1:maxIt) {
    c_last = c  # save last iteration's result(used for comparison)
    c = (lower+upper)/2  # get next iteration's value at which function is evaluated at
    
    f_c = f(c,lambda,etas) # evaluate function at new point, (save this as we reuse it 2x)
    
    if  ( f(lower,lambda,etas) * f_c < 0 ) { # if their product's sign is negative,  this can only happen if their signs are opposite (2+ or 2- will give + result)
      upper = c  } else  { # then right boundary is moved closer
        lower = c   # otherwise left boundary is moved closer
      } 
    
    # check to see if we converged
    if( abs( c_last - c) < DBL_EPSILON || abs(f_c) < DBL_EPSILON ) # if either the change between iterations is too small, OR th
    { 
      #print(c_last) 
      #print(c)
      break 
    }
  }
  #print(i) # how many iterations it took
  
  return(c)
}

##################################################################################################
# significance testing & confidence intervals for REML
##################################################################################################

significanceTest_REML = function(REML_LL, y, X) {
  OLSLL = OLS_LL(y, X) # calculate the loglikelihood of a fixed effects only model
  D_Chisq = -2 * (OLSLL - REML_LL)
  DF_diff = 1 # we assume that there is only 1 degree of freedom difference, as we are testing the effect of the Vg term
  
  # REML hypothesis testing (Vischer):    the p value is from a mixture of 2 ChiSQ distributions: one with 0 and the other with 1 Degrees of freedom
  #p_value = 0.5 * pchisq(D_Chisq, 0, lower.tail = FALSE) + 0.5 * pchisq(D_Chisq, DF_diff, lower.tail = FALSE)
  p_value = pchisq(D_Chisq, DF_diff, lower.tail = FALSE) * 0.5  # halving the 'higher' Likelihood's P value is the same as adding up the 2 mixture halves
  return(p_value) #  IE: if p is significant, then REML is better, IE there IS h2

}



# http://stats.stackexchange.com/questions/155474/r-why-does-lrtest-not-match-anovatest-lrt

OLS_LL = function(y, X) {
  n = length(y)
  p = ncol(X)
  DF = n - p
  Xt = t(X)
  betaHat = solve(Xt%*%X) %*% Xt %*%y
  yHat = X%*%betaHat
  e = y - yHat
  SSE = t(e)%*%e 
  sigmaSQHat_MSE = SSE / DF
  # ResidualStdError = sqrt(sigmaSQHat_MSE)
  
  # OLS likelihood
  #LL = -n/2 * log(2 * pi) - n/2 * log(sigmaSQHat_MSE) - 1/(2*sigmaSQHat_MSE) * SSE
  #LL2 = 0.5 * ( -n*log(2 * pi) -n*log(sigmaSQHat_MSE) - 1/sigmaSQHat_MSE * SSE )
  LL = 0.5 * ( -n* ( log(2*pi) + log(sigmaSQHat_MSE) ) - 1/sigmaSQHat_MSE * SSE )
  return(LL)
}

# http://www.mayin.org/ajayshah/KB/R/documents/mle/ols-lf.pdf

#   0.5*  (nq*(log(nq/(2*pi))-1-log(sum(etas*etas/(lambda+delta))))-sum(log(lambda+delta)))



# calculating the SE of h2, as suggested by (Visscher et al, 2015)
#h2 = maxva / (maxva + maxve)
#SE_approx = h2_SE_approx(y,eig.R$value)
#SE_approx2 = h2_SE_approx2(y,eig.R$value)
#SE_precise = h2_SE_precise(y, h2,eig.R$value)

#h2_UB_aprx =  h2  + SE_approx * 1.96
#h2_LB_aprx =  h2  - SE_approx * 1.96

#h2_UB_aprx2 =  h2  + SE_approx2 * 1.96
#h2_LB_aprx2 =  h2  - SE_approx2 * 1.96

#h2_UB_prec =  h2  + SE_precise * 1.96
#h2_LB_prec =  h2  - SE_precise * 1.96


#print( paste("h2 is", round(h2,3) ) )
#print( paste("h2 approx CI is", round(h2_LB_aprx,3), "-", round(h2_UB_aprx,3) ) )
#print( paste("h2 approx2 CI is", round(h2_LB_aprx2,3), "-", round(h2_UB_aprx2,3) ) )
#print( paste("h2 precise CI is", round(h2_LB_prec,3), "-", round(h2_UB_prec,3) ) )

h2_SE_approx2 = function(y, eigenValues_of_K) { # get the SE for h2 via the approximation method further simplified
  N = length(y)
  E_a = N * var(eigenValues_of_K)
  SE = sqrt ( 2 /   E_a )
  return(SE)
}

#h2_SE_approx = function(y, eigenValues_of_K) { # get the SE for h2 via the approximation method
#  N = length(y)
#  E_a = N * var(eigenValues_of_K)
#  SE = sqrt ( 2 / (  E_a * (1 - 1/N) ) )
#  return(SE)
#}

#h2_SE_precise = function(y, h2, eigenValues_of_K) { # get the SE for h2 via the precise formulas
#  N = length(y)
#  a = sum( (eigenValues_of_K-1)^2 / (1 + h2 *(eigenValues_of_K -1) )^2 )
#  b = sum( (eigenValues_of_K-1)   / (1 + h2 *(eigenValues_of_K -1) )   )
#  SE = sqrt ( 2/ (a - b^2/N)  )
#  return(SE)
#}

##################################################################################################
# Computing BLUPs for training Data, and back calculating marker effects / predicting phenotypes for validation data
##################################################################################################



# Kernel ridge via 'traditional' formula: cannot invert K, so we have to add some to the diagonals, and as a result it performs poorly
# (and if K is invertible, then: g = K^-1(K^T + Delta*I)^-1 Y^T*K )
computeBLUPs_kerelRidge = function(y, K, X, U, lamda, delta) { # BLUp via kernel ridge
  # Recover the H inverse matrix from the eigen decpomposition
  Hinv <- U %*% ( t(U)/(lamda + delta) ) # reconstruct the Hinv, from U * (Ut / (Eigenvalues + Delta) )
  
  # Calculate Fixed effects
  W <- crossprod(X,Hinv%*%X)  # W is the (X H^-1 Xt)^1 (IE the denominator for the weighted LS)
  beta <- array(solve(W,crossprod(X,Hinv%*%y)))  # apply the B = (X H^-1 Xt)^1 XtHY, but as we have Hinv, we need to invert it again...
  

  
  y_residual = (y - X%*%beta)
  
  n = ncol(K)
  I = diag(n)
   g = solve(K+ I*0.001) %*% solve (t(K) + I*delta)   %*% K %*% y_residual
  #g = solve(K) %*% solve (t(K) + I*delta)   %*% K %*% y_residual
  #         nxn               nxn                 nx1     nxn            
  
  return(g)
}

# Kernel ridge via simplified formula (identical to above): still cannot invert K, so we have to add some to the diagonals, this actually performs OK
computeBLUPs_kerelRidge2 = function(y, K, X, U, lamda, delta) {
  # Recover the H inverse matrix from the eigen decpomposition
  Hinv <- U %*% ( t(U)/(lamda + delta) ) # reconstruct the Hinv, from U * (Ut / (Eigenvalues + Delta) )
  
  # Calculate Fixed effects
  W <- crossprod(X,Hinv%*%X)  # W is the (X H^-1 Xt)^1 (IE the denominator for the weighted LS)
  beta <- array(solve(W,crossprod(X,Hinv%*%y)))  # apply the B = (X H^-1 Xt)^1 XtHY, but as we have Hinv, we need to invert it again...
  
  
  
  y_residual = (y - X%*%beta)
  
  n = ncol(K)
  I = diag(n)

  g = solve (I + solve(K+ I*0.001)*delta) %*% y_residual
  
  # g = K^-1(K^T + Delta*I)^-1 Y^T*K )
  # g =     (I+K^T+Delta*I)^-1 Y)
  
  # vs rrblup:
  # (K *(K + delta)^1 )y
  
  # g = solve(K+ I*delta) %*% y_residual  # r = 0.3592015
  #         nxn               nxn                 nx1     nxn            
  
  return(g)
}

# alternative Kernel Ridge formula: g = K (K +ID)^-1 y: this always works too
computeBLUPs_kerelRidge3 = function(y, K, delta) {
  n = ncol(K)
  I = diag(n)
  
  g = K %*% solve(K+ I*delta) %*% y
       
  
  return(g)
}

# via Henderson's Equation: this always works, but relies on the REML framework
computeBLUPs = function(y, K, X, U, lamda, delta) {
  # Recover the H inverse matrix from the eigen decpomposition
  Hinv <- U %*% ( t(U)/(lamda + delta) ) # reconstruct the Hinv, from U * (Ut / (Eigenvalues + Delta) )
  
  # Calculate Fixed effects
  W <- crossprod(X,Hinv%*%X)  # W is the (X H^-1 Xt)^1 (IE the denominator for the weighted LS)
  beta <- array(solve(W,crossprod(X,Hinv%*%y)))  # apply the B = (X H^-1 Xt)^1 XtHY, but as we have Hinv, we need to invert it again...
  
  K_Hinv <- K %*% Hinv
  g <- array(K_Hinv %*% (y - X%*%beta)) # the formula for blups = (K *H^1 )  g = (K *H^1)y = (K *(K + delta)^1 )y
  
  return(g)
}


# back-calculate Beta: get individual SNP effects from the breeding values (must be used by the Kernel/Henderson formulas)
####################################################

backCalculate_Beta_BLUP = function(g, X) {  # when there are more predictors than individuals (GWAS case)
  Xt = t(X)
  Beta_BLUP = Xt %*%  solve(X%*%Xt ) %*%g
  #           9x2        2x9 * 9x2= 2x1   * 1x1 =  9x2 * 2x2 * 2x1 = 9x2 * 2x1 = 9x1
  return(Beta_BLUP)
}

backCalculate_Beta_BLUP_moren = function(g, X) { # when there are more individuals than predictors: this is the standad OLS formula (non-GWAS case)
  Xt = t(X)
  #Beta_BLUP = Xt %*% solve(X%*%Xt) %*%g
  #           2x5        5x2 * 2x5   * 5x1 =   2x5 * 5x5 * 5x1 = 2x5 * 5x1= 2x1
  
  Beta_BLUP = solve(Xt%*%X)  %*%Xt %*% g
  return(Beta_BLUP)
}

####################################################

# Kernel ridge via 'regular' Ridge formula:   Beta =(XtX +DI)^-1 XtY (THIS IS THE MAIN ONE I USE)
computeBLUPs_RidgeBLUP = function(y,X, delta) { 
  
  Xt = t(X)  # 100 x 500
  n = nrow(Xt)
  I = diag(n)
  XtY = Xt%*%y
  
  Beta_BLUP = solve(Xt%*%X +delta *I) %*%XtY
  blup = X%*%Beta_BLUP 
  
  return(list(BLUP=blup,BETA=Beta_BLUP))
}
# usage:

# Bresults =computeBLUPs_RidgeBLUP(y, Standardised_SNPs, results$delta) # regular Ridge BLUP
# compare against training set
# cor(Bresults$BLUP, y)

# compare against validation set
# BLUP_hat = X %*% Bresults$BETA
# cor(BLUP_hat, y_validation)



# Ridge via the Xt(XXt +D)^-1 y
#computeBLUPs_RidgeBLUP2 = function(y,X, delta) { ## this produces a mathematically equivalent results to the above
#  
#  Xt = t(X)  # 100 x 500
#  n = length(y)
#  I = diag(n)
#  
#  Beta_BLUP =  Xt %*% solve(X%*%Xt +delta *I) %*% y
#  BLUP = X%*%Beta_BLUP # 500 x100 x 100x1 = 500x1
#  
#  return(BLUP)
#}



# calculates the Standard Errors for the BLUPs, this is largely useless
# usage:  calc_BLUPSE(U = eig.R$vectors, eig.R$values, maxdelta, maxva, K, X)
#calc_BLUPSE <- function(U, phi, lambda.opt, Vu.opt, K, X) {
#  # Recover the H inverse matrix from the eigen decpomposition
#  Hinv <- U %*% ( t(U)/(phi+lambda.opt) ) # reconstruct the Hinv, from EigenVectors * (EigenVectors^T / (Eigenvalues + Delta) )
#  KZt = K # as Z is identity this is the same
#  KZt.Hinv <- KZt %*% Hinv
#  
#  W <- crossprod(X,Hinv%*%X)  # W is the (X H^-1 Xt)^1 (IE the denominator for the weighted LS)
#  Winv <- solve(W)
#  WW <- tcrossprod(KZt.Hinv,KZt)
#  WWW <- KZt.Hinv%*%X
#  
#  u.SE <- array(sqrt( Vu.opt * (diag(K) - diag(WW) + diag(tcrossprod(WWW%*%Winv,WWW)))) )
#  
#  print("standard error of the BLUPs")
#  print(u.SE)
#}
