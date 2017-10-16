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

from ....application.utils import geno_qc




def calc_Kinship(X) :
    p = X.shape[1]
    return (  ( np.dot(X,X.T) ) / p )  # 1 x 3 * 3x1 = 1x1
    


def calc_Kinship_Gaussian2(X, theta = 0.5) : # basically exponentiates a pairwise distance matrix
    p = X.shape[1]
    eucl_dist = eucledian_distance_matrix(X)
    D = eucl_dist/2/np.sqrt(p) # scale it by number of SNPs (and 2)
    K = np.exp(-(D/theta)**2) # gaussian formula, scale the matrix, square it, then exponentiate its negative
    return (  K ) 







# recodes Genotype matrix into dominance contrasts ( doesn't alter the original)
# X = raw_SNPs
def recodeSNPs_to_Dominance(X) :
    # duplicate matrix 
    X_new = X.copy() # in python we have to request deep copies
    p = X.shape[1]

    # go through all genotype calls, and transform them into dominance terms
    # in PLINK we code: AA = 0, Aa =1 and aa = 2
    # convert (AA,Aa,aa)->(1,1,0);
    for i in range(p): # go through each col
      X_new[:,i][  X_new[:,i] ==2] = 1  #in dominance terms: aa and Aa = 1 (IE both hets and hom alts are 1, IE both 1 and 2 are 1)
                                        # and AA=0 (IE hom refs are 0), so we just leave them alone
    return (  X_new ) 



#def recodeSNPs_to_Dominance_slow2(X) : # 20x slower than the above
    # duplicate matrix 
#    X_new = np.zeros((X.shape[0], X.shape[1])) # in python we have to request deep copies

    
    # go through all genotype calls, and transform them into dominance terms
#    p = X.shape[1]
#    n = X.shape[0]

#    for i in range(n):
#        for j in range(p):
#            # in PLINK we code: AA = 0 and aa = 2
#            if(X[i,j] == 1 or X[i,j] == 2) :
#                X_new[i,j] = 1 #in dominance terms: AA and aa = -1
#            else :
#                X_new[i,j] = 0 # and Aa=0

    
 #   return (  X_new ) 




# merges 2 kinships
# def mergeKinhips(K1,K2, K1_SNPs, K2_SNPs)  :
#    grandTotal = K1_SNPs + K2_SNPs
#    weight1 = K1_SNPs / grandTotal
#    weight2 = K2_SNPs / grandTotal
#    return(K1 * weight1 +   K2 * weight2  ) # blends the 2 kinships together at the appropriat weight
    

def calc_Kinship_Dominance(X, weights = None): # weights is an optional diagonal matrix that contains LDAK/MAFS weights
    X_Dom =recodeSNPs_to_Dominance(X) # recode into dominance contrasts
    
    if weights is not None :
        X_Dom = X_Dom.dot(weights)
                                  
    X_Dom = geno_qc.standardise_Genotypes(X_Dom)  # standardise into Z-scores
    K = calc_Kinship( X_Dom  )  # calculate basic 'linear' kinship  
    return (K) 



# Computes the eucledian distance of each row against all other rows, and puts results into a symmetric matrix
def eucledian_distance_matrix(X) :
    distanceMatrix = np.zeros((X.shape[0], X.shape[0]))
    
    for i in range(0, (X.shape[0]-1) ):  # from individual 1 to one before last
        SNPs_ind_i = X[i,:] ; ## get individual i's SNPS
        
        for j in range(i+1,X.shape[0] ): # test i against all the rest of the individuals
            SNPs_ind_j = X[j,:] ;   
            value = np.sqrt(np.sum((  SNPs_ind_i.T - SNPs_ind_j)**2))   
            distanceMatrix[i ,j] = distanceMatrix[j,i]  =  value

    return(distanceMatrix)
    