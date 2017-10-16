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
def recodeSNPs_to_Dominance(X) :
    # duplicate matrix 
    X_new = np.zeros((X.shape[0], X.shape[1])) # in python we have to request deep copies
    
    # go through all genotype calls, and transform them into dominance terms
    p = X.shape[1]
    n = X.shape[0]

    for i in range(n):
        for j in range(p):
            # in PLINK we code: AA = 0 and aa = 2
            if(X[i,j] != 1) :
                X_new[i,j] = -1 #in dominance terms: AA and aa = -1
            else :
                X_new[i,j] = 0 # and Aa=0
    
    return (  X_new ) 
    

def calc_Kinship_Dominance(X):
    X_Dom =recodeSNPs_to_Dominance(X) # recode into dominance contrasts
    Standardised_SNPs = geno_qc.standardise_Genotypes(X_Dom)  # standardise into Z-scores
    K = calc_Kinship( Standardised_SNPs  )  # calculate basic 'linear' kinship  
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
    