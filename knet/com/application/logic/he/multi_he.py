# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:46:12 2017

@author: Marton
"""
import numpy as np
import scipy

from ....application.utils import regression
from ....application.utils import geno_qc
from ....application.logic.reml import kinship
from ....io import knet_IO
import gc

# fastest way of obtaining all Combination pairs (5x faster than for loops, and 3x faster than itertools)
def getCombinationIndices(n) :
    counter = n -1
    startPos = 0
    endpos = counter
    num_Pairs = int( scipy.special.binom(n,2) )
    indices2 = np.zeros( (num_Pairs,2), dtype=np.int ) # generate blank array of the right size
    for i in range(n)  : #go through all individuals
        # paste 2 arrays at the right position, col 1: the same individual, against col 2: all the other individuals
        indices2[startPos:endpos, 0] = np.full(counter, i, dtype=int) # 1st col, same individual 3000 times
        indices2[startPos:endpos, 1] =  np.array(   range( (i+1), n),    ) # 2nd col, from 2nd individual up until last
        
        
        startPos = endpos # start again where we left off
        counter = counter-1 # we will get one less after each loop
        endpos = startPos + counter
       
    return (indices2)
    

#raw_SNPs = SNPs
#Standardised_SNPs = X
#epistasis = 2
# dominance = True
def computePairs(raw_SNPs, dominance = False, epistasis = -1, covariates = None, weights = None) :
    n = raw_SNPs.shape[0]
    num_Pairs = int( scipy.special.binom(n,2) )
    
    numCols = 1
    if dominance : numCols = numCols+1
    if epistasis > 1 : numCols = numCols+1
    if covariates is not None : numCols = numCols+1
    
     # produce master design matrix of the right dimensions
    designMatrix = np.zeros( (num_Pairs,numCols))
    currentCol = 0
    
    # get Additive pairs
    Standardised_SNPs = geno_qc.standardise_Genotypes(raw_SNPs)
    # apply Weights  ?? but what about reusing it for epistasis ???
    K_additive = kinship.calc_Kinship(Standardised_SNPs)
    del Standardised_SNPs ; gc.collect()
    grm_indices = np.triu_indices(n, k=1)   # k = 1 EXCLUDES the diagonal elements, must use triu instead of tril, as tril gets it by 'row order' and we want 'col order' as that is how the Ys are computed
    designMatrix[:,currentCol] = K_additive[grm_indices[0], grm_indices[1]] 
    currentCol = currentCol+1
   
   
    # get Epistasis pairs
    if epistasis >1  :
        print("computing epistatic kinship at level: ",epistasis )
        #K_epistasis = calc_Kinship_Gaussian2(Standardised_SNPs,0.95)
        #K_epistasis = K_additive * K_additive
        K_epistasis = K_additive**epistasis # this copies K_additive, doesn't overwrite it
        designMatrix[:,currentCol] = K_epistasis[grm_indices[0], grm_indices[1]]
        currentCol = currentCol+1
        del K_epistasis  ; gc.collect() 
        
    del K_additive ; gc.collect() # AFTEr K_additive has been (potentially) used ( but before we use memory for the dominance stuff)
    
    
    # get Dominance pairs
    if dominance  :
        print("computing dominance kinship")
        K_dominance = kinship.calc_Kinship_Dominance(raw_SNPs)
        gc.collect()
        designMatrix[:,currentCol] = K_dominance[grm_indices[0], grm_indices[1]]
        currentCol = currentCol+1
        del K_dominance  ; gc.collect() 
        
     

        
    
    if  covariates is not None  :
        Standardised_covs = geno_qc.standardise_Genotypes(covariates)
        covariate_covariance = kinship.calc_Kinship(Standardised_covs)
        designMatrix[:,currentCol] =covariate_covariance[grm_indices[0], grm_indices[1]]
        currentCol = currentCol+1
        
        
    del grm_indices ; gc.collect()
    

    return(designMatrix)

# Y =y
# same as the function below but each predictor after the first, is 'residualised' IE the previous predictors are taken out of them, so MHER only runs on their residuals wrt to the other predictors
# this is cuz the kinship matrices are not independent predictors, as they represent overlapping signals
# so we need to fit a model like this:
# Z_res = lm(Z~X)$residuals # IE  get the bits of Z that are unique to Z and not found in X
# lm(Y ~ X + Z_res) # 
# numCols = 2
def HE_Multi_residuals (args, Y) :
    print("MHER -residual is running", flush=True)
    # create the phenotype contrasts ( Z = (Yi-Yj)^2)
    n = Y.shape[0]
    ## calculate relatedness / IBS pairwise table for all individuals:
    indices = getCombinationIndices(n)
    Yi = np.take(Y, indices[:,0])
    Yj = np.take(Y, indices[:,1])
    
    all_Outcomes = (Yi - Yj)**2
        
    # create the appropriate sized  'all outcomes' matrix ( with the right epistasis level)             
    num_Pairs = int( scipy.special.binom(n,2) )
    
    numCols = 1 # 1 for additive
    if args.epi > 1 : numCols = numCols+args.epi-1 # for each level of epistasis we add another column, but only after 1
    if args.domkin : numCols = numCols+1
    dominanceCol = -1
    #if covariates is not None : numCols = numCols+1 

    # produce design matrix of the right dimensions
    designMatrix = np.zeros( (num_Pairs,numCols))
    currentCol = 0
   
    # get lower triangle indices (will be reused for each kinship)
    grm_indices = np.triu_indices(n, k=1)   # k = 1 EXCLUDES the diagonal elements, must use triu instead of tril, as tril gets it by 'row order' and we want 'col order' as that is how the Ys are computed
    
               
    # 1) load additive kinship
    print("loading additive kinship", flush=True)
    K_data = knet_IO.loadGCTA_GRM(args.addkin)
    K_additive = K_data["K"]  
    numSNPs = K_data["N"][0]
    print("loaded additive kinship for", str(K_additive.shape[0]) , "individuals", flush=True)
    del K_data; gc.collect() 
   
    
    # extract additive pairs
    designMatrix[:,currentCol] = K_additive[grm_indices[0], grm_indices[1]]  
    currentCol = currentCol+1
   
    
    # if epistasis was requested, generate epistatic kinships from the additive kinships
    if args.epi > 1 :
        # for each level of epistasis, generate one column
        for i in range(2, (args.epi +1) ) :
            print("generating epistatic kinship at level", i, flush=True)
            K_epistasis = K_additive**i # this copies K_additive, doesn't overwrite it
            
            # get the bits of Epistasis that are unique to Epistasis and not found in Additive
            epistaticPredictors = K_epistasis[grm_indices[0], grm_indices[1]]
            epistaticPredictors = regression.multiReg(epistaticPredictors, designMatrix[:,0:currentCol])["res"]
            
            designMatrix[:,currentCol] = epistaticPredictors
            currentCol = currentCol+1
            del K_epistasis  ; gc.collect() 
            
    
    # dispose additive kinship
    del K_additive ; gc.collect() # AFTEr K_additive has been (potentially) used ( but before we use memory for the dominance stuff)
    

    
    # load dominance if dom kinship is supplied
    if args.domkin :
        print("loading dominance kinship", flush=True)
        K_data = knet_IO.loadGCTA_GRM(args.domkin)
        K_dominance = K_data["K"]  
        del K_data; gc.collect() 
        
        # get the bits of dominance that are unique to dominance and not found in Additive+ Epistasis
        dominancePredictors = K_dominance[grm_indices[0], grm_indices[1]] # extract dominance pairs
        dominancePredictors = regression.multiReg(dominancePredictors, designMatrix[:,0:currentCol])["res"]         
        designMatrix[:,currentCol] = dominancePredictors
        
        dominanceCol = currentCol
        currentCol = currentCol+1
        del K_dominance  ; gc.collect() 
    
  
   
    # fit all models: start from smallest model, and loop until the ncol of allpairs
    allModels = list()
    for i in range(0,currentCol +1) :
        print("fitting model with " , i , " parameters", flush=True)
        H_E_regression = multiReg(all_Outcomes, designMatrix[:,0:i]) 
        
        # make note on if this model includess dominance or not
        colDominance = -1
        if i >= dominanceCol : colDominance = dominanceCol # this will store the col index of the dominance VC ( as we removed the intercept term these should match again...)

        # get BIC
        SSE = H_E_regression["sse"]
        BIC = getBIC(num_Pairs, SSE, (i+1) ) # i is the number of predictors +1 for the intercept
        
        # here, filter out all VCs that are non-significant ( p < 0.05) 
        VC_SE =  0.5 * H_E_regression["se"][1:len(H_E_regression["se"])]  # don't need the intercept
        VC = -0.5 * H_E_regression["beta"][1:len(H_E_regression["beta"])]    # don't need the intercept
        p_vals = H_E_regression["p_values"][1:len(H_E_regression["p_values"])]
          
        # compute the total genetic variance
        SumVC_sig = 0 # the sum of the significant variance components
        SumVC_SE_sig = 0 # the sum of the significant variance components' Standard Errors
        for i in range( len(p_vals) )  :
            if p_vals[i] <= 0.05 :  # only include significant Vas
                SumVC_sig = SumVC_sig +  VC[i] 
                SumVC_SE_sig = SumVC_SE_sig +  VC_SE[i] 
         
        V_all = np.var(Y)
        Ve =  V_all - SumVC_sig  # noise variance is the total Pheno variance minus ALL of the other variance components (the significant ones)
        h2_measured = SumVC_sig / V_all  # Broad-sense h2            
           
     
        quantile = SumVC_SE_sig * 1.96
        h2_CI95_ub = (SumVC_sig + quantile) /  V_all
        h2_CI95_lb = (SumVC_sig - quantile) /  V_all
        # Proper way to estimate CI95 for h2: 
        # h2_measured_up_orig = (Va + Va_SE * 1.96)  /V_all
        # not sure how this would generalise to multiple 
        # h2_measured_up_orig
   
        allModels.append(     {"h2":h2_measured,"h2_ub":h2_CI95_ub,"h2_lb":h2_CI95_lb, "vc":VC, "ve":Ve, "vc_se":VC_SE, "p":p_vals, "bic":BIC, "domcol": colDominance  }   )    
        del H_E_regression  ; gc.collect()
    
    # save results to text file
    knet_IO.writeVCResults(args.out, allModels)



def HE_Multi_external (args, Y) :
    print("MHER -original is running", flush=True)
    # create the phenotype contrasts ( Z = (Yi-Yj)^2)
    n = Y.shape[0]
    ## calculate relatedness / IBS pairwise table for all individuals:
    indices = getCombinationIndices(n)
    Yi = np.take(Y, indices[:,0])
    Yj = np.take(Y, indices[:,1])
    
    all_Outcomes = (Yi - Yj)**2
        
    # create the appropriate sized  'all outcomes' matrix ( with the right epistasis level)             
    num_Pairs = int( scipy.special.binom(n,2) )
    
    numCols = 1 # 1 for additive
    if args.epi > 1 : numCols = numCols+args.epi-1 # for each level of epistasis we add another column, but only after 1
    if args.domkin : numCols = numCols+1
    dominanceCol = -1
    #if covariates is not None : numCols = numCols+1 

    # produce design matrix of the right dimensions
    designMatrix = np.zeros( (num_Pairs,numCols))
    currentCol = 0
   
    # get lower triangle indices (will be reused for each kinship)
    grm_indices = np.triu_indices(n, k=1)   # k = 1 EXCLUDES the diagonal elements, must use triu instead of tril, as tril gets it by 'row order' and we want 'col order' as that is how the Ys are computed
    
               
    # 1) load additive kinship
    print("loading additive kinship", flush=True)
    K_data = knet_IO.loadGCTA_GRM(args.addkin)
    K_additive = K_data["K"]  
    numSNPs = K_data["N"][0]
    print("loaded additive kinship for", str(K_additive.shape[0]) , "individuals", flush=True)
    del K_data; gc.collect() 
   
    
    # extract additive pairs
    designMatrix[:,currentCol] = K_additive[grm_indices[0], grm_indices[1]]  
    currentCol = currentCol+1
   
    
    # if epistasis was requested, generate epistatic kinships from the additive kinships
    if args.epi > 1 :
        # for each level of epistasis, generate one column
        for i in range(2, (args.epi +1) ) :
            print("generating epistatic kinship at level", i, flush=True)
            K_epistasis = K_additive**i # this copies K_additive, doesn't overwrite it
            designMatrix[:,currentCol] = K_epistasis[grm_indices[0], grm_indices[1]]
            currentCol = currentCol+1
            del K_epistasis  ; gc.collect() 
            
    
    # dispose additive kinship
    del K_additive ; gc.collect() # AFTEr K_additive has been (potentially) used ( but before we use memory for the dominance stuff)
    

    
    # load dominance if dom kinship is supplied
    if args.domkin :
        print("loading dominance kinship", flush=True)
        K_data = knet_IO.loadGCTA_GRM(args.domkin)
        K_dominance = K_data["K"]  
        del K_data; gc.collect() 
        # extract dominance pairs
        designMatrix[:,currentCol] = K_dominance[grm_indices[0], grm_indices[1]]
        dominanceCol = currentCol
        currentCol = currentCol+1
        del K_dominance  ; gc.collect() 
    
  
   
    # fit all models: start from smallest model, and loop until the ncol of allpairs
    allModels = list()
    for i in range(0,currentCol +1) :
        print("fitting model with " , i , " parameters", flush=True)
        H_E_regression = regression.multiReg(all_Outcomes, designMatrix[:,0:i]) 
        
        # make note on if this model includess dominance or not
        colDominance = -1
        if i >= dominanceCol : colDominance = dominanceCol # this will store the col index of the dominance VC ( as we removed the intercept term these should match again...)

        # get BIC
        SSE = H_E_regression["sse"]
        BIC = regression.getBIC(num_Pairs, SSE, (i+1) ) # i is the number of predictors +1 for the intercept
        
        # here, filter out all VCs that are non-significant ( p < 0.05) 
        VC_SE =  0.5 * H_E_regression["se"][1:len(H_E_regression["se"])]  # don't need the intercept
        VC = -0.5 * H_E_regression["beta"][1:len(H_E_regression["beta"])]    # don't need the intercept
        p_vals = H_E_regression["p_values"][1:len(H_E_regression["p_values"])]
          
        # compute the total genetic variance
        SumVC_sig = 0 # the sum of the significant variance components
        SumVC_SE_sig = 0 # the sum of the significant variance components' Standard Errors
        for i in range( len(p_vals) )  :
            if p_vals[i] <= 0.05 :  # only include significant Vas
                SumVC_sig = SumVC_sig +  VC[i] 
                SumVC_SE_sig = SumVC_SE_sig +  VC_SE[i] 
         
        V_all = np.var(Y)
        Ve =  V_all - SumVC_sig  # noise variance is the total Pheno variance minus ALL of the other variance components (the significant ones)
        h2_measured = SumVC_sig / V_all  # Broad-sense h2            
           
     
        quantile = SumVC_SE_sig * 1.96
        h2_CI95_ub = (SumVC_sig + quantile) /  V_all
        h2_CI95_lb = (SumVC_sig - quantile) /  V_all
        # Proper way to estimate CI95 for h2: 
        # h2_measured_up_orig = (Va + Va_SE * 1.96)  /V_all
        # not sure how this would generalise to multiple 
        # h2_measured_up_orig
   
        allModels.append(     {"h2":h2_measured,"h2_ub":h2_CI95_ub,"h2_lb":h2_CI95_lb, "vc":VC, "ve":Ve, "vc_se":VC_SE, "p":p_vals, "bic":BIC, "domcol": colDominance  }   )    
        del H_E_regression  ; gc.collect()
    
    # save results to text file
    knet_IO.writeVCResults(args.out, allModels)



def HE_Multi (Y, all_relatedness) :
    n = Y.shape[0]
    ## calculate relatedness / IBS pairwise table for all individuals:
    indices = getCombinationIndices(n)
    Yi = np.take(Y, indices[:,0])
    Yj = np.take(Y, indices[:,1])
    
    all_Outcomes = (Yi - Yj)**2
        
    
    H_E_regression = regression.multiReg(all_Outcomes, all_relatedness) 


    
    # here, filter out all VCs that are non-significant ( p < 0.05) 
    Va_SE =  0.5 * H_E_regression["se"][1:len(H_E_regression["se"])]  # don't need the intercept
    Va = -0.5 * H_E_regression["beta"][1:len(H_E_regression["beta"])]    # don't need the intercept
    p_vals = H_E_regression["$p_values"][1:len(H_E_regression["p_values"])]
      

    SumVa_sig = 0 # the sum of the significant variance components
    for i in range( len(p_vals) )  : # only include significant Vas
        if p_vals[i] <= 0.05 :
            SumVa_sig = SumVa_sig +  Va[i] 
        
   
    
    V_all = np.var(Y)
    Ve =  V_all - SumVa_sig  # noise variance is the total Pheno variance minus ALL of the other variance components
    h2_measured = SumVa_sig / V_all  # Broad sense h2
    
  
      
    return ( {"h2":h2_measured, "va":Va, "ve":Ve, "va_se":Va_SE, "p":p_vals  } )
    
    
    # Proper way to estimate CI95 for h2: 
    # h2_measured_up_orig = (Va + Va_SE * 1.96)  /V_all
    # not sure how this would generalise to multiple 
    # h2_measured_up_orig
