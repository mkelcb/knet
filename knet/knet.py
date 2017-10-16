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


# Dependencies:
# numpy
# scipy



# https://docs.python.org/3/library/argparse.html
#https://docs.python.org/3/howto/argparse.html 
import argparse
from pathlib import Path


from com.application.logic.knet import knet_manager
from com.application.logic.scanner import scanner
from com.application.utils import plotgen

from com.io import knet_IO
import os
import gc
import numpy as np

def set_Threads(args) : 
    if args.threads is not None  :
        os.environ['MKL_NUM_THREADS'] = args.threads # '16'  # use h ere the N , where N: the number of cores acailable or limit to 1
        os.environ['MKL_DYNAMIC'] = 'FALSE'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        print("set MKL number of threads to: " + str(args.threads))
   
    
def set_nixMem(args) : 
    if args.nixMem is not None  :
        import resource # this only exists on Unix/Linux based systems
        rsrc = resource.RLIMIT_AS
        soft, hard = resource.getrlimit(rsrc)
        print('Soft limit starts as  :', soft)
        print('Hard limit starts as  :', hard)
        
        resource.setrlimit(rsrc, (args.nixMem * 1048576, hard)) #limit
        
        soft, hard = resource.getrlimit(rsrc)
        print('Soft limit changed to :', soft)
        print('Hard limit changed to  :', hard)


def rundom(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet recoding genotype matrix into dominance contrasts started') 
    from com.application.logic.reml import kinship
    import shutil
    
    # load data
    genotypeData = knet_IO.loadPLINK(args.bfile, loadPhenos = False, replaceMissing = True) 
    M = genotypeData["M"]
    del genotypeData ; gc.collect()  # dont need this
    
    # recode to dominance
    M = kinship.recodeSNPs_to_Dominance(M)

    #write it to disk
    knet_IO.writePLINK(args.out,M) 
    
    # need to copy the .bim/.fam , as pyplink does not write those...
    shutil.copyfile(args.bfile + ".bim", args.out +".bim")
    shutil.copyfile(args.bfile + ".fam", args.out +".fam")
    print("written dominance genotype data to", args.out)



def runmhe(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet estimating Variance Components via Multiple Haseman-Elston Regression started') 
    from com.application.utils import geno_qc
    # load plink binary (or eigen summary) / phenotypes
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : recodecc = False # IE if this is FALSE, then we will NOT recode 
  
    # load training data
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
    y = geno_qc.standardise_Genotypes(y)



    from com.application.logic.he import multi_he
    
    # depending on what MHE is requested run that
    if args.res == 0 : multi_he.HE_Multi_external(args, y)
    else : multi_he.HE_Multi_residuals(args, y)
    

    
    # concern, shouldn't we 'square' the XX^T, BEFORE dividing it by p ? and dividing it by p AFTER squaring it??
    # or that would blow things out of proportion??
    
    
    

def runkinmerge(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet merging Kinships started')  
    
    from com.application.logic.reml import kinship
    
    allKinLocations = knet_IO.loadKinshipLocations(args.allkins)
    
    # need to know the number of SNPs that were used for each kinship, so we can weight them according to the total
    kinshipSNPs = list()
    for i in range(len(allKinLocations)) :
        kinshipSNPs.append( knet_IO.load_N_fromGCTA_GRM( allKinLocations[i] ) )
        gc.collect()
    
    totalSNPs = np.sum(kinshipSNPs)
    kinship_total = None
    IDs = None
    print('will load', str(len(allKinLocations)) , " kinship matrices, with a total of ",str( int(totalSNPs) ), " SNPs", flush=True )
    
    for i in range(len(allKinLocations)) :
       K_new = knet_IO.loadGCTA_GRM(allKinLocations[i])
       K = K_new["K"]  
       currentSNPs = K_new["N"][0]
       IDs = K_new["ids"]
       weight = currentSNPs/totalSNPs
       del K_new; gc.collect()
       if kinship_total is None : # if this is the first kiship we have loaded
            kinship_total = K   *  weight
       else :
            kinship_total  = kinship_total + K   *  weight
            

       del K; gc.collect()
       print("merged kinship", (i+1) , " out of:", len(allKinLocations),  "weighed at:", weight, flush=True )
      
    knet_IO.writeGCTA_GRM(args.out,kinship_total, IDs, totalSNPs)
    # K_totalLoaded = loadGCTA_GRM('../../../0cluster/results/broadsenseh2/kinship_total')["K"]

def runcalckins(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet calculating Kinship started')  
    

    from com.application.logic.reml import kinship
    from com.application.utils import geno_qc
    
    # load data
    genotypeData = knet_IO.loadPLINK(args.bfile, loadPhenos = False) 
    M = genotypeData["M"]
    irsIds = genotypeData["rsid"] 
    IDs = genotypeData["IDs"] 
    del genotypeData ; gc.collect()  # dont need this
    
    # QC data
    qc_data = geno_qc.genoQC_all(M, rsIds = irsIds)
    M = qc_data["X"]
    rsIds_qc = qc_data["rsIds"] # will need these, as after removing some SNPs, I cannot just appyl the LDAK weights, but will need to match them up....
    indicesToRemove = qc_data["indicesToRemove"] # i might actually be able to get away by using this o nthe LDAK weights
    MAFs =qc_data["MAFs"] # in case this will be needed to apply a score based on it
    del qc_data; del irsIds; gc.collect()
    
    
    # load weights if any
    weights = None
    if args.MAFS : # if MAF Score weights are requested
        print("computing MAF scores", flush=True)
        weights = geno_qc.computeMAFScore(MAFs, args.MAFS)
      
    if args.weights : # load LDAK weights
        LDAK_weights = knet_IO.loadLDAKWeights(args.weights)
        gc.collect()
        print("Loaded number of LDAK Weights: ", len(LDAK_weights), flush=True)
        LDAK_weights = np.delete(LDAK_weights, indicesToRemove) # delete any SNPs that were removed the internal QC process 
        print("After deleting QCd SNPs(",len(indicesToRemove),") remeining weights: ", len(LDAK_weights), flush=True)
        if weights is not None: # if we requested MAF Scores as wegihts then the total weights will be their product
            weights = weights * LDAK_weights
        else  : weights = LDAK_weights # otherwise just use the LDAK weights
        
        # the LDAK weights set some SNPs to 0, this will then cause zscore to fail ( dividing by std dev 0)
        # solution: remove all 0 values from both the weights as well as from the M design matrix
        nonZeroWeightsIndices = np.nonzero(weights)[0] # get all nonzero weight's indices
        M = np.take(M, nonZeroWeightsIndices, axis=1) # only keep these for M
        weights = np.take(weights, nonZeroWeightsIndices) # as well as for the weights
    
    if weights is not None : weights = np.diag(weights) # turn it into a diagonal matrix
        
    numSNPs = M.shape[1]
    
    # if a dominance kinship was requested
    if args.dominance     :
        print("computing dominance kinship", flush=True)
        # get dominance kinship
        K_dominance = kinship.calc_Kinship_Dominance(M, weights) ## apply weightsfor LD / MAFS (if any)
        gc.collect()
        # write it to disk then dispose
        knet_IO.writeGCTA_GRM(args.out + "_dom",K_dominance, IDs, numSNPs)
        del K_dominance; gc.collect()
    
    # Additive Kinship
    print("computing additive kinship", flush=True)
    if weights is not None: M = M.dot(weights)
    M = geno_qc.standardise_Genotypes(M)
    K_additive = kinship.calc_Kinship(M)
    knet_IO.writeGCTA_GRM(args.out + "_add",K_additive, IDs, numSNPs)
 
#    
#    
#    part = M.shape[1] / 4
#    part1 = int( (part *2) )
#    part2 = part1 + int( (part) )
#    part3 =  M.shape[1]
#
#    M1 = M[:,0:part1]
#    M2 = M[:,part1:part2]
#    M3 = M[:,part2:part3]
#    M1.shape[1] + M2.shape[1] + M3.shape[1]
#   
#    
#    K_additive = calc_Kinship(M1)
#    knet_IO.writeGCTA_GRM(args.out + "_part1",K_additive, IDs, M1.shape[1])
#    
#    K_additive = calc_Kinship(M2)
#    knet_IO.writeGCTA_GRM(args.out + "_part2",K_additive, IDs, M2.shape[1])
#    
#    K_additive = calc_Kinship(M3)
#    knet_IO.writeGCTA_GRM(args.out + "_part3",K_additive, IDs, M3.shape[1])
#    
#    
#    del K_additive; gc.collect()
#    
#    del M; del M1; del M2; del M3; del MAFs; del rsIds_qc;
#    gc.collect()
#
#    K_additive_total = calc_Kinship(M)
#    knet_IO.writeGCTA_GRM(args.out + "_total",K_additive_total, IDs, M.shape[1])

              
                                 
def runrrblup_big(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet RidgeRegression BLUP started')  
    
    from com.application.logic.reml import reml
    from com.application.logic.reml import kinship
    from com.application.utils import geno_qc
    # load plink binary (or eigen summary) / phenotypes
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : recodecc = False # IE if this is FALSE, then we will NOT recode 
  
    # load training data
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 

    genotypeData = knet_IO.loadPLINK(args.bfile, loadPhenos = False) 
    M = genotypeData["M"]
    irsIds = genotypeData["rsid"]
    del genotypeData ; gc.collect()  # dont need this
    
    qc_data = geno_qc.genoQC_all(M, rsIds = irsIds)
    M = qc_data["X"]
    rsIds_qc = qc_data["rsIds"] # save away the surviving SNP list that we have used 
    indicesToRemove = qc_data["indicesToRemove"]
    del qc_data; gc.collect()
    M = geno_qc.standardise_Genotypes(M) # overwrite M dont store it 2x
    gc.collect()
    
    # get SNP coefs
    results = reml.computeBLUPs_RidgeBLUP(y, M, args.delta)
    Beta = results.BETA
    del M; del results; gc.collect()
    print("computed Ridge Coefs")
    # load validation sets
    #y_validation = knet_IO.loadPLINKPheno(args.validPhen, caseControl = cc, recodeCaseControl = recodecc) 
    
    genotypeData = knet_IO.loadPLINK(args.validSet, loadPhenos = False) 
    M_validation = genotypeData["M"] 
    del genotypeData ; gc.collect() # dont need this
    qc_data = geno_qc.removeList(M_validation, indicesToRemove)
    M_validation = qc_data["X"] 
    del qc_data; gc.collect() 
    M_validation= geno_qc.standardise_Genotypes(M_validation) 
    gc.collect() 
    print("After standardising, validation data in MBs is: ",geno_qc.getSizeInMBs(M_validation) )



    # make predictions
    yhat = reml.predictYhat(M_validation, Beta)
    # this is a 1D array, not 2D
   
    fileName = args.out + "yhat.txt"
    with open(fileName, "w") as file:
        file.write("Profile"  + "\n")
        for i in range( len(yhat) ) :
            line = str(yhat[i] )

                
            file.write( line +  "\n")   # file.write(  ( str(yhat[i])[2:-1] ).replace("  ", " ").replace(" ", "\t") +  "\n")

## same a sother but it uses the Ridge formula of XXt, IE will use much less memory
def runrrblup(args) :
    print("runrrblup", flush=True)
    set_Threads(args)
    set_nixMem(args) 
    print('Knet RidgeRegression BLUP started', flush=True)  
    
    from com.application.logic.reml import reml
    from com.application.logic.reml import kinship
    from com.application.utils import geno_qc
    # load plink binary (or eigen summary) / phenotypes
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : recodecc = False # IE if this is FALSE, then we will NOT recode 
    # load training data
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
    genotypeData = knet_IO.loadPLINK(args.bfile, loadPhenos = False) 
    M = genotypeData["M"]
    irsIds = genotypeData["rsid"]
    del genotypeData ; gc.collect()  # dont need this
    qc_data = geno_qc.genoQC_all(M, rsIds = irsIds)
    M = qc_data["X"]
    rsIds_qc = qc_data["rsIds"] # save away the surviving SNP list that we have used 
    indicesToRemove = qc_data["indicesToRemove"]
    del qc_data; gc.collect()
    M = geno_qc.standardise_Genotypes(M) # overwrite M dont store it 2x
    gc.collect()
    
    # get SNP coefs
    g = reml.computeBLUPs_RidgeBLUP_morep(y, M, args.delta)
    Beta = reml.backCalculate_Beta_BLUP(g,M) 
    del M; gc.collect()
    print("computed Ridge Coefs", flush=True)
    # load validation sets
    #y_validation = knet_IO.loadPLINKPheno(args.validPhen, caseControl = cc, recodeCaseControl = recodecc) 
    
    genotypeData = knet_IO.loadPLINK(args.validSet, loadPhenos = False) 
    M_validation = genotypeData["M"] 
    del genotypeData ; gc.collect() # dont need this
    qc_data = geno_qc.removeList(M_validation, indicesToRemove)
    M_validation = qc_data["X"] 
    del qc_data; gc.collect() 
    M_validation= geno_qc.standardise_Genotypes(M_validation) 
    gc.collect() 
    print("After standardising, validation data in MBs is: ",geno_qc.getSizeInMBs(M_validation) , flush=True)

    #fileName = args.out + "beta.txt"
    #with open(fileName, "w") as file:
    #    file.write("Beta"  + "\n")
    #    for i in range( len(Beta) ) :
    #        line = str(Beta[i] )   
    #        file.write( line +  "\n")  


    indicesToRemove = list()
    for i in range( M_validation.shape[1] ) :
        if np.isnan(M_validation[:,i]).any() :
            print("!!!Validation set has NaNs at column:" + str(i), flush=True)
            indicesToRemove.append(i) 
            
    # need to remove any NaNs from both the genotype matrix and its corresponding Beta
    print("num SNPs BEFORE removing: ",  Beta.shape[0] )
    M_validation = np.delete(M_validation, indicesToRemove, axis=1)
    Beta = np.delete(Beta, indicesToRemove, axis=0)
    print("num SNPs AFTER removing: ",  Beta.shape[0] )
         
    
    
    # make predictions
    yhat = reml.predictYhat(M_validation, Beta)

    # this is a 1D array, not 2D
   
    fileName = args.out + "yhat.txt"
    with open(fileName, "w") as file:
        file.write("Profile"  + "\n")
        for i in range( len(yhat) ) :
            line = str(yhat[i] )   
            file.write( line +  "\n")  
   
        
def runh2(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet H2 analyis started')  
    if args.eig is None and args.bfile is None and args.K is None:
        print("either an eigen decomposition or a PLINK binary or a Kinship matrix is requried")

    else :
        from com.application.logic.reml import reml
        from com.application.logic.reml import kinship
        from com.application.utils import geno_qc
        # load plink binary (or eigen summary) / phenotypes
        cc = True
        if args.cc == 0 : cc = False
       
        recodecc = True
        if args.recodecc == 0 : recodecc = False # IE if this is FALSE, then we will NOT recode 
      
        y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
        y = geno_qc.standardise_Genotypes(y)
        
        if args.eig is None and args.K is None : # if an eigen summary wasn't supplied, IE we don't have it
            print("calculating REML from sratch, no Eigen summary was supplied")
            genotypeData = knet_IO.loadPLINK(args.bfile, loadPhenos = False) 
            M = genotypeData["M"]

            qc_data = geno_qc.genoQC_all(M, rsIds = genotypeData["rsid"])
            M = qc_data["X"]
            gc.collect()
            M = geno_qc.standardise_Genotypes(M) # overwrite M dont store it 2x
            gc.collect()
            K = kinship.calc_Kinship( M  ) # 3. create kinship matrix from block 
            del M ; gc.collect()  # delete M as we no longer need it   
            results = reml.REML_GWAS(y, K) # 4. check if there is any h2 in this block via EMMA
            del K ; gc.collect()
        
        elif args.K : # if a kinship matrix was supplied
            print("calculating REML from Kinship matrix supplied")
            K_new = knet_IO.loadGCTA_GRM(args.K)
            K = K_new["K"]  
            del K_new; gc.collect()
       
            results = reml.REML_GWAS(y, K) # 4. check if there is any h2 in this block via EMMA
        
        else :
            print("loading saved eigen sums from: " + args.eig)
            loadedEigSum = knet_IO.loadEigenSum(args.eig)[0] # load eigen decomposition
            results = reml.REML_GWAS(y, eigenSummary = loadedEigSum) # 4. check if there is any h2 in this block via EMMA


        
        eigSum = results["eigSum"] # just resave the one we have got  
        h2 = results["vg"] / ( results["vg"] + results["ve"])
        h2_SE = reml.h2_SE_approx2(y, eigSum.values)

        print("h2: " , h2 , " / h2 SE: ", h2_SE, " / delta: ", results["delta"])
        fileName = args.out + "reml.txt"
        with open(fileName, "w") as file:
            file.write("h2=" + str(h2)  + "\n")
            file.write("h2_SE=" + str(h2_SE)  + "\n")
            file.write("delta=" + str(results["delta"])  + "\n")
            file.write("ve=" + str(results["ve"])  + "\n")
            file.write("vg=" + str(results["vg"])  + "\n")
            file.write("REML_LL=" + str(results["REML"])  + "\n")
            
        # now write out the eigen summaries too ( but only if it wasn't supplied in the first place)
        if args.eig is None : knet_IO.writeEigenSum(args.out,  [eigSum ] )  # the below function expects a list


def runKnet(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet Neural net started')

    # load regions
    regions = None
    amblup_regions = None # this has a different structure
    if args.regions :
        regions =  knet_IO.loadRegionData(args.regions) # load regions#
        
    elif  args.amblupregions and args.amblupreml :
        print('loading AMLBUP regions')

        remlData = knet_IO.loadLDAKRegionsDeltas(args.amblupreml)
        regionData = knet_IO.loadLDAKRegions(args.amblupregions)
        amblup_regions = {"REGIONS":regionData, "DELTAS":remlData, "CONV":args.conv  }

    priors = None
    if args.priors :
        priors =  knet_IO.loadSummaryStats(args.priors) # load 'priors'
    

    
    # check if we need to load saved state weights
    loadedWeightsData = None
    
    if args.loadWeights :

        loadedWeightsData =list()
        moreFiles = True
        counter = 0
        
        while moreFiles : # keep going until we run out of files to load
           # currentLocation = args.loadWeights + str(counter)  # files are expected to be named as regions1.txt, regions2.txt, etc
            my_file = Path(args.loadWeights + "_"+ str(counter) +"_0.bin") # check if the main weights file exists
                          
            if my_file.is_file(): # check if it exists
                loadedWeightsData.append(list())
                
                for j in range(4) :  # there are 4 filtes, 1 for each, W, W bias, momentum and Momentub bias
                   loadedWeightsData[counter].append( knet_IO.loadMatrixFromDisk(args.loadWeights + "_"+ str(counter) + "_" + str(j)) )
                    # each weight is a matrix ( even the biases), and they are coded as name_LAYER_W/Bias/Momentum/Momentum_bias (so chrom_0_0 is layer 1's Weights_W)
                counter = counter +1
            else : moreFiles = False
            
        
     
    # pass this into knet manager, along with all the conditional params
    knet_results = knet_manager.runKnet(args, args.epochs, args.learnRate, args.momentum, regions, args.evalFreq, args.savFreq, args.predictPheno, loadedWeightsData, args.saveWeights, args.randomSeed, args.hidCount, args.hidl2, args.hidAct, amblup_regions, priors)
    gc.collect() 


    
    # write epoch results out
    results_its = knet_results["results"]["results"]              
    fileName = args.out + "nn_results.txt"
    with open(fileName, "w") as file: 
        
        line = "epochs"
        if "train_accuracy" in results_its: line = line + "\t" + "train_accuracy"
        if "test_accuracy" in results_its: line = line + "\t" + "test_accuracy"
        file.write(line  + "\n")
         
        for i in range( len(results_its["epochs"])  ):
            line = str(results_its["epochs"][i]) 
            if "train_accuracy" in results_its: line = line + "\t" + str(results_its["train_accuracy"][i])
            if "test_accuracy" in results_its: line = line + "\t" + str(results_its["test_accuracy"][i])
            file.write(line + "\n")            
        
    
    # generate plot of the results
    if len(results_its["epochs"]) > 0 :
        plotgen.exportNNPlot(results_its, args.out + "nnplot")
    
    
    # write out the SNPs that were used for the analysis
    rsIds = knet_results["rsIds"]
    fileName = args.out + "nn_SNPs.txt"
    with open(fileName, "w") as file: 
        for i in range( len(rsIds)  ):
            file.write(rsIds[i]  + "\n")
         


    # write final predictions out ( if this was requested)
    yhat = knet_results["yhat"]
   
    # recode yhat into single col: I think this is a bad idea as this will basically threshold everyone to be all 1s
    outputShape = 1
    if len(yhat.shape) > 1 : outputShape = yhat.shape[1]
    # if outputShape > 1 : yhat = knet_IO.recodeOneHotCaseControl(yhat)
    
    if yhat is not None :    
        fileName = args.out + "yhat.txt"
        with open(fileName, "w") as file:
            file.write("Profile"  + "\n")
            for i in range(yhat.shape[0]) :
                line = str(yhat[i][0] )
                for j in range(1, len(yhat[i]) ):
                    line = line + "\t" + str(yhat[i][j] )
                    
                file.write( line +  "\n")   # file.write(  ( str(yhat[i])[2:-1] ).replace("  ", " ").replace(" ", "\t") +  "\n")

          
            
            
            
            
    # write final weights out
    # results["weights"]
    weights_nn = None
    if knet_results["weights"] is not None:
        weights_nn = knet_results["weights"]
        for i in range(len(weights_nn)) :
            for j in range(len(weights_nn[i])) :
                knet_IO.writeMatrixToDisk( args.saveWeights + "_" + str(i)+ "_" + str(j)  , weights_nn[i][j])
                # each weight is a matrix ( even the biases), and they are coded as name_LAYER_W/Bias/Momentum/Momentum_bias (so chrom_0_0 is layer 1's Weights_W)
   
            

  
def runScanner(args) : 
    set_nixMem(args) 
    set_Threads(args)
    print("Knet scanner started")

    # check if we want to load the eigen decomposition summaries for each region or not
   # loadedEigSum = None
  #  if args.loadEigSum is not None :
  #      print("loading saved eigen sums from: " + args.loadEigSum)
  #      loadedEigSum = knet_IO.loadEigenSum(args.loadEigSum)
    
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : recodecc = False

    # load plink binary / phenotypes
    genotypeData = knet_IO.loadPLINK(args.scanner, loadPhenos = False) 
    M = genotypeData["M"]
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
 
    # obtain regions
    regionResults = scanner.findRegions(y, M, irsIds = genotypeData["rsid"], blockSize = args.filterSize, stride = args.stride,  X = None)

    # check if we want to save the eigen decomposition summaries for each region or not
 #   if args.saveEigSum is not None:
  #      print("saving eigen decompositions to: " + args.saveEigSum)
 #       knet_IO.writeEigenSum(args.saveEigSum, regionResults["eigSum"] )
        
    # write regions onto disk
    knet_IO.writeRegionData(args.out,regionResults["REGIONS"], regionResults["DELTAS"])
    print("written regions to: "  + args.out)
    
    
    # write out the SNPs that were used for the analysis
    rsIds = regionResults["rsIds"]
    fileName = args.out + "_SNPs.txt"
    with open(fileName, "w") as file: 
        for i in range( len(rsIds)  ):
            file.write(rsIds[i]  + "\n")
            

    
    
def runMerge(args) :  
    set_nixMem(args) 
    set_Threads(args)
    print('Knet merging started, from: ' + args.merge + " to: "  + args.out)
    # check for other required arguments

    location = args.merge
    outLocation = args.out
    
    moreFiles = True
    counter = 1
    allRegions = list()
    while moreFiles : # keep going until we run out of files to load
        currentLocation = location + str(counter) + ".txt" # files are expected to be named as regions1.txt, regions2.txt, etc
        my_file = Path(currentLocation)

        if my_file.is_file(): # check if it exists
            allRegions.append( knet_IO.loadRegionData(currentLocation) ) # load regions
        else : moreFiles = False
        counter = counter +1
    
    # concat them into a single list
    results = scanner.concatRegions(allRegions)
    
    # write these onto disk
    knet_IO.writeRegionData(outLocation,results["REGIONS"], results["DELTAS"] )


##################################################################################################
# setup COmmand line parser
##################################################################################################

parser = argparse.ArgumentParser()



# overall
parser.add_argument("--out",required=True, help='an output location is always required')
parser.add_argument("--threads",required=False, help='set number of threads used by multithreaded operations')
parser.add_argument("--nixMem",required=False, type=int, help='Memory limit for *nix based systems in Megabytes')


subparsers = parser.add_subparsers()
subparsers.required = True
subparsers.dest = 'either knet, scanner, h2, kinship, kinmerge or merge' # hack to make subparser required

# create the parser for the "a" command
parser_knet = subparsers.add_parser('knet')
parser_knet.add_argument('--knet', required=True) # the location of the train set binaries
parser_knet.add_argument("--pheno", required=True)
parser_knet.set_defaults(func=runKnet)

# knet subparams
parser_knet.add_argument("--regions", required=False)  # ,required=False   
parser_knet.add_argument("--loadWeights") # from where we want to load the weights
parser_knet.add_argument("--saveWeights") # where we wnt to save weights
parser_knet.add_argument("--savFreq", default=-1, type=int) # how frequently we make backups of Weights
parser_knet.add_argument("--epochs", default=100, type=int) # how many epochs
parser_knet.add_argument("--learnRate", default=0.005, type=float) 
parser_knet.add_argument("--momentum", default=-1, type=float)   # -1 means 'disabled'
parser_knet.add_argument("--validSet") # the location for the binaries for the validation set
parser_knet.add_argument("--validPhen") # the location for the binaries for the validation set phenotypes
parser_knet.add_argument("--evalFreq", default=10, type=int) # how frequently we evaluate prediction accuracy (-1 for disabled)                     
parser_knet.add_argument("--cc", type=int)  # ,required=False  # if phenotype is case control
parser_knet.add_argument("--recodecc", type=int)  # ,required=False       # if we want to recode case control to quantitative
parser_knet.add_argument("--randomSeed", default=1, type=int)                       
parser_knet.add_argument("--hidCount", default=0, type=int)     # number of hidden layers
parser_knet.add_argument("--hidl2", default=0.0, type=float)        # the L2 regularizer shrinkage param       
parser_knet.add_argument("--hidAct", default=0, type=int)        # the hidden layer activations ( 0 = softplus, 1 = sigmoid, 2 = leaky RELU, 3 = linear)
parser_knet.add_argument("--amblupregions") # amblup regions directory, this contains the number of regions and the SNPs in each region
parser_knet.add_argument("--amblupreml") # amblup reml file location, this contains the regional heritabilities                  
parser_knet.add_argument("--conv", default=0, type=int)  # if we  should use locally connected / convolutional topology whn using amblup
parser_knet.add_argument("--priors") # from where we want to load the SNP 'priors'
          
# parser_knet.add_argument("--topology", required=True) # the location of the file that describes the network's topology (IE number and size of layers etc)
parser_knet.add_argument("--predictPheno", default=-1, type=int) # if network should save phenotype predictions to a location at the end, for a validation set                  
               
                        
parser_scanner = subparsers.add_parser('scanner')
parser_scanner.add_argument('--scanner', required=True)
parser_scanner.add_argument("--pheno", required=True)
parser_scanner.set_defaults(func=runScanner)

parser_merge = subparsers.add_parser('merge')
parser_merge.add_argument('--merge', required=True)
parser_merge.set_defaults(func=runMerge)

# narrow sense h2 analysis
parser_h2 = subparsers.add_parser('h2')
parser_h2.add_argument('--eig') # the location of the eigen decomposition
parser_h2.add_argument('--bfile') # the location of the plink binaries 
parser_h2.add_argument('--K') # the location of the Kinship matrix     
parser_h2.add_argument("--pheno", required=True)
parser_h2.add_argument("--cc", type=int)  # ,required=False
parser_h2.add_argument("--recodecc", type=int)  # ,required=False     
parser_h2.set_defaults(func=runh2)
 

# Ridge-BLUP
parser_rrblup = subparsers.add_parser('rrblup')
parser_rrblup.add_argument('--bfile', required=True) # the location of the plink binaries 
parser_rrblup.add_argument("--delta", required=True, type=float)
parser_rrblup.add_argument("--pheno", required=True)
parser_rrblup.add_argument("--cc", type=int)  # ,required=False
parser_rrblup.add_argument("--recodecc", type=int)  # ,required=False     
parser_rrblup.set_defaults(func=runrrblup)
parser_rrblup.add_argument("--validSet", required=True) # the location for the binaries for the validation set
#parser_rrblup.add_argument("--validPhen", required=True) # the location for the binaries for the validation set phenotypes

                           
# Kinship
parser_kin = subparsers.add_parser('kinship')
parser_kin.add_argument('--bfile', required=True) # the location of the plink binaries 
parser_kin.add_argument("--dominance") # if we should compute dominance instead of the usual additive kinship
parser_kin.add_argument('--weights') # weights used to scale the SNPs ( usually from LDAK)
parser_kin.add_argument('--MAFS', type=float) # alpha: if Score based on MAF should be used (MAF(1-MAF))^(1-alpha) as per speed at al
parser_kin.set_defaults(func=runcalckins)
                     
# Kinship merging
parser_kinmerge = subparsers.add_parser('kinmerge')
parser_kinmerge.add_argument('--allkins', required=True) # a text file that contains a list of kinship matrices to be merged, 1 per line without header (GCTA format)
parser_kinmerge.set_defaults(func=runkinmerge)
                     
                           
                    
# Multi-HE
parser_mhe = subparsers.add_parser('mhe')
parser_mhe.add_argument('--addkin', required=True) # location of the additive kinship matrix
parser_mhe.add_argument('--domkin') # location of the dominance kinship matrix
parser_mhe.add_argument("--epi", default=-1, type=int) # the level of epistasis (disabled if < 2)
parser_mhe.add_argument("--pheno", required=True)
parser_mhe.add_argument("--cc", type=int)  # ,required=False
parser_mhe.add_argument("--recodecc", type=int)  # ,required=False   
parser_mhe.add_argument("--res", default=0, type=int) # if the MHER is running on original (0) or Residualised version (1)
parser_mhe.set_defaults(func=runmhe)
     
    
# recoding the PLINK binary int dominance contrasts 
parser_redom = subparsers.add_parser('redom')
parser_redom.add_argument('--bfile', required=True) # the location of the plink binaries 
parser_redom.set_defaults(func=rundom)
 
           
                          
                           

# scanner subparams
parser_scanner.add_argument("--stride", default=25, type=int)  # ,required=False
parser_scanner.add_argument("--filterSize", default=50, type=int)  # ,required=False
parser_scanner.add_argument("--saveEigSum")  # ,required=False
parser_scanner.add_argument("--loadEigSum")  # ,required=False
parser_scanner.add_argument("--cc", type=int)  # ,required=False
parser_scanner.add_argument("--recodecc", type=int)  # ,required=False          
                   
# retreive command line arguments
args = parser.parse_args()
args.func(args)

# toy test
# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/toyregions_ --threads 2 scanner --scanner /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy --pheno /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy.pheno --saveEigSum /nfs/users/nfs_m/mk23/test/pytest/toyeig



# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/22 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22.pheno --saveEigSum /nfs/users/nfs_m/mk23/test/pytest/f1/22eig_
#python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/22 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22.pheno

#python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/21 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/21 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/21.pheno


#python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/15 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/15 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/15.pheno

#python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/1 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/1 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/1.pheno



# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/15_s100_ --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/15 --filterSize 100 --stride 50 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/15.pheno


# python knet.py --out /nfs/users/nfs_m/mk23/test/pytest/toyregions_ --threads 8 scanner --scanner /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy --pheno /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy.pheno --loadEigSum /nfs/users/nfs_m/mk23/test/pytest/toyeig


#   python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/22_s100d_ --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22 --filterSize 100 --stride 50 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22.pheno

# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/22_s100t_ --threads 2 scanner --scanner /nfs/users/nfs_m/mk23/test/pytest/f1/22_toy_long --filterSize 100 --stride 50 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22.pheno


##################################################################################################
##################################################################################################
# Local Tests

# SCANNER
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/regions22_','scanner', '--scanner','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno', '--stride', '50', '--filterSize', '100']) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
#args.func(args)

# KNET MAIN
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/chr22','knet', '--knet','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno',  '--regions', '../../../0cluster/results/knettest/regions22_', '--epochs', '10', '--learnRate', '0.00005', '--momentum', '0.9', '--validSet', '../../../0cluster/data/knettest/22_toy_long_valid' ,'--validPhen', '../../../0cluster/data/knettest/22_toy_long_valid.pheno', '--evalFreq', '10' ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'

# Knet main as case control one hot                  
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/chr22','knet', '--knet','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno',  '--regions', '../../../0cluster/results/knettest/regions22_', '--epochs', '10', '--learnRate', '0.00005', '--momentum', '0.9', '--validSet', '../../../0cluster/data/knettest/22_toy_long_valid' ,'--validPhen', '../../../0cluster/data/knettest/22_toy_long_valid.pheno', '--evalFreq', '10',  '--recodecc'    , '0' ,  '--hidCount'    , '5' ,  '--hidl2'    , '0.2' ,  '--hidAct'    , '2'    ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
                      
                        
                        
 # load / save weights                   
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/chr22','knet', '--knet','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno',  '--regions', '../../../0cluster/results/knettest/regions22_', '--epochs', '10', '--learnRate', '0.00005', '--momentum', '0.9', '--validSet', '../../../0cluster/data/knettest/22_toy_long_valid' ,'--validPhen', '../../../0cluster/data/knettest/22_toy_long_valid.pheno', '--evalFreq', '10' , '--loadWeights',  '../../../0cluster/results/knettest/chr22']) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
  
                        
# h2 analysis:
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/h2/chr1','h2', '--bfile','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno']) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'

# same as above but  loading eigsum
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/h2/chr1','h2', '--eig','../../../0cluster/results/knettest/h2/chr1', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno']) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
 
                        
# Ridge BLUP
args = parser.parse_args(['--out', '../../../0cluster/results/broadsenseh2/bluptest','rrblup', '--bfile','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno', '--validSet', '../../../0cluster/data/knettest/22_toy_long_valid', '--delta', '20.0'])
 
                        
#Kinship
args = parser.parse_args(['--out', '../../../0cluster/results/broadsenseh2/kinship','kinship', '--bfile','../../../0cluster/data/knettest/22_toy_long_train', '--weights', '../../../0cluster/data/knettest/weights.short', '--MAFS', '-0.25',  '--dominance' , '1']) 
 
                        
                    
#Kinship merge
args = parser.parse_args(['--out', '../../../0cluster/results/broadsenseh2/kinship_merged','kinmerge', '--allkins','../../../0cluster/results/broadsenseh2/kinlist.txt']) 
                      

# Multi HE
args = parser.parse_args(['--out', '../../../0cluster/results/broadsenseh2/heh2','mhe', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno', '--addkin','../../../0cluster/results/broadsenseh2/kinship_add', '--domkin','../../../0cluster/results/broadsenseh2/kinship_dom', '--epi','2', '--res', '1']) 
   

# GCTA kinship
args = parser.parse_args(['--out', '../../../0cluster/results/gcta/knetver','h2', '--bfile','../../../0cluster/results/gcta/wtccc2_hg19_toy', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno']) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'


# recode to Dominance
args = parser.parse_args(['--out', '../../../0cluster/results/gcta/knetdom','redom', '--bfile','../../../0cluster/results/gcta/wtccc2_hg19_toy']) 

       

# Knet AMBLUP regions
args = parser.parse_args(['--out', '../../../0cluster/results/knet_amblup/kamblup','knet', '--knet','../../../0cluster/data/data/toy/wtccc2_hg19_toy', '--pheno', '../../../0cluster/data/data/toy/wtccc2_hg19_toy.pheno', '--epochs', '10', '--learnRate', '0.00005', '--momentum', '0.9', '--validSet', '../../../0cluster/data/data/toy/wtccc2_hg19_toy' ,'--validPhen', '../../../0cluster/data/data/toy/wtccc2_hg19_toy.pheno', '--evalFreq', '10',  '--recodecc'    , '1' ,  '--hidCount'    , '5' ,  '--hidl2'    , '0.2' ,  '--hidAct'    , '2'      ,  '--amblupreml'    , '../../../0cluster/data/data/toy/amblup_1.reml'  ,  '--amblupregions'    , '../../../0cluster/data/data/toy/chunks_amb1/'  ,  '--conv'    , '0'  ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'


  
#        
# --saveWeights
# --savFreq


                        
                        
                        
                        
