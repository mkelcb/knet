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

import gc
import numpy as np
from numpy.linalg import norm 
from scipy import stats
from pathlib import Path
import random
import os
import time
import sys 
import matplotlib.pyplot as plt   
from functools import partial
from types import SimpleNamespace
import copy
from ast import literal_eval



from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_scores
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from ....application.utils.plotgen import exportNNPlot
from ....application.utils.geno_qc import removeList, genoQC_all, standardise_Genotypes, getSizeInMBs 
from ....application.logic.knet.knet_main_pytorch import weight_init, EPSILON, learn, registerDeCovHooks, setModelMode , NETWORK_DATATYPE, getNetworkDatatype_numpy, getModel
from ....io import knet_IO
from ....application.logic.reml import reml
from ....application.utils import regression

# delta = (Ve/Vg)
# delta = (1-h2) / h2
#args, args.epochs, args.learnRate, args.momentum, args.evalFreq, args.savFreq, args.predictPheno, args.loadWeights, args.saveWeights, args.randomSeed, args.hidCount, args.hidl2, args.hidAct

#args = parser.parse_args(['--out', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/tests/0pytorch_tests/' ,'knet', '--knet', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/data/genetic/short', '--pheno', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/data/genetic/phenew.pheno.phen',  '--epochs', '21', '--learnRate', '0.00005', '--momentum', '0.9', '--evalFreq', '1',  '--recodecc'    , '0' ,  '--hidCount'    , '5' ,  '--hidl2'    , '1.0' ,  '--hidAct'    , '2' , '--cc', '0' ,'--inference', '0'   ]) # ,'--loadWeights', 'C:/0Datasets/NNs/genetic/weights/' ,'--snpIndices', 'C:/0Datasets/NNs/genetic/nn_SNPs_indices.txt' ,'--mns', 'C:/0Datasets/NNs/genetic/data_mns','--sstd', 'C:/0Datasets/NNs/genetic/data_sstd','--snpIDs', 'C:/0Datasets/NNs/genetic/nn_SNPs.txt'


###############################################################################
# QC and Training
###############################################################################      
def runKnet(args) :
    print("OLS Baseline")
    # default QC settings
    _minObserved = 0.95
    _minMAF = 0.01
    _minVariance = 0.02
    
    # load plink binary / phenotypes want to load them here, so that we can release the memory once the raw data is no longer used
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : recodecc = False
    start = time.time()
    genotypeData = knet_IO.loadPLINK(args.knet, loadPhenos = False) 
    A1_alleles = genotypeData["A1"]
    M = genotypeData["M"]
    irsIds = genotypeData["rsid"]
    IDs = genotypeData["IDs"] 
    IDs[0] = np.array(IDs[0]) ; IDs[1] = np.array(IDs[1])
    indicesKept = np.asarray( range(M.shape[1]) )
    
    del genotypeData ; gc.collect() # dont need this
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
    y = stats.zscore(y) # zscore it so that Beta -> h2 computations work    
    y = y.reshape(-1,1) # enforce 2D

    # if we have a validation set
    M_validation = None
    y_validation = None
    if args.validSet :

        genotypeData = knet_IO.loadPLINK(args.validSet, loadPhenos = False, replaceMissing = True) # , replaceMissing = True want to replace -1 with 0s, as we otherwise would have -1s, as later we just delete indices that failed QC for the training set, but won't care for individual missing datas
        M_validation = genotypeData["M"] 
        #M = M_validation.copy()
        #print("BEFORE STANDARDISING: M and M_validation are equal:", np.array_equal(M,M_validation) )
        IDs_validation = genotypeData["IDs"] 
        del genotypeData ; gc.collect() # dont need this
        
        if args.validPhen :
            y_validation = knet_IO.loadPLINKPheno(args.validPhen, caseControl = cc, recodeCaseControl = recodecc) 
            y_validation = stats.zscore(y_validation) # zscore it so that Beta -> h2 computations work
            y_validation = y_validation.reshape(-1,1) # enforce 2D
    end = time.time(); printElapsedTime(start,end, "loading data took: ")
        
    if args.inference == 0 : 
        # 1. standardise data
        if args.qc == 1 :
            start = time.time()
            qc_data = genoQC_all(M, rsIds = irsIds, minObserved = _minObserved, minMAF = _minMAF, minVariance = _minVariance) # we MUST perform QC with the EXACT SAME settings as the 'region scanner' otherwise the region coordinates will be mismatched
            #M = qc_data["X"] 
            rsIds_qc = qc_data["rsIds"] # save away the surviving SNP list that we have used 
            indicesToRemove = qc_data["indicesToRemove"]
            indicesKept = qc_data["indicesKept"]
            irsIds = rsIds_qc.tolist()
            
            del qc_data; gc.collect() # overwrite
            
            qc_data = removeList(M, indicesToRemove)
            
            A1_alleles = np.delete(A1_alleles, indicesToRemove)

            
            M = qc_data["X"]
            del qc_data; gc.collect() # overwrite
            end = time.time(); printElapsedTime(start,end, "QC took: ")
        else :
            M[M==-1] = 0 # have to make sure that the missing genotype is NOT encoded as -1, even when we don't perform QC
            print("Skipping internal QC", flush=True)
        start = time.time()
        M, mns, sstd = standardise_Genotypes(M) ; gc.collect()
        end = time.time(); printElapsedTime(start,end, "standardising data took: ")
        print("After standardising, training data in MBs is: ",getSizeInMBs(M) )
    else :
        print("Inference data QC", flush=True)
        if args.snpIndices is not None :
            indicesToKeep = knet_IO.loadIndices(args.snpIndices)
            M = M[:,indicesToKeep]
            
        start = time.time()
        mns  = knet_IO.loadVectorFromDisk( args.mns  , 'float32')  # these are always float32 even in 64 runs
        sstd = knet_IO.loadVectorFromDisk( args.sstd , 'float32')  
        #snpIDs = knet_IO.loadsnpIDs(args.snpIDs)
        
        M[M==-1] = 0  # have to make sure that the missing genotype is NOT encoded as -1, even when we don't perform QC
        M = M.astype('float32')
        M -= mns
        M /= sstd
        end = time.time(); printElapsedTime(start,end, "standardising data via loaded params took: ")
 
    # get Zscores: have to standardise ONLY over the training, and not the training+ validation together: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
    if M_validation is not None : 
        if args.qc == 1 :
            # depending on if we are in inference mode, make sure we have the same set of SNPs
            if args.inference == 0 : M_validation = np.delete(M_validation, indicesToRemove, axis=1)
            else : M_validation = M_validation[:,indicesToKeep]
            
        #qc_data = removeList(M_validation, indicesToRemove)
        M_validation = M_validation.astype('float32')
        M_validation -= mns
        M_validation /= sstd
        print("M and M_validation are equal:", np.array_equal(M,M_validation) )
        #indices_validation = np.asarray( range(len(M_validation)) ) # is used for storting
        print("After standardising, validation data in MBs is: ",getSizeInMBs(M_validation) )
    

    # Pre-process data:
    np.random.seed(args.randomSeed)
    random.seed(args.randomSeed)

        
    # 2. create minibatch list
    numIndividuals = M.shape[0] 
    numSNPs = M.shape[1] # numSNPs = bed.get_nb_markers(), as we may have removed SNPs, we want to know how many are left
    num_y_classes = 1 # how many columns are there, IE how many classes 
    len_M = len(M)
    len_M_validation = 0
 
    # 6a. Analysis: train model
    if args.inference == 0 :
        print("Analysis Run with delta: " + str(args.delta), flush = True)

        start = time.time()
        #results = learn(model,device, args, train_GWAS, train_y, test_GWAS, test_y, eval_train=True, eval_test=True, eval_freq = args.evalFreq, decayEnabled = False)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        #M  = np.random.normal(size= (10,5 ) )
        #y = np.random.normal(size= (10 ) )
        #y = y.reshape(-1,1) # enforce 2D

        # fit Ridge-BLUP model: determine if its a n < p or n > p situation
        if numIndividuals < numSNPs : # want the 'n x n' version, as that will be faster
            print("Ridge with more p than n")
            # as XtX is non invertible as is, we HAVE to have a delta 
            BLUP = reml.computeBLUPs_RidgeBLUP_morep(y, M, args.delta)
            Beta = reml.backCalculate_Beta_BLUP(BLUP, M)   
             
        else :  # want the 'p x p' version, as there are less p than n 
            print("Ridge with more n than p")
            # as XtX IS invertible, if a delta was no specified, we fit a basic multiple-OLS
            if args.delta == -1 :
                print("No Shrinkage specified, fitting OLS multiple regression instead of ridge")
                results = regression.multiReg(y,M)
                Beta = results["beta"]
                Beta = Beta[1:Beta.shape[0]]  # OLS includes a beta for the intercept, which we don't need, so we get rid of it here
            else :
                results = reml.computeBLUPs_RidgeBLUP(y, M, args.delta)
                Beta = results.BETA 
        del M; del results; gc.collect()
        print("computed Ridge Coefs")

        writeBetasToDisk(irsIds, A1_alleles, Beta, args)
        


        # write out the SNPs that were used for the analysis
        fileName = args.out + "nn_SNPs.txt"
        with open(fileName, "w") as file: 
            for i in range( len(irsIds)  ):
                file.write(irsIds[i]  + "\n")
        
        # write out the indices of the original dataset's coordinates for convenience
        if indicesKept is not None: # in case we skipped QC
            fileName = args.out + "SNPs_indices.txt"
            with open(fileName, "w") as file: 
                for i in range( len(indicesKept)  ):
                    file.write( str(indicesKept[i])  + "\n")    

        # produce predictions for validation set     
        if len_M_validation > 0 :
            yhat = reml.predictYhat(M_validation, Beta) # make predictions
            writePRStoDisk(yhat,args )
           
    # 6. b analysis: inference we build polygenic risk scores
    else :
        print("Inference Run", flush = True)

        # load which indices to keep
        indicesKept = loadIndices(args.indices)
        M = M[:,indicesKept]
        
        # load Betas
        Beta = loadBetas(args.betas)[0]
 
        yhat = reml.predictYhat(M, Beta) # the Train set here will refer to the TEST set
        writePRStoDisk(yhat,args )

   
###############################################################################
# Helper functions
###############################################################################   
         # irsIds = ["1","2","3","4","5"]
def writeBetasToDisk(irsIds, A1_alleles, Beta, args) :
    fileName = args.out + "Beta.csv"
    with open(fileName, "w") as file:
        for i in range( len(Beta) ) :
            line = str(irsIds[i]) + " " + str(A1_alleles[i]) + " " + str(Beta[i,0]  )
            file.write( line +  "\n")   # file.write(  ( str(yhat[i])[2:-1] ).replace("  ", " ").replace(" ", "\t") +  "\n")      


def writePRStoDisk(yhat,args ) :
    # this is a 1D array, not 2D
    fileName = args.out + "yhat.txt"
    with open(fileName, "w") as file:
        file.write("Profile"  + "\n")
        for i in range( len(yhat) ) :
            line = str(yhat[i] )  
            file.write( line +  "\n")   # file.write(  ( str(yhat[i])[2:-1] ).replace("  ", " ").replace(" ", "\t") +  "\n")      
#outFile = args.out + "SNPs_indices.txt"

def loadIndices(outFile) :
    indices= list()
    with open(outFile , "r") as file:
        for i in file:
            itmp = i.rstrip().split()
            indices.append( int(itmp[0]) )

    indices = np.array(indices)
    return(indices) 


#outFile = args.out + "Beta.csv"
def loadBetas(outFile) :
    RSIds = list()
    Alleles = list()
    Betas= list()
    with open(outFile , "r") as file:
        for i in file:
            itmp = i.rstrip().split()
            RSIds.append( itmp[0] )
            Alleles.append( itmp[1] )
            Betas.append( float(itmp[2]) )

    Betas = np.array(Betas)
    Betas = Betas.reshape(-1,1) # enforce 2D
  
    return(Betas, RSIds, Alleles) 


###############################################################################
# Helper utils
###############################################################################  
    
def printElapsedTime(start,end, text ="") : # https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds), flush=True)
        
        
