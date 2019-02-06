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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_scores
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from ....application.utils.plotgen import exportNNPlot
from ....application.utils.geno_qc import removeList, genoQC_all, standardise_Genotypes, getSizeInMBs 
from ....application.logic.knet.knet_main_pytorch import weight_init, EPSILON, learn, registerDeCovHooks, setModelMode , NETWORK_DATATYPE, getNetworkDatatype_numpy, getModel, isLayerActivation
from ....io import knet_IO

# delta = (Ve/Vg)
# delta = (1-h2) / h2
#args, args.epochs, args.learnRate, args.momentum, args.evalFreq, args.savFreq, args.predictPheno, args.loadWeights, args.saveWeights, args.randomSeed, args.hidCount, args.hidl2, args.hidAct

#args = parser.parse_args(['--out', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/tests/0pytorch_tests/' ,'knet', '--knet', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/data/genetic/short', '--pheno', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/data/genetic/phenew.pheno.phen',  '--epochs', '21', '--learnRate', '0.00005', '--momentum', '0.9', '--evalFreq', '1',  '--recodecc'    , '0' ,  '--hidCount'    , '5' ,  '--hidl2'    , '1.0' ,  '--hidAct'    , '2' , '--cc', '0' ,'--inference', '0'   ]) # ,'--loadWeights', 'C:/0Datasets/NNs/genetic/weights/' ,'--snpIndices', 'C:/0Datasets/NNs/genetic/nn_SNPs_indices.txt' ,'--mns', 'C:/0Datasets/NNs/genetic/data_mns','--sstd', 'C:/0Datasets/NNs/genetic/data_sstd','--snpIDs', 'C:/0Datasets/NNs/genetic/nn_SNPs.txt'


###############################################################################
# global vars
###############################################################################
device = None

###############################################################################
# Model construction
###############################################################################
def build_model(args, numSNPs, num_y_classes, suppressPrint = False ) :
    hLayerCount = args.hidCount
    BNEnabled = int(args.bnorm) == 1
    torch.manual_seed(args.randomSeed)
    layers = []
    lastLayerSize = args.firstLayerSize
    lastOutput = numSNPs
    
    if args.convLayers > 0 :  # Conv1D definition [in_channels, out_channels, kernel_size]  # in_channels = # SNPs, out_channels = number of neurons/filters 
        if suppressPrint == False : print("Adding "+str(args.convLayers)+" conv layers, with initial input dimension: " + str(lastOutput), flush=True)

        # we add the first conv layer without maxpool, just to reduce the dimensionality
        currentNumFilters= args.convFilters
        currentStride = 3
        filter_size = 5 # as it turns out it is not actually a problem if the conv outputs something that isn't an integer, so we just need to downsample it
        layers.append( nn.Conv1d(lastOutput, currentNumFilters, filter_size, stride=currentStride, padding=0) )
        if BNEnabled : layers.append( nn.BatchNorm1d(currentNumFilters ) )
        addActivation(layers,args.hidAct)
        if args.dropout != -1 : addDropout(layers,args)

        lastOutput = int( (lastOutput - filter_size +2) / currentStride + 1 ) # as these can only be integers
        if suppressPrint == False : print("filter size : " + str(filter_size), flush=True)
        currentStride = 1
        pool_size = 2
        for i in range(1, args.convLayers +1) :
            # decide on filter size, depending on input, Conv layers must always produce even outputs so that maxpool can half them 
            filter_size = 3
            if lastOutput % 2 != 0 : filter_size = 4 # if the current output is not even, then we have to use a filter size of 4, otherwise we get fractions after the maxpool operation
            ## currentNumFilters = (i+1) * args.convFilters
            currentNumFilters = currentNumFilters // 2  
            layers.append( nn.Conv1d(lastOutput, currentNumFilters, filter_size, stride=currentStride, padding=0) ) # "same" does not work in pytorch
            if BNEnabled : layers.append( nn.BatchNorm1d(currentNumFilters ) )   
            addActivation(layers,args.hidAct)
            if args.dropout != -1 : addDropout(layers,args)
            
            currentFlatFeatures = lastOutput *filter_size # the number of features if CNN layer flattened ( need to know this before we change 'lastOutput' in the next line)
            
            lastOutput = int( (lastOutput - filter_size +2) / currentStride + 1 ) # as these can only be integers

            if suppressPrint == False : print("filter size affter Conv ("+str(i)+") : " + str(filter_size) + " / output: " + str(lastOutput), flush=True)
            
            lastOutput = (lastOutput - pool_size) / pool_size + 1 # compute what dimensions the conv+maxpool operations are going to leave for the next layer
            if suppressPrint == False : print("filter size affter Maxpool ("+str(i)+") : " + str(filter_size) + " / output: " + str(lastOutput), flush=True)
            layers.append( nn.MaxPool1d(pool_size) )
            
        layers.append( Flatten() ) # Flatten the data for input into the plain hidden layer
        lastOutput =currentFlatFeatures # find out how many columns this will have

    for i in range(1,hLayerCount+1) : # iterate 1 based, otherwise we will get a reduction after the first layer, no matter the widthReductionRate, as 0 is divisible by anything
        layers.append( nn.Linear(lastOutput, lastLayerSize) ) # torch layers are parametrized as: In , Out (IE num rows, num cols (IE the number of neurons))
        
        if BNEnabled : layers.append( nn.BatchNorm1d(lastLayerSize ) )   
        addActivation(layers,args.hidAct)
        if args.dropout != -1 : addDropout(layers,args)
        
        if suppressPrint == False : print("added layer at depth: " + str(i) + " with width: " + str(lastLayerSize))
        
        lastOutput = lastLayerSize
        # control the 'fatness' of the network: we reduce the width at a given rate: if this is 1, then at every subsequent layer, if its 2, then every 2nd layer etc
        if i % args.widthReductionRate == 0 :  lastLayerSize = lastLayerSize // 2
        if lastLayerSize < 2 : break # if 
        
    if args.inf_neurons > 0 :
        lastLayerSize = args.inf_neurons
        layers.append( nn.Linear(lastOutput, lastLayerSize) )
        lastOutput = lastLayerSize
        addActivation(layers,args.hidAct)
        if suppressPrint == False : print("added inference layer with width: " + str(lastLayerSize))
        
    if num_y_classes > 1 : 
        layers.append( nn.Linear(lastOutput, num_y_classes) )
        layers.append( nn.Softmax() )
    else :
        layers.append( nn.Linear(lastOutput, 1) )

    # initialize the model: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    model = nn.Sequential(*layers)
    model.apply(weight_init)
    
    # register hooks
    registerDeCovHooks(model, args) 
    
    return(model)
      #np.sqrt(0.5) * 0.3

def addActivation(layers, hidAct): 
    if hidAct == 1 : layers.append( nn.Sigmoid() )	
    elif hidAct == 2 : layers.append( nn.ReLU()	)
    elif hidAct == 5 : layers.append( nn.LeakyReLU(negative_slope=0.001) )	
    elif hidAct == 4 : layers.append( nn.Softplus()	)
    elif hidAct == 6 : layers.append( nn.SELU()	)
   # elif hidAct == 3 :  print("no activatioN")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

    
def addDropout(layers, args): 
    if args.hidAct == 6 : layers.append( nn.AlphaDropout(p=args.dropout)	) # use SELU for alpha dropout
    else : layers.append( nn.Dropout(p=args.dropout)	)
    
        
###############################################################################
# QC and Training
###############################################################################      
def runKnet(args) :
    print("KNeT via Pytorch backend")
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
    M = genotypeData["M"]
    irsIds = genotypeData["rsid"]
    IDs = genotypeData["IDs"] 
    IDs[0] = np.array(IDs[0]) ; IDs[1] = np.array(IDs[1])
    indicesKept = np.asarray( range(M.shape[1]) )
    
    
    del genotypeData ; gc.collect() # dont need this
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
    y = stats.zscore(y) # zscore it so that Beta -> h2 computations work    
    y = y.reshape(-1,1) # enforce 2D


    if args.oversampling : # apply oversampling logic, before data was standardised (IE it is still int8 rather than float)
        print("applying oversampling logic")
        allCaseIDs = loadStringList(args.oversampling) # load all the case IDs for the oversampling logic
        # map the case IDs to the shuffled indices

        print("Original cases list has : " + str(len(allCaseIDs)) + " samples", flush=True)
        allCaseIDs = allCaseIDs[ np.in1d(allCaseIDs, IDs[1]) ] # want to make sure not have any IDs on the 'case' list that don't exist in our data ( it is possible that the cases)
        print("Dataset has : " + str(len(allCaseIDs)) + " of these", flush=True)

        sort_idx = IDs[1].argsort()
        allCaseIndices = sort_idx[np.searchsorted(IDs[1],allCaseIDs,sorter = sort_idx)]
        allCaseIndices = np.squeeze(allCaseIndices)
           
        # remove all of these indices fro both y and M
        M_cases = M[allCaseIndices,:]
        M = np.delete(M, allCaseIndices, axis = 0) # overwrite it so that we conserve RAM
        y_cases = y[allCaseIndices]
        y = np.delete(y, allCaseIndices, axis = 0)
        IDs_0_cases = IDs[0][allCaseIndices]
        IDs_1_cases = IDs[1][allCaseIndices]
        IDs[0] = np.delete(IDs[0], allCaseIndices, axis = 0)
        IDs[1] = np.delete(IDs[1], allCaseIndices, axis = 0)
        
        # we want as many cases as controls
        caseIndices = np.asarray( range(len(M_cases)) )
        
        caseIndex_oversampled = np.random.choice(caseIndices, size = len(M), replace=True) # this creates bootstrap samples
        M_cases_oversampled = M_cases[caseIndex_oversampled,]
        y_cases_oversampled = y_cases[caseIndex_oversampled]
        IDs_0_cases_oversampled = IDs_0_cases[caseIndex_oversampled]
        IDs_1_cases_oversampled = IDs_1_cases[caseIndex_oversampled] 

        # join all the matrices back together
        M_merged = np.zeros( ( len(M) + len(M_cases_oversampled),M.shape[1]) , dtype =M.dtype) # pre allocate a matrix with the correct size
        M_merged[0:len(M),:] = M # much faster to just paste this in (column stack is 10x slower)
        M_merged[len(M):( len(M) + len(M_cases_oversampled) ),:] = M_cases_oversampled # much faster to just paste this in (column stack is 10x slower)
        M = M_merged ; del M_merged
        
        y_merged = np.zeros( ( len(y) + len(y_cases_oversampled),y.shape[1]) , dtype =y.dtype)
        y_merged[0:len(y),:] = y # much faster to just paste this in (column stack is 10x slower)
        y_merged[len(y):( len(y) + len(y_cases_oversampled) ),:] = y_cases_oversampled # much faster to just paste this in (column stack is 10x slower)
        y = y_merged ; del y_merged
        
        IDs_0_merged = np.zeros( ( len(IDs[0]) + len(IDs_0_cases_oversampled)) , dtype =IDs[0].dtype)
        IDs_0_merged[0:len(IDs[0])] = IDs[0] # much faster to just paste this in (column stack is 10x slower)
        IDs_0_merged[len(IDs[0]):( len(IDs[0]) + len(IDs_0_cases_oversampled) )] = IDs_0_cases_oversampled # much faster to just paste this in (column stack is 10x slower)
        IDs[0] = IDs_0_merged ; del IDs_0_merged
     
        IDs_1_merged = np.zeros( ( len(IDs[1]) + len(IDs_1_cases_oversampled)) , dtype =IDs[1].dtype)
        IDs_1_merged[0:len(IDs[1])] = IDs[1] # much faster to just paste this in (column stack is 10x slower)
        IDs_1_merged[len(IDs[1]):( len(IDs[1]) + len(IDs_1_cases_oversampled) )] = IDs_1_cases_oversampled # much faster to just paste this in (column stack is 10x slower)
        IDs[1] = IDs_1_merged ; del IDs_1_merged   
        del IDs_0_cases_oversampled; del IDs_1_cases_oversampled; del M_cases_oversampled; del y_cases_oversampled; del M_cases; del y_cases; del IDs_0_cases; del IDs_1_cases;
        


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
    # Shuffle data before producing the minibatches to avoid having all-case or all-control minibatches
    
    np.random.seed(args.randomSeed)
    random.seed(args.randomSeed)
    if args.inference == 0 : # only shuffle order for analysis runs, as for inference we want to keep the original
        start = time.time()
  
        indices = np.asarray( range(len(M)) ) # is used for storting
        random.shuffle(indices)
        M = M[indices]
        y = y[indices]
        IDs[0] = IDs[0][indices]
        IDs[1] = IDs[1][indices] # this stores the IIDs as strings in the order the data is in the M matrix
        
    
        end = time.time(); printElapsedTime(start,end, "shuffling data took: ")
        
    # 2. create minibatch list
    numIndividuals = M.shape[0] 
    numSNPs = M.shape[1] # numSNPs = bed.get_nb_markers(), as we may have removed SNPs, we want to know how many are left
    num_y_classes = 1 # how many columns are there, IE how many classes 
    len_M = len(M)
    len_M_validation = 0
 
    # reshape data to be the right dimensions for Convolutions
    if args.convLayers > 0 :
        #input_shape = [numSNPs,1,]  # for conv layers it is, [num_indis,in_channels/num_SNPs, num_neurons/out_channels,num_filters]
        M = M.reshape(M.shape[0], M.shape[1], 1)  # M = M.reshape(M.shape[0], 1 , 1, M.shape[1])
        if M_validation is not None :
            M_validation = M_validation.reshape(M_validation.shape[0],M_validation.shape[1],  1)  ## M_validation.reshape(M_validation.shape[0], 1 , 1, M_validation.shape[1]) 

    #else : input_shape =  [numSNPs] # for linear layers it is [num_indis,num_SNPs]

    minibatch_size =  args.batch_size #M.shape[0]  # 64   #minibatch_size = 128
    if args.batch_size == 0 : minibatch_size = len(M)
    num_batches = len(M) // minibatch_size

    # 3. create minibatches (cant do this in a function, as that would make it unavoidable to use 2x the RAM)
    startTime = time.time()
    start = 0
    end = minibatch_size
    train_GWAS = list()
    train_y = list()
    minibatch_size =  args.batch_size #M.shape[0]  # 64
    if args.batch_size == 0 : minibatch_size = len(M)
    num_batches = len(M) // minibatch_size 
    
    # oversampling logic: ensure that each minibatch has 50% cases, even when dataset is unbalanced
    # need to map the cases' indices to the M/y matrices... use this to find the indices...

    # remove all cases from the total and separate into a separate array
    
    # pick 50% minibatch size controls as normal, RANDOMLY pick 50% minibatch of the cases, and join these together (both X/Y)
    # (this results in increased RAM usage, IE with ~1% cases this would equal to ~2x the RAM)
    # will need to see how the PRS / remainder bathc logic is affected
    # oversampling is only enabled for Analysis runs and only for the training set (IE the 'remainderBatch' for the training set will never be used)
    
    y_batched = y.copy()
    for i in range(num_batches) :  # # do this in a more RAM efficient way: keep deleting the bits from the original matrix to free up space as we go along otherwise this step would double the RAM requirements temporarily
        train_GWAS.append(M[0:minibatch_size]  )
        M = M[minibatch_size:len(M)]
        
        train_y.append(y_batched[0:minibatch_size])
        y_batched = y_batched[minibatch_size:len(y_batched)]
        print("adding batch " + str(i)  + ", minibatch size: " + str(minibatch_size) + " / num left in pool: " + str(len(M))  )
        gc.collect()

    #print("train_GWAS[0].shape: " + str( train_GWAS[0].shape)  + " // train_y.shape: " + str( train_y[0].shape) )

    remainderBatch = M
    remainderBatch_valid = None
    #if args.inference == 1 : remainderBatch = M
    # we keep the remainder of the 'M' matrix for the inference runs, as then the 'train' set is the test set...
    #del M; gc.collect() # free up memory
        
    if M_validation is not None : 
        len_M_validation = len(M_validation) 
        if args.batch_size == 0 : minibatch_size = len(M_validation)
        
        test_GWAS = list()
        test_y = list()
        num_batches = len(M_validation) // minibatch_size
        print("len_M_validation is: " + str(len_M_validation)  + ", minibatch size: " + str(minibatch_size)  + " args.batch_size: " + str(args.batch_size) + " num_batches is: " + str(num_batches))
        start = 0
        end = minibatch_size
        for i in range(num_batches) :
            #test_GWAS.append(M_validation[start:end]  )
            test_GWAS.append(M_validation[0:minibatch_size]  )
            M_validation = M_validation[minibatch_size:len(M_validation)]
 
            test_y.append(y_validation[start:end])
            print("adding batch " + str(i)  + " , start/end: " + str(start) + "/" + str(end)  )
            start = end
            end += minibatch_size  
        remainderBatch_valid = M_validation
        #print("First minibatch is: ", test_GWAS[0][0,])
        # del M_validation; gc.collect() # free up memory, cant do this as we need this for the PRS calculation....
    else :
        test_GWAS = None 
        test_y = None
    end = time.time(); printElapsedTime(startTime,end, "creating minibatches took: ")    
    #print("Last minibatch is: ", train_GWAS[-1][0,])
    #print("Last minibatch As Valid is: ", test_GWAS[-1][0,])   

    # scale the delta by minibatch_size, if we dont have minibatches
    #ratio = float(minibatch_size) / numIndividuals # this is 1 if there are no minibatches
    #print("orig L2 Regularizer : " + str(args.hidl2) + " minibatches scaled to " + str(hiddenShrinkage * ratio) )
    #hiddenShrinkage *= ratio

    # setup device
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu ==1 else "cpu")

    # 4a: if hyoperopt was enabled then determine the best hyperparam settings automatically
    if args.hyperopt == 1 :
        print("determining best params via hyperopt")
        start = time.time()
        best_pars = optimize_model_pytorch(device, args, train_GWAS, train_y, test_GWAS, test_y, out_folder = args.out +"hyperopt/", startupJobs = 5, maxevals = 20)    
        writeKNeT_bestPars(args.out ,best_pars)    
        #best_pars = loadKNeT_bestPars(args.out)
        args = mergeArgsAndParams(args,best_pars)
        end = time.time(); printElapsedTime(startTime,end, "hyperopt took: ")
        
    # 4b. create model 
    model = build_model(args, numSNPs, num_y_classes) # Don't pre-allocate memory; allocate as-needed ??
    # data parallelism: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    if torch.cuda.device_count() > 1 and args.inference == 0 and args.gpu ==1 : # do NOT use dataparallel GPU for inference runs as the hooks don't work: # these may not work on DataParallel models ??: https://pytorch.org/docs/0.3.1/nn.html?highlight=hook#dataparallel-layers-multi-gpu-distributed
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      print("WARNING: multi-GPUs don't work reliably with hooks, so DeCov regularizer will not be applied") # see: https://discuss.pytorch.org/t/aggregating-the-results-of-forward-backward-hook-on-nn-dataparallel-multi-gpu/28981/3   # https://discuss.pytorch.org/t/weird-output-of-forward-hook-when-use-multi-gpus/29437/2
      model = nn.DataParallel(model)
    model.to(device)

    # 6a. Analysis: train model
    if args.inference == 0 :
        print("Analysis Run", flush = True)

        start = time.time()
        results = learn(model,device, args, train_GWAS, train_y, test_GWAS, test_y, eval_train=True, eval_test=True, eval_freq = args.evalFreq, decayEnabled = False)
        end = time.time(); printElapsedTime(start,end, "training model took: ")
        results_its = results["results"]  
      
        if args.earlystop == 1 :  # if Early stop was requested, we retrain model up until model with highest validation accuracy
            print("EARLY STOP?")
            highestAcc_epoch=  results['results']['highestAcc_epoch'] + 1 # these are 0 based
            if highestAcc_epoch != args.epochs : # but only if it wasn't already the highest accuracy
                print("Re-training to early stop at epoch: " + str(highestAcc_epoch)  + " as highest accuracy there was: " + str(results['results']['highestAcc']), flush = True)
                args.epochs = highestAcc_epoch
                model = build_model(args, numSNPs, num_y_classes) # Don't pre-allocate memory; allocate as-needed ??
                if torch.cuda.device_count() > 1 and args.inference == 0 and args.gpu ==1 : model = nn.DataParallel(model) # do NOT use dataparallel GPU for inference runs as the hooks don't work: # these may not work on DataParallel models ??: https://pytorch.org/docs/0.3.1/nn.html?highlight=hook#dataparallel-layers-multi-gpu-distributed 
                model.to(device)
                start = time.time()
                results = learn(model,device, args, train_GWAS, train_y, test_GWAS, test_y, eval_train=True, eval_test=True, eval_freq = args.evalFreq, decayEnabled = False)
                end = time.time(); printElapsedTime(start,end, "training model took: ")
                results_its = results["results"]  
        with open( args.out + "epochs_used.txt", "w") as file: file.write("epochs_used=" + str(args.epochs) ) # write out the early stop epoch used
        
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        
        # 5. Save model params
        if args.saveWeights is not None : torch.save(getModel(model).state_dict(), args.saveWeights)
  
        #write training data means / stds to disk so that we could use those for inference runs later
        print("writing means/stds to disk with datatype: "  + str(sstd.dtype))
        print("sstd shape is: " + str(sstd.shape) + " / mns shape: " + str(mns.shape))
        
        knet_IO.writeVectorToDisk( args.out + "data_mns" , mns, mns.dtype)  
        knet_IO.writeVectorToDisk( args.out + "data_sstd" , sstd, sstd.dtype)  
       
        if test_GWAS is not None :
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
                exportNNPlot(results_its, args.out + "nnplot")
  
        # write out the SNPs that were used for the analysis
        fileName = args.out + "nn_SNPs.txt"
        with open(fileName, "w") as file: 
            for i in range( len(irsIds)  ):
                file.write(irsIds[i]  + "\n")
        
        # write out the indices of the original dataset's coordinates for convenience
        if indicesKept is not None: # in case we skipped QC
            fileName = args.out + "nn_SNPs_indices.txt"
            with open(fileName, "w") as file: 
                for i in range( len(indicesKept)  ):
                    file.write( str(indicesKept[i])  + "\n")    
             
        # produce predictions for validation set     
        if len_M_validation > 0 :
            producePRS(model,args,remainderBatch_valid, test_GWAS, IDs_validation, len_M_validation , args.out + "yhat.txt", args.out + "FIDs.txt", y_validation, args.out + "KNET_PRS")         
            
#            model2 = build_model(args, numSNPs, num_y_classes) # Don't pre-allocate memory; allocate as-needed ??
##            # data parallelism: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
#            if torch.cuda.device_count() > 1 and args.inference == 0 and args.gpu ==1 : # do NOT use dataparallel GPU for inference runs as the hooks don't work: # these may not work on DataParallel models ??: https://pytorch.org/docs/0.3.1/nn.html?highlight=hook#dataparallel-layers-multi-gpu-distributed
#              print("Let's use", torch.cuda.device_count(), "GPUs!")
#              print("WARNING: multi-GPUs don't work reliably with hooks, so DeCov regularizer will not be applied") # see: https://discuss.pytorch.org/t/aggregating-the-results-of-forward-backward-hook-on-nn-dataparallel-multi-gpu/28981/3   # https://discuss.pytorch.org/t/weird-output-of-forward-hook-when-use-multi-gpus/29437/2
#              model2 = nn.DataParallel(model2)
#            model2.to(device)
#    
#            print("Inference Run dummy", flush = True)
#            model2.load_state_dict(torch.load(args.saveWeights))
#            producePRS(model2,args,remainderBatch, test_GWAS, IDs_validation, len_M_validation , args.out + "yhat_TEST_DUMMY.txt", args.out + "FIDs_TEST_DUMMY.txt", y_validation, args.out + "KNET_PRS_TEST_DUMMY")

    # 6. b analysis: inference we build polygenic risk scores
    else :
        print("Inference Run", flush = True)
        model.load_state_dict(torch.load(args.loadWeights))
        profileName= "yhat_TEST.txt"
        rSQoutName ="KNET_PRS_TEST"
        if args.linearInference == 1 : 
            model = getInferenceModelWithoutActivation(model)
            profileName = "yhat_TEST_noAct.txt" 
            rSQoutName ="KNET_PRS_TEST_noAct"
        
        
        # the Train set here will refer to the TEST set
        producePRS(model,args,remainderBatch, train_GWAS, IDs, len_M , args.out + profileName, args.out + "FIDs_TEST.txt", y, args.out + rSQoutName)

   
###############################################################################
# Helper functions
###############################################################################
def loadStringList(outFile) :  # used by the oversampling logic to load in a list of cases
    items = list()
    with open(outFile, "r") as file: 
        for i in file:
            itmp = i.rstrip().split()
            items.append(itmp)
            
    items = np.array(items)

    return( items)   
        
def getInferenceModelWithoutActivation(model) : # produces an identical model, but without the activation layers (IE to get a linear predictor)
    print("switching off activations for linear infernece")
    model = getModel(model)
    origLayers = list(model)
    subsetModelLayers = list()
    for i in range(len(origLayers)) :
        if isLayerActivation(origLayers[i]) == False or i == ( len(origLayers) -1 ) : subsetModelLayers.append(origLayers[i]) # we add the last layer, even if that is activation, as that is needed to get the right shaped output
    modelInference = nn.Sequential(*subsetModelLayers) # create a subset model of onl y
    #modelInference.eval() # otherwise dropout and other layers would act up

    return(modelInference)
    
def producePRS(model,args,remainderBatch, miniBatches, IndiIDs, len_total , outLoc_yhat, outLoc_FIDs, ytrue, outLoc_PRS) :
    model_training_orig = model.training # otherwise dropout and other layers would act up
    setModelMode(model, False)
    global device
    # write final predictions out
    yhats = list()
    totalSofar= 0
    with torch.no_grad(): # reduce memory usage and speed up computations but you won’t be able to backprop; source: https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
        for i in range(len(miniBatches)) : # loop through all minbatches
            totalSofar += len(miniBatches[i])
            b_data = torch.from_numpy(miniBatches[i]).to(device)
            yhats.append( model(b_data).detach().cpu().numpy() )
       
        if totalSofar < len_total :
            print("minibatches did not cover all training samples, so we create last batch out of the remainders")
            #lastBatch_X = origData[totalSofar:len_total]
            b_data = torch.from_numpy(remainderBatch).to(device)
            yhats.append( model(b_data).detach().cpu().numpy() )
    
    yhat_all = np.concatenate(yhats)
    print("after merging, we have yhat predictions for : " + str(len(yhat_all)) + " samples", flush=True)
    setModelMode(model, model_training_orig)  # reset model into traning mode

    fileName = outLoc_yhat
    with open(fileName, "w") as file:
        file.write("Profile"  + "\n")
        
        for i in range(yhat_all.shape[0]) :
            line = str(yhat_all[i][0] )
            for j in range(1, len(yhat_all[i]) ):
                line = line + "\t" + str(yhat_all[i][j] )              
            file.write( line +  "\n")   # file.write(  ( str(yhat[i])[2:-1] ).replace("  ", " ").replace(" ", "\t") +  "\n")

    # also write out the FID / IIDs in the same order, just as a sanity check (compare this against the .fam files)
    fileName = outLoc_FIDs
    with open(fileName, "w") as file:
        file.write("FID" + "\t" + "IID" + "\n")

        for i in range( len(IndiIDs[0]) ) :
            line = IndiIDs[0][i] + "\t" + IndiIDs[1][i]
            file.write( line +  "\n") 
            
    # write out the final r^2
    yhat_all += EPSILON # for numerical stability
    rSQ = np.corrcoef( ytrue, yhat_all, rowvar=0)[1,0]**2      
       
    with open(outLoc_PRS, "w") as file: 
            file.write(str(rSQ) ) 

   
###############################################################################
# Inference plotting
###############################################################################
targetLayerActivations = None

def get_targetLayerActivations():
    
    def hook(model, input, output):
        global targetLayerActivations
        #if model.training : # ONLY add this if the model is in training mode, bad idea, as we only 
        targetLayerActivations = output  # output.detach()

    return hook

def produceActivation (model, device,targetLayerIndex,artificial_SNP_data):

    artificial_SNP_data = torch.from_numpy(artificial_SNP_data).to(device)
     
    origLayers = list(model)
    # targetLayerIndex = findPrevNextActivationLayer(model,targetLayerIndex, startIndexOK = True) # dont do this, we should assume that the correct layer was already chosen outside
    if targetLayerIndex < 0 : targetLayerIndex = len(model) +targetLayerIndex
    lastLayerSlice = targetLayerIndex +1 # , +1 as this is a slice, that is EXCLUSIVE, IE 0:len
    subsetModelLayers = origLayers[0:lastLayerSlice]
    modelInference = nn.Sequential(*subsetModelLayers) # create a subset model of onl y
    modelInference.eval() # otherwise dropout and other layers would act up
    
    # setup hook for FP, to capture intermediate activation
    activationHook = modelInference[targetLayerIndex].register_forward_hook(get_targetLayerActivations()) 
    
    modelInference(artificial_SNP_data) # FP the data
    
    global targetLayerActivations
 
    targetLayerActivations = targetLayerActivations.detach().cpu().numpy() # obtain the activation, this would have been set via the hook

    # remove hook
    activationHook.remove()         
    return(targetLayerActivations)
    
    
# targetLayerIndex has to be CORRECT
def produceActivationPlot(model,device,interactionsToTest,numNeurons, targetLayerIndex, totalNumSNPs, plotFolder_used, strength = 1, outFileStem = "true", normalized = None , scale = [8,8], rSQ = -1, doAll = False, subtractNullActivations = True ) :
    offset = 1
    if doAll == False : offset = 0
    activationMap = np.zeros( (len(interactionsToTest) +offset,numNeurons) ) # the activation map will have 1 row for each SNP, (AND +1 for when all are active), and one column for each neuron
    SNPlabels = list()
    neuronlabels = list()
    neuronlabels = list(range(numNeurons)) 
    neuronlabels =[x+1 for x in neuronlabels]
    #nullActivations.shape
    nullActivations = produceActivation(model,device,targetLayerIndex, np.zeros( (1,totalNumSNPs) , dtype = getNetworkDatatype_numpy() ))
    for i in range(len(interactionsToTest) +offset) :
        # produce an artifial person's SNP data with only the proposed interactions having values
        artificial_SNP_data =  np.zeros( (1,totalNumSNPs) , dtype = getNetworkDatatype_numpy())
        if i == len(interactionsToTest) : # if it is the last item, IE when all are interactions are active
            SNPlabels.append("All")
            for j in range(len(interactionsToTest) ) :
                SNP_set =  interactionsToTest[j]
                artificial_SNP_data[:,SNP_set] += strength ## add the SNPs in at each location
        else :
            SNP_set =  interactionsToTest[(i-1)] # all_possible_interactions[i]
            SNPlabels.append( np.array2string(SNP_set) )

        artificial_SNP_data[:,SNP_set] += strength

        targetLayerActivations = produceActivation(model,device,targetLayerIndex,artificial_SNP_data)
        if subtractNullActivations: targetLayerActivations -= nullActivations
        

        activationMap[i,:] = targetLayerActivations
           
    activationMap = np.abs(activationMap)
    # plot heatmap on grid: https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib   and    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    os.makedirs(os.path.dirname(plotFolder_used), exist_ok=True)  
    fig_file = plotFolder_used +outFileStem

    fig, ax = plt.subplots(figsize=scale)
    if normalized is not None : ax.imshow(activationMap, cmap='gray', interpolation='nearest', vmin=normalized[0], vmax=normalized[1])
    else : ax.imshow(activationMap, cmap='gray', interpolation='nearest')
    plt.xlabel("Neurons")
    plt.ylabel("Interaction candidates")
    #plt.figure(figsize=(12,16))
    # ... and label them with the respective list entries
    ax.set_yticks(np.arange(len(SNPlabels)))
    ax.set_yticklabels(SNPlabels)
    ax.set_xticks(np.arange(len(neuronlabels)))
    ax.set_xticklabels(neuronlabels)

    titleTest = "Neuron activations for SNPs: " + outFileStem 
    if rSQ != -1 : titleTest += "( r^2: " + str( round(rSQ,2) ) + ")"
    ax.set_title(titleTest)
    
    fig.tight_layout()
    
    plt.show()
    fig.savefig(fig_file)  
    #fig_file = fig_file + "2"
    minValue = np.min(activationMap)
    maxValue = np.max(activationMap)
    normalized = [minValue, maxValue]
    return( normalized, activationMap )

    
###############################################################################
# Hyperopt
###############################################################################
minLayers=2
minNeurons=100
maxEpochs=100
def parameter_space_pytorch():
	global minLayers
	global minNeurons
	global maxEpochs
	# Set up a list for the parameter search space.
	space = {
	# the number of LSTM neurons
	'firstLayerSize': set_int_range('firstLayerSize', minNeurons,5000), #  +1 as otherwise Upper bond is exclusive, IE a range of 5-21 would only cover 5-20
	'epochs': set_int_range('epochs', 10,maxEpochs ), #  +1 as otherwise Upper bond is exclusive, IE a range of 5-21 would only cover 5-20  
	'hidCount': set_int_range('hidCount', minLayers,11), #  +1 as otherwise Upper bond is exclusive, IE a range of 5-21 would only cover 5-20
	'dropout': hp.uniform('dropout', 0.0, 0.9), 
	'learnRate': hp.uniform('learnRate', 0.000001, 0.01) #,  # loguniform
	#'optimizer': hp.choice('optimizer',[0,1, 2]), # which optimizer to use, 0 SGD, 1 ADAM and 2 AMSGrad   
	#  'relu': hp.choice('relu',[0,1]),
   # 'BNEnabled': hp.choice('BNEnabled',[0,1]),
   # 'l2': hp.uniform('l2', 0.0, 20.0), # corresponds to h2 choice between 100% (delta = 0) to 5% (delta = 20)
   # 'orthoReg': hp.uniform('orthoReg', 0.1, 1.0)  # loguniform   # 0.0001, 0.1
	}
	return space


def set_int_range(name, myMin, myMax): # Set up a parameter range based on the given min and max values.
	# If the myMin and myMax values are equal, don't search over this parameter.
	if(myMin == myMax):
		return myMin

	# Swap the values so they are in the correct order if necessary.
	if(myMin > myMax):
		t=myMax
		myMax=myMin
		myMin=t
		
	# Randomly search over all integer values between the myMin and myMax
	return hp.choice(name, np.arange(myMin,myMax, dtype=int))
    

def plot_optimization_pytorch(trials, parameter, regression = False, out_folder= ""):
    os.makedirs(os.path.dirname(out_folder), exist_ok=True) 
    # Create the base figure and axes.
    fig = plt.figure()
    ax = plt.subplot(111)

    # Pull out the trial data for the requested parameter.
    xs = [t['misc']['vals'][parameter] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]

    if parameter == 'hidCount':  # really hacky solution to start not at 0, but at the proper minimum
        #print("plotting layers, so we offset it")
        global minLayers
        for i in range(len(xs) ) :
            xs[i] = [s + minLayers for s in xs[i]] # element wise offset each value by the minimum value 
   
    if parameter == 'firstLayerSize':  # really hacky solution to start not at 0, but at the proper minimum
        #print("plotting layers, so we offset it")
        global minNeurons
        for i in range(len(xs) ) :
            xs[i] = [s + minNeurons for s in xs[i]] # element wise offset each value by the minimum value 

    # for nested/conditional params we need to remove values where parameter was disabled
    indicesToRemove = list()
    for i in range(len(xs) ) :
        if len(xs[i]) == 0 : 
            indicesToRemove.append(i) 
			#print("removing index ", i)

    # xs_2 = list( numpy.delete(xs, indicesToRemove) ) # this somehow 'flattens' nested lists ie [[0], [0]] becomes [0 , 0] NOT GOOD!
    xs = [i for j, i in enumerate(xs) if j not in indicesToRemove]
    ys = [i for j, i in enumerate(ys) if j not in indicesToRemove]
   
    IDFileName=str(os.getcwd())+"/"+parameter+"_results.txt"
    with open(IDFileName, "w") as idFile: 
        idFile.write(parameter + "\t" + "response" + "\n")
        for i in range( len(xs) ):
            idFile.write( str(xs[i][0]) + "\t" + str(ys[i]) + "\n")

	# Draw the plot.
    ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5)
    ax.set_xlabel(parameter, fontsize=12)
    ylabel ='AUC'
    if regression : ylabel = 'r^2'
    ax.set_ylabel(ylabel, fontsize=12)

    # Save the figure to file.
    fig_file=out_folder+"/"+parameter+"_optimisation.png"
    print("fig_file is: " + str(fig_file))
    fig.savefig(fig_file)


def mergeArgsAndParams(args,params) :
    argsMerged = copy.deepcopy(args) # createa deep copy of the original args object
    argsMerged = vars(argsMerged) # convert it to dictionary, so that we can easily copy the key/value pairs

    # go through the keys params and overwrite the corresponding entry in the args  (the argnames must match)
    for key, value in params.items():
        argsMerged[key] = value

    #argsMerged = { 'no_cuda': False, 'batch_size': 64, 'test_batch_size': 1000, 'epochs': 10, 'lr':0.01, 'momentum': 0.5, 'seed': 1, 'log_interval': 10 }
    argsMerged = SimpleNamespace(**argsMerged) # convert back to namespace, as that is what the build_model expects
    return(argsMerged)   
#params = {'epochs': 43, 'layers': 9, 'learnRate': 0.007538107949269826, 'neurons': 733, 'orthoReg': 0.9169236124790001, 'p_dropout': 0.7599124435352762}
     
numTrials_pytorch = 0
def trial_pytorch(params,device, args, train_GWAS, train_y, test_GWAS, test_y):
    global supressOutput
    global numTrials_pytorch
    numTrials_pytorch += 1

    # create model 
    argsMerged = mergeArgsAndParams(args,params) # produce a unified args object that has the 'on trial' parameters from hyperopt
    model = build_model(argsMerged, train_GWAS[0].shape[1], train_y[0].shape[1],suppressPrint = True) # Don't pre-allocate memory; allocate as-needed ??
    if torch.cuda.device_count() > 1 and args.inference == 0 : # do NOT use dataparallel GPU for inference runs as the hooks don't work: # these may not work on DataParallel models ??: https://pytorch.org/docs/0.3.1/nn.html?highlight=hook#dataparallel-layers-multi-gpu-distributed
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      print("WARNING: multi-GPUs don't work reliably with hooks, so DeCov regularizer will not be applied") # see: https://discuss.pytorch.org/t/aggregating-the-results-of-forward-backward-hook-on-nn-dataparallel-multi-gpu/28981/3   # https://discuss.pytorch.org/t/weird-output-of-forward-hook-when-use-multi-gpus/29437/2
      model = nn.DataParallel(model) 
    model.to(device)
    
    # train model
    results = learn(model,device, argsMerged, train_GWAS, train_y, test_GWAS, test_y, eval_train=True, eval_test=True, eval_freq = args.evalFreq, decayEnabled = False, suppressPrint = True)        
    acc = results['results']['test_accuracy'][-1]
    highestAcc=  results['results']['highestAcc']
    highestAcc_epoch=  results['results']['highestAcc_epoch']

    if supressOutput == False: print('Trial ',str(numTrials_pytorch),' with parameters: ', str(params), " final r^2: ", str(acc), "but highest r^2 was : " , str(highestAcc), " at epoch: " , str(highestAcc_epoch) )
    # params['epochs'] = int(highestAcc_epoch) # overwrite this , this does NOT change the record in the trials.trails object

    if np.isnan(acc) : 
        if supressOutput == False: print("loss is nan, set it to 0 ")
        acc = 0
        
    attachments = {'highestAcc':highestAcc, 'highestAcc_epoch': highestAcc_epoch} 

    # Return the statistics for the best model in this trial.
    return {'loss': -acc, 'status': STATUS_OK, 'attachments': attachments} # loss = error in yhat.... higher the error, the LOWER the accuracy, so if we invert accuracy, then the loss will be lower... and as accuracy/correlation are both better if higher this is still OK


supressOutput = False
def optimize_model_pytorch(device, args, train_GWAS, train_y, test_GWAS, test_y, out_folder ="", startupJobs = 40, maxevals = 200, noOut = False):
    global numTrials_pytorch
    numTrials_pytorch= 0

    trials = Trials()
    trial_wrapper = partial(trial_pytorch,device = device, args = args , train_GWAS = train_GWAS, train_y = train_y , test_GWAS = test_GWAS , test_y = test_y)

    best_pars = fmin(trial_wrapper, parameter_space_pytorch(), algo=partial(tpe.suggest, n_startup_jobs=(startupJobs) ), max_evals=maxevals, trials=trials)

    # Print the selected 'best' hyperparameters.
    if noOut == False: print('\nBest hyperparameter settings: ',space_eval(parameter_space_pytorch(), best_pars),'\n')

    # loops through the 1st entry in the dict that holds all the lookup keys
    regression = True

    for p in trials.trials[0]['misc']['idxs']: plot_optimization_pytorch(trials, p, regression, out_folder = out_folder) 

    best_pars = space_eval(parameter_space_pytorch(), best_pars) # this turns the indices into the actual params into the valid aprameter space
    
    # override the epochs with the early start
    lowestLossIndex = np.argmin(trials.losses())
    trials.trial_attachments(trials.trials[lowestLossIndex])['highestAcc_epoch']
    best_pars['earlyStopEpochs'] = trials.trial_attachments(trials.trials[lowestLossIndex])['highestAcc_epoch']
    best_pars['earlyStopEpochs'] += 1 # as epochs are 0 based otherwise...
    best_pars['epochs'] = best_pars['earlyStopEpochs'] 
    if best_pars['epochs'] <= 0 : best_pars['epochs'] = 1 # we dont want a network without any training, as that will cause a problem for deep dreaming
    return(best_pars)


def writeKNeT_bestPars(outFile, best_pars ) :
    with open(outFile + ".best_pars", "w") as file: 
            file.write(str(best_pars) ) 


def loadKNeT_bestPars(outFile) :
    s=""
    with open(outFile + ".best_pars", "r") as file: 
        for i in file:
            s+=i
    best_pars= literal_eval(s)        
    return(best_pars) 


###############################################################################
# Helper utils
###############################################################################  
    
def printElapsedTime(start,end, text ="") : # https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds), flush=True)
        










