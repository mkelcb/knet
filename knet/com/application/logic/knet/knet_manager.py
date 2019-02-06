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

from ....application.utils import plotgen
from ....application.utils import geno_qc
from ....application.logic.knet import knet_main
from ....io import knet_IO
import gc
import numpy as np
from numpy.linalg import norm 
from scipy import stats
from pathlib import Path
import random
import os

lastLayerSize_MAX = int(1000) # int(4096 /2)

# delta = (Ve/Vg)
# delta = (1-h2) / h2
#args, args.epochs, args.learnRate, args.momentum, args.evalFreq, args.savFreq, args.predictPheno, args.loadWeights, args.saveWeights, args.randomSeed, args.hidCount, args.hidl2, args.hidAct
#  V(G)    0.168545        0.004763
#V(e)    0.006826        0.002168

  
def addActivation(myNet, hidAct): 
    if hidAct == 1 :  H_Act = knet_main.knSigmoid( myNet)
    elif hidAct == 2 :  H_Act = knet_main.knRELU( myNet)
    elif hidAct == 3 :  print("no activatioN")
    elif hidAct == 5 :  H_Act = knet_main.knLeakyRELU( myNet)
    else : H_Act = knet_main.knSoftplus( myNet)
    
            
def getNetworkStructure(myNet) :

    layernum = 0
    
    for layerIndex in range(0,len(myNet.layers)) :
        layer = myNet.layers[layerIndex]

        if type(layer) == knet_main.knnLayer: # for non input types, we have

            if layer.Weights_W is not None :
                layernum += 1
                print("layer " + str(layernum) + " has weight matrix shaped: " + str(layer.Weights_W.shape))
                

    
def runKnet(args) :
    hLayerCount = args.hidCount
    hiddenShrinkage = args.hidl2

 
    # default QC settings used for all non AMBLUP versions
    _minObserved = 0.95
    _minMAF = 0.01
    _minVariance = 0.02
    
    # load plink binary / phenotypes want to load them here, so that we can release the memory once the raw data is no longer used
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : recodecc = False

    
    genotypeData = knet_IO.loadPLINK(args.knet, loadPhenos = False) 
    M = genotypeData["M"]
    irsIds = genotypeData["rsid"]
    IDs = genotypeData["IDs"] 
    indicesKept = np.asarray( range(M.shape[1]) )
    
    del genotypeData ; gc.collect() # dont need this
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
    y = stats.zscore(y) # zscore it so that Beta -> h2 computations work
 
    # if we have a validation set
    M_validation = None
    y_validation = None
    if args.validSet :
        genotypeData = knet_IO.loadPLINK(args.validSet, loadPhenos = False, replaceMissing = True) # want to replace -1 with 0s, as we otherwise would have -1s, as later we just delete indices that failed QC for the training set, but won't care for individual missing datas
        M_validation = genotypeData["M"] 
        IDs_validation = genotypeData["IDs"] 
        print("Loaded number of people for validatin: ", len(M_validation), flush=True )
        del genotypeData ; gc.collect() # dont need this
        
        if args.validPhen :
            y_validation = knet_IO.loadPLINKPheno(args.validPhen, caseControl = cc, recodeCaseControl = recodecc) 
            y_validation = stats.zscore(y_validation) # zscore it so that Beta -> h2 computations work

 
    if args.inference == 0 : 
        # 1. standardise data
        if args.qc == 1 :
            qc_data = geno_qc.genoQC_all(M, rsIds = irsIds, minObserved = _minObserved, minMAF = _minMAF, minVariance = _minVariance) # we MUST perform QC with the EXACT SAME settings as the 'region scanner' otherwise the region coordinates will be mismatched
            #M = qc_data["X"] 
            rsIds_qc = qc_data["rsIds"] # save away the surviving SNP list that we have used 
            indicesToRemove = qc_data["indicesToRemove"]
            indicesKept = qc_data["indicesKept"]
            irsIds = rsIds_qc.tolist()
            
            del qc_data; gc.collect() # overwrite
            
            qc_data = geno_qc.removeList(M, indicesToRemove)
            M = qc_data["X"]
            del qc_data; gc.collect() # overwrite
        else : print("Skipping internal QC", flush=True)  
        
        M, mns, sstd = geno_qc.standardise_Genotypes(M) ; gc.collect()
        print("After standardising, training data in MBs is: ",geno_qc.getSizeInMBs(M) )

    else :
        print("Inference data QC", flush=True)
        if args.snpIndices is not None :
            indicesToKeep = knet_IO.loadIndices(args.snpIndices)
            M = M[:,indicesToKeep]
            
        mns  = knet_IO.loadVectorFromDisk( args.mns  , 'float32')  # these are always float32 even in 64 runs
        sstd = knet_IO.loadVectorFromDisk( args.sstd , 'float32')  
        snpIDs = knet_IO.loadsnpIDs(args.snpIDs)
        
        

        M = M.astype('float32')
        M -= mns
        M /= sstd
        # load final list of RSids
        # load mean /SDs
    
    #M = geno_qc.standardise_Genotypes(M) ; gc.collect()
    #print("After standardising, training data in MBs is: ",geno_qc.getSizeInMBs(M) )
    

    # get Zscores: have to standardise ONLY over the training, and not the training+ validation together: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
        # will have to implement this for genetic data
    if M_validation is not None : 
        if args.qc == 1 :
            # depending on if we are in inference mode, make sure we have the same set of SNPs
            if args.inference == 0 : M_validation = np.delete(M_validation, indicesToRemove, axis=1)
            else : M_validation = M_validation[:,indicesToKeep]
            
        #qc_data = geno_qc.removeList(M_validation, indicesToRemove)
        M_validation = M_validation.astype('float32')
        M_validation -= mns
        M_validation /= sstd
        indices_validation = np.asarray( range(len(M_validation)) ) # is used for storting
        print("After standardising, validation data in MBs is: ",geno_qc.getSizeInMBs(M_validation) )
        

    # Pre-process data:
    evalTrainResults = True    
    BNEnabled = int(args.bnorm) == 1

    decay_Enabled = int(args.lr_decay) == 1

    # Shuffle data before producing the minibatches to avoid having all-case or all-control minibatches
    np.random.seed(args.randomSeed)
    random.seed(args.randomSeed)
    indices = np.asarray( range(len(M)) ) # is used for storting
    random.shuffle(indices)
    M = M[indices]
    y = y[indices]
    IDs[0] = np.array(IDs[0])
    IDs[1] = np.array(IDs[1])
    IDs[0] = IDs[0][indices]
    IDs[1] = IDs[1][indices]

    # reshape data to be the right dimensions for Convolutions
    if args.convLayers > 0 :
        M = M.reshape(M.shape[0], 1 , 1, M.shape[1])
        if M_validation is not None :
            M_validation = M_validation.reshape(M_validation.shape[0], 1 , 1, M_validation.shape[1]) 

    # 2. create minibatch list
    numIndividuals = M.shape[0] 
    numSNPs = M.shape[1] # numSNPs = bed.get_nb_markers(), as we may have removed SNPs, we want to know how many are left
    len_M = len(M)
    len_M_validation = 0
    
    train_GWAS = list()
    train_y = list()
    minibatch_size =  args.batch_size #M.shape[0]  # 64
    
    if args.batch_size == 0 : minibatch_size = len(M)
    num_batches = len(M) // minibatch_size

    # scale the delta by minibatch_size, if we dont have minibatches
    ratio = float(minibatch_size) / numIndividuals # this is 1 if there are no minibatches
    print("orig L2 Regularizer : " + str(hiddenShrinkage) + " minibatches scaled to " + str(hiddenShrinkage * ratio) )
    hiddenShrinkage *= ratio

    start = 0
    end = minibatch_size
    
#    for i in range(num_batches) :
#        train_GWAS.append(M[start:end]  )
#        train_y.append(y[start:end])
#        print("adding batch " + str(i)  + " , start/end: " + str(start) + "/" + str(end)  )
#        start = end
#        end += minibatch_size  
      
    y_batched = y.copy()
    # do this in a more RAM efficient way: keep deleting the bits from the original matrix to free up space as we go along otherwise this step would double the RAM requirements temporarily
    for i in range(num_batches) :
        train_GWAS.append(M[0:minibatch_size]  )
        M = M[minibatch_size:len(M)]
        
        train_y.append(y_batched[0:minibatch_size])
        y_batched = y_batched[minibatch_size:len(y_batched)]
        print("adding batch " + str(i)  + ", minibatch size: " + str(minibatch_size) + " / num left in pool: " + str(len(M))  )
        gc.collect()
        
    print("train_GWAS[0].shape: " + str( train_GWAS[0].shape)  + " // train_y.shape: " + str( train_y[0].shape) )
    del M; gc.collect() # free up memory
        
    if M_validation is not None : 
        len_M_validation = len(M_validation) 
        if args.batch_size == 0 : minibatch_size = len(M_validation)
        
        test_GWAS = list()
        test_y = list()
        evalResults = True
        num_batches = len(M_validation) // minibatch_size
        print("len_M_validation is: " + str(len_M_validation)  + ", minibatch size: " + str(minibatch_size)  + " args.batch_size: " + str(args.batch_size) + " num_batches is: " + str(num_batches))
        start = 0
        end = minibatch_size
        for i in range(num_batches) :
            test_GWAS.append(M_validation[start:end]  )
            test_y.append(y_validation[start:end])
            print("adding batch " + str(i)  + " , start/end: " + str(start) + "/" + str(end)  )
            start = end
            end += minibatch_size  
            
        # del M_validation; gc.collect() # free up memory, cant do this as we need this for the PRS calculation....

    else :
        test_GWAS = None 
        test_y = None
        evalResults = False







    # 3. initialise network params     
    floatPrecision = "float" +str(args.float) 
    print("floatPrecision is: " + floatPrecision)
    knet_main.setDataType(floatPrecision)
    myNet = knet_main.knn(optimizer = args.optimizer) 

    if args.gpu == 1 : 
        print("attempting to init GPU", flush=True)
        knet_main.initGPU()
        print("GPU successfully set", flush=True)
        
    knet_main.set_seed(args.randomSeed)
    if args.orig == 1 : 
        print("setting KNeT optimizer 0 to original version", flush=True)
        knet_main.KNET_ORIG = True

    Input = knet_main.knnLayer( myNet,np.array([-1]), knet_main.LAYER_SUBTYPE_INPUT)
    
    # if conv was enabled we then do NOT regularize stuff at the first FC layer as we only want to regularize by h2 once
    hiddenREGULARIRIZER = "REGULARIZER_RIDGE"
    shrinkage = hiddenShrinkage
    
    if args.convLayers > 0 :
        lastOutput = train_GWAS[0].shape[-1] # the input to the first layer is the last element of the shape array, eg: 33380  
        print("Adding "+str(args.convLayers)+" conv layers, with initial input dimension: " + str(lastOutput), flush=True)

        
        # first conv layer has special logic, we must make it so that size=stride, to avoid the massive space expansion
        # first, find the smallest size/stride that will result in a whole number output size:
#        for i in range(4,21) : # filter sizes of 4 to 20 are considered
#            trialOutput = lastOutput
#            currentStride = filter_size = i
#            trialOutput = (trialOutput - filter_size +2) / currentStride + 1
#            print("trialOutput : " + str(trialOutput) + " / filter_size: " + str(filter_size) + " / currentStride: " + str(currentStride) )
#            if trialOutput % 1 == 0 :
#                print("first Conv layer filter/stride will be: " + str(filter_size), flush=True)
#                break
        currentNumFilters= args.convFilters
        currentStride = 3
        filter_size = 5 # as it turns out it is not actually a problem if the conv outputs something that isn't an integer, so we just need to downsample it
        
        Conv_Layer = knet_main.knnConvLayer( myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN, regularizer = hiddenREGULARIRIZER, shrinkageParam = hiddenShrinkage, p_dropout = args.dropout, n_filters = currentNumFilters, h_filter=1, w_filter=filter_size, padding=1, stride=currentStride, oneD = True)
        if BNEnabled : Spatial_Bnorm = knet_main.knnSpatialBatchNorm(  myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN)
        addActivation(myNet,args.hidAct)
        lastOutput = (lastOutput - filter_size +2) / currentStride + 1
        lastOutput = int(lastOutput) # as these can only be integers
        hiddenREGULARIRIZER = knet_main.REGULARIZER_NONE # only add regularizer for first layer, subsequent layers will always have none
        shrinkage = 0.0
        
        
        currentStride = 1
        pool_size = 2
        for i in range(1, args.convLayers +1) :
            
            # decide on filter size, depending on input, Conv layers must always produce even outputs so that maxpool can half them
            filter_size = 3
            if lastOutput % 2 != 0 : filter_size = 4 # if the current output is not even, then we have to use a filter size of 4, otherwise we get fractions after the maxpool operation
            ## currentNumFilters = (i+1) * args.convFilters
            currentNumFilters = currentNumFilters // 2
            Conv_Layer = knet_main.knnConvLayer( myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN, regularizer = hiddenREGULARIRIZER, shrinkageParam = shrinkage, p_dropout = args.dropout, n_filters = currentNumFilters, h_filter=1, w_filter=filter_size, padding=1, stride=currentStride, oneD = True)
            if BNEnabled : Spatial_Bnorm = knet_main.knnSpatialBatchNorm(  myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN)
            addActivation(myNet,args.hidAct)
            lastOutput = (lastOutput - filter_size +2) / currentStride + 1
            lastOutput = int(lastOutput) # as these can only be integers
            
            MaxPool_Layer = knet_main.knnMaxPool(myNet, oneD = True)   
            lastOutput = (lastOutput - pool_size) / pool_size + 1 # compute what dimensions the conv+maxpool operations are going to leave for the next layer
            
            
            
    
#        Conv_Layer = knet_main.knnConvLayer( myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN, regularizer = "REGULARIZER_RIDGE", shrinkageParam = hiddenShrinkage, p_dropout = args.dropout, n_filters = 128, h_filter=1, w_filter=8, padding=1, stride=4, oneD = True)
#        if BNEnabled : Spatial_Bnorm = knet_main.knnSpatialBatchNorm(  myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN)
#        addActivation(myNet,args.hidAct)
#        MaxPool_Layer = knet_main.knnMaxPool(myNet, oneD = True)
# 
#        Conv_Layer = knet_main.knnConvLayer( myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN, regularizer = "REGULARIZER_NONE", shrinkageParam = 0., p_dropout = args.dropout, n_filters = 128, h_filter=1, w_filter=5, padding=1, stride=2, oneD = True) # will have to be 6 for next one ( 8 for last one)
#        if BNEnabled : Spatial_Bnorm = knet_main.knnSpatialBatchNorm(  myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN)
#        addActivation(myNet,args.hidAct)
#        MaxPool_Layer = knet_main.knnMaxPool(myNet, oneD = True)  
 
#        Conv_Layer = knet_main.knnConvLayer( myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN, regularizer = "REGULARIZER_NONE", shrinkageParam = 0., p_dropout = args.dropout, n_filters = 128, h_filter=1, w_filter=5, padding=1, stride=1, oneD = True)
#        if BNEnabled : Spatial_Bnorm = knet_main.knnSpatialBatchNorm(  myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN)
#        addActivation(myNet,args.hidAct)
#        MaxPool_Layer = knet_main.knnMaxPool(myNet, oneD = True)
#  
#        Conv_Layer = knet_main.knnConvLayer( myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN, regularizer = "REGULARIZER_NONE", shrinkageParam = 0., p_dropout = args.dropout, n_filters = 128, h_filter=1, w_filter=5, padding=1, stride=1, oneD = True)
#        if BNEnabled : Spatial_Bnorm = knet_main.knnSpatialBatchNorm(  myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN)
#        addActivation(myNet,args.hidAct)
#        MaxPool_Layer = knet_main.knnMaxPool(myNet, oneD = True)
# 
        Flatten_Layer = knet_main.knnFlatten(myNet)
        


        
    lastLayerSize = args.firstLayerSize #lastLayerSize_MAX
    
    for i in range(1,hLayerCount+1) : # iterate 1 based, otherwise we will get a reduction after the first layer, no matter the widthReductionRate, as 0 is divisible by anything
        if i > 1 or args.convLayers > 0 : # only add regularizer for first layer, subsequent layers will always have none
            hiddenREGULARIRIZER  = knet_main.REGULARIZER_NONE
            shrinkage = 0.0
        #if i == (hLayerCount-1) : lastWidth = 2 # enforce so that the last widht is always 2, ie 1 neuron makes it MORE like the other LESS likely
        
        
        H_Layer = knet_main.knnLayer(myNet, [lastLayerSize], knet_main.LAYER_SUBTYPE_HIDDEN, regularizer = hiddenREGULARIRIZER, shrinkageParam = shrinkage, p_dropout = args.dropout)
        if BNEnabled : Bnorm = knet_main.knnBatchNorm(  myNet, [-1],knet_main.LAYER_SUBTYPE_HIDDEN)
        addActivation(myNet,args.hidAct)
        
        print("added layer at depth: " + str(i) + " with width: " + str(lastLayerSize) + " / shrinkage: " + str(shrinkage))
        
        # control the 'fatness' of the network: we reduce the width at a given rate: if this is 1, then at every subsequent layer, if its 2, then every 2nd layer etc
        if i % args.widthReductionRate == 0 :  lastLayerSize = lastLayerSize // 2
        
        if lastLayerSize < 2 : break # if 
        


    
    Output = knet_main.knnLayer( myNet,np.array([ y.reshape(y.shape[0],-1).shape[1] ]), knet_main.LAYER_SUBTYPE_OUTPUT, regularizer = "REGULARIZER_NONE", shrinkageParam = 0.0)
    if len( y.shape) > 1 : Out_Act = knet_main.knSoftmax( myNet)
    
#knet_main.checkConvOutput(myNet, [1,5194  ])

    if args.convLayers > 0 : knet_main.checkConvOutput(myNet, [*train_GWAS[0][0][0].shape])
    knet_main.getNetworkMemUsage(myNet,train_GWAS[0].shape) #  of RAM
    

    if args.inference == 0 :
        print("Analysis Run", flush = True)
        results = myNet.learn(train_GWAS, train_y, test_GWAS, test_y, eval_test=evalResults,eval_train=evalTrainResults, num_epochs=args.epochs, eta=args.learnRate,  eval_freq = args.evalFreq, friction = args.momentum, decayEnabled = decay_Enabled)
        getNetworkStructure(myNet)
        
        #writeKNetParamsToDisk(myNet, "C:/0Datasets/NNs/knet_genetic_fc/knet")
        if args.saveWeights is not None : writeKNetParamsToDisk(myNet, args.saveWeights, knet_main.NETWORK_DATATYPE)

        # write epoch results out
        results_its = results["results"]#["results"]  
        os.makedirs(os.path.dirname(args.out), exist_ok=True)     
    
        #write training data means / stds to disk so that we could use those for inference runs later
        print("writing means/stds to disk with datatype: "  + str(sstd.dtype))
        print("sstd shape is: " + str(sstd.shape) + " / mns shape: " + str(mns.shape))
        
        knet_IO.writeVectorToDisk( args.out + "data_mns" , mns, mns.dtype)  
        knet_IO.writeVectorToDisk( args.out + "data_sstd" , sstd, sstd.dtype)  
    
           
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
             
                
        if len_M_validation > 0 :
            producePRS(myNet,M_validation, test_GWAS, IDs_validation, len_M_validation , args.out + "yhat.txt", args.out + "FIDs.txt", y_validation, args.out + "KNET_PRS")
            
#            # write final predictions out
#            yhats = list()
#            totalSofar= 0
#            for i in range(len(test_GWAS)) : # loop through all minbatches
#                totalSofar += len(test_GWAS[i])
#                yhats.append( myNet.forward_propagate(test_GWAS[i],False, forceCast_toCPU = True) )
#           
#            
#            if totalSofar < len_M_validation :
#                print("minibatches did not cover all training samples, so we create last batch out of the remainders")
#                lastBatch_X = M_validation[totalSofar:len_M_validation]
#                yhats.append( myNet.forward_propagate(lastBatch_X,False, forceCast_toCPU = True) )
#            
#            
#            #yhats = list()
#            #yhats.append( np.array([ [0],[1],[2] ]))    
#            #yhats.append( np.array([ [3],[4],[5] ]))  
#            #yhats.append( np.array([ [6],[7],[8] ]))
#            yhat_all = np.concatenate(yhats)
#            print("after merging, we have yhat predictions for : " + str(len(yhat_all)) + " samples", flush=True)
#    
#            print("yhat_all.shape: " + str(yhat_all.shape) + " // indices_validation.shape: " + str(indices_validation.shape) + " // indices.shape: " + str(indices.shape) )
#        
#
#            fileName = args.out + "yhat.txt"
#            with open(fileName, "w") as file:
#                file.write("Profile"  + "\n")
#                
#        
#                for i in range(yhat_all.shape[0]) :
#                    line = str(yhat_all[i][0] )
#                    for j in range(1, len(yhat_all[i]) ):
#                        line = line + "\t" + str(yhat_all[i][j] )
#                        
#                    file.write( line +  "\n")   # file.write(  ( str(yhat[i])[2:-1] ).replace("  ", " ").replace(" ", "\t") +  "\n")
#        
#            # also write out the FID / IIDs in the same order, just as a sanity check (compare this against the .fam files)
#            
#            fileName = args.out + "FIDs.txt"
#            with open(fileName, "w") as file:
#                file.write("FID" + "\t" + "IID" + "\n")
#        
#                for i in range( len(IDs_validation[0]) ) :
#                    line = IDs_validation[0][i] + "\t" + IDs_validation[1][i]
#        
#                    file.write( line +  "\n") 
                    
    else :
        print("Inference Run", flush = True)
        loadKNetParams(myNet, args.loadWeights, knet_main.NETWORK_DATATYPE)
        
        if args.garson == 1: 
            print("producing importance scores via the garson algorithm")
            NNinference = myNet.dream_Garson()
        else :
            print("producing importance scores via deep dreaming")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            # forward propagate with the 1st sample of the training set 
            yhat = myNet.forward_propagate(train_GWAS[0], train = False, saveInput = False, forceCast_toCPU = True)
            
            suppressPrint_orig = myNet.suppressPrint
            myNet.suppressPrint = True
            StartImage = None
            #StartImage = np.random.normal( size=(1,X_test.shape[1]))
            print("producing inference with number of iterations: " + str(args.dreamit), flush=True)
            dream = myNet.dream(0, 100,StartImage,args.dreamit , mFilterSize = 0, blur = 0.0, l2decay = 0.0, small_norm_percentile = 0,lr = 1.5,normalize = False, small_val_percentile = 0)
            NNinference = dream[0].ravel()
            NNinference[np.isnan(NNinference)]=0.0
            myNet.suppressPrint = suppressPrint_orig 
        
            # Here this would need to be more constrained:
            # both LD and MAF need to be taken into account

        knet_IO.writeSNPeffects(args.out + "dream",snpIDs, NNinference)
        
        # the validation here will refer to the TEST set
        if len_M_validation > 0 :
            producePRS(myNet,M_validation, test_GWAS, IDs_validation, len_M_validation , args.out + "yhat.txt", args.out + "FIDs.txt", y_validation, args.out + "KNET_PRS")
   


def producePRS(myNet,origData, miniBatches, IndiIDs, len_total , outLoc_yhat, outLoc_FIDs, ytrue, outLoc_PRS) :
    # write final predictions out
    yhats = list()
    totalSofar= 0
    for i in range(len(miniBatches)) : # loop through all minbatches
        totalSofar += len(miniBatches[i])
        yhats.append( myNet.forward_propagate(miniBatches[i],False, forceCast_toCPU = True) )
   
    
    if totalSofar < len_total :
        print("minibatches did not cover all training samples, so we create last batch out of the remainders")
        lastBatch_X = origData[totalSofar:len_total]
        yhats.append( myNet.forward_propagate(lastBatch_X,False, forceCast_toCPU = True) )
    
    yhat_all = np.concatenate(yhats)
    print("after merging, we have yhat predictions for : " + str(len(yhat_all)) + " samples", flush=True)


  

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
    yhat_all += knet_main.EPSILON # for numerical stability
    rSQ = np.corrcoef( ytrue, yhat_all, rowvar=0)[1,0]**2      
       
    with open(outLoc_PRS, "w") as file: 
            file.write(str(rSQ) ) 
     
                    
def writeKNetParamsToDisk(myNet, targetDir, datatype = 'float32') :    
    os.makedirs(os.path.dirname(targetDir), exist_ok=True)
    
    for i in range( len(myNet.layers) ) :
        if myNet.layers[i] :
            
            if isinstance(myNet.layers[i],knet_main.knnLayer) or isinstance(myNet.layers[i],knet_main.knnSpatialBatchNorm) or isinstance(myNet.layers[i],knet_main.knnBatchNorm) : # if its a layer with trainable params
                if myNet.layers[i].subtype != knet_main.LAYER_SUBTYPE_INPUT : # if its not an input layer
                    # it has at least 6 trainable params: Weights, Momentum, Past_Grads, (2x for bias too)
                    print("writing params for layer " +  type(myNet.layers[i]).__name__  )
                    # MUST cast them to CPU before attempting to write out, otherwise GPU will hang there
                    knet_IO.writeMatrixToDisk( targetDir + "_" + str(i)+ "_w" , knet_main.castOutputToCPU(myNet.layers[i].Weights_W), datatype)  
                    knet_IO.writeMatrixToDisk( targetDir + "_" + str(i)+ "_wb" , knet_main.castOutputToCPU(myNet.layers[i].Weights_bias), datatype)  
                    knet_IO.writeMatrixToDisk( targetDir + "_" + str(i)+ "_m" , knet_main.castOutputToCPU(myNet.layers[i].Momentum), datatype)  
                    knet_IO.writeMatrixToDisk( targetDir + "_" + str(i)+ "_mb" , knet_main.castOutputToCPU(myNet.layers[i].Bias_Momentum), datatype)   
                    knet_IO.writeMatrixToDisk( targetDir + "_" + str(i)+ "_p" , knet_main.castOutputToCPU(myNet.layers[i].Past_Grads), datatype)  
                    knet_IO.writeMatrixToDisk( targetDir + "_" + str(i)+ "_pb" , knet_main.castOutputToCPU(myNet.layers[i].Past_Grads_bias), datatype)  
        
            if isinstance(myNet.layers[i],knet_main.knnSpatialBatchNorm) or isinstance(myNet.layers[i],knet_main.knnBatchNorm) : # if it is a batchnorm type, then it will have another 2 trainable params
                    knet_IO.writeVectorToDisk( targetDir + "_" + str(i)+ "_rv" , knet_main.castOutputToCPU(myNet.layers[i].running_var), datatype)     
                    knet_IO.writeVectorToDisk( targetDir + "_" + str(i)+ "_rm" , knet_main.castOutputToCPU(myNet.layers[i].running_mean),datatype)                       



def loadKNetParams(myNet, targetDir, datatype = 'float32') :
    for i in range( len(myNet.layers) ) :
        if myNet.layers[i] :
            
            if isinstance(myNet.layers[i],knet_main.knnLayer) or isinstance(myNet.layers[i],knet_main.knnSpatialBatchNorm) or isinstance(myNet.layers[i],knet_main.knnBatchNorm) : # if its a layer with trainable params
                if myNet.layers[i].subtype != knet_main.LAYER_SUBTYPE_INPUT : # if its not an input layer
                    # it has at least 6 trainable params: Weights, Momentum, Past_Grads, (2x for bias too)
                    print("loading params for layer " +  type(myNet.layers[i]).__name__  )
                    myNet.layers[i].Weights_W = knet_IO.loadMatrixFromDisk( targetDir + "_" + str(i)+ "_w" ,datatype)  
                    myNet.layers[i].Weights_bias = knet_IO.loadMatrixFromDisk( targetDir + "_" + str(i)+ "_wb",datatype )  
                    myNet.layers[i].Momentum = knet_IO.loadMatrixFromDisk( targetDir + "_" + str(i)+ "_m",datatype )  
                    myNet.layers[i].Bias_Momentum = knet_IO.loadMatrixFromDisk( targetDir + "_" + str(i)+ "_mb",datatype )   
                    myNet.layers[i].Past_Grads = knet_IO.loadMatrixFromDisk( targetDir + "_" + str(i)+ "_p" ,datatype)  
                    myNet.layers[i].Past_Grads_bias = knet_IO.loadMatrixFromDisk( targetDir + "_" + str(i)+ "_pb" ,datatype)  
        
            if isinstance(myNet.layers[i],knet_main.knnSpatialBatchNorm) or isinstance(myNet.layers[i],knet_main.knnBatchNorm) : # if it is a batchnorm type, then it will have another 2 trainable params
                    myNet.layers[i].running_var = knet_IO.loadVectorFromDisk( targetDir + "_" + str(i)+ "_rv" ,datatype)     
                    myNet.layers[i].running_mean = knet_IO.loadVectorFromDisk( targetDir + "_" + str(i)+ "_rm",datatype)     
                    
    myNet.connectLayers() 
    
   
 #   inputData = train_GWAS[0]
#    outPutdata = train_y[0]
def performGradientCheck(myNet, inputData, outPutdata) :  # the net, Standardised SNPs, and y
    # Gradient Test
    grad_current = myNet.getCurrentWeightGradients(inputData, outPutdata)
    numgrad = myNet.gradientCheck(inputData, outPutdata)
    myNorm = norm(grad_current-numgrad)/norm(grad_current+numgrad)
    return(myNorm )



