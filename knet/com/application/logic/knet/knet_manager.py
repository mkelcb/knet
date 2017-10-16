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


from ....application.utils import geno_qc
from ....application.logic.knet import knet_main
from ....io import knet_IO
import gc
import numpy as np
from numpy.linalg import norm 

lastLayerSize_MAX = int(4096 /2)

def parseParams(iregions) :
    reginUnits  = np.zeros(0)
    for i in range( len(iregions) ): # go through each region
        # number of units in region = length of region
        numUnits = iregions[i][1] - iregions[i][0]
        # print(paste("region end:", iregions[[i]][2], " / regionStart:",  ))
        reginUnits = np.concatenate( (reginUnits, [numUnits]) )
        
    
    return(reginUnits)

def runKnet(args, epochs, learnRate, momentum, regions = None, evalFreq = -1, savFreq =-1, predictPheno = False, loadedWeightsData = None, saveWeights = None, randomSeed =1, hLayerCount = 0, hiddenShrinkage = 0.0, hLayerActivation = 0.0, amblup_regions = None, priors = None) :

    # default QC settings used for all non AMBLUP versions
    _minObserved = 0.95
    _minMAF = 0.01
    _minVariance = 0.02
    
    if amblup_regions is not None :
        # if we are loading AMBLUP regions, we do NOT want to QC them, but rely on AMBLUP entirely
        _minObserved = -1
        _minMAF = -1
        _minVariance = -1
    
    # load plink binary / phenotypes want to load them here, so that we can release the memory once the raw data is no longer used
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : recodecc = False

    
    genotypeData = knet_IO.loadPLINK(args.knet, loadPhenos = False) 
    M = genotypeData["M"]
    irsIds = genotypeData["rsid"]
    del genotypeData ; gc.collect() # dont need this
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
 
    # if we have a validation set
    M_validation = None
    y_validation = None
    if args.validSet :
        genotypeData = knet_IO.loadPLINK(args.validSet, loadPhenos = False, replaceMissing = True) # want to replace -1 with 0s, as we otherwise would have -1s, as later we just delete indices that failed QC for the training set, but won't care for individual missing datas
        M_validation = genotypeData["M"] 
        del genotypeData ; gc.collect() # dont need this
        
        if args.validPhen :
            y_validation = knet_IO.loadPLINKPheno(args.validPhen, caseControl = cc, recodeCaseControl = recodecc) 
            

    
    
    # 1. standardise data ( this will only do anything for non-ABMLUP cases)
    qc_data = geno_qc.genoQC_all(M, rsIds = irsIds, minObserved = _minObserved, minMAF = _minMAF, minVariance = _minVariance) # we MUST perform QC with the EXACT SAME settings as the 'region scanner' otherwise the region coordinates will be mismatched
    #M = qc_data["X"] 
    rsIds_qc = qc_data["rsIds"] # save away the surviving SNP list that we have used 
    indicesToRemove = qc_data["indicesToRemove"]
    del qc_data; gc.collect() # overwrite

    #M = geno_qc.standardise_Genotypes(M) ; gc.collect()
    #print("After standardising, training data in MBs is: ",geno_qc.getSizeInMBs(M) )
    

    if M_validation is not None : 
        qc_data = geno_qc.genoQC_all(M_validation, rsIds = irsIds, minObserved = _minObserved, minMAF = _minMAF, minVariance = _minVariance)
        indicesToRemove_validation = qc_data["indicesToRemove"]
        del qc_data; gc.collect() 
        # want ALL the SNPs that violate QC either the training or the validation set, and remove union from both
        # this ensures that we will never have monomorphic variants in either one of them
        numSNPsToRemove = len(indicesToRemove)
        indicesToRemove = list(set(indicesToRemove) | set(indicesToRemove_validation)) # # '|' is union: merge unique elements in each list
        if len(indicesToRemove) > numSNPsToRemove  : print("there were " , str(len(indicesToRemove) - numSNPsToRemove), " extra SNPs that needed to be removed due failing QC in the validation set")
    
        qc_data = geno_qc.removeList(M_validation, indicesToRemove)
        M_validation = qc_data["X"] 
        del qc_data; gc.collect() 
        M_validation= geno_qc.standardise_Genotypes(M_validation)  ; gc.collect() 
        print("After standardising, validation data in MBs is: ",geno_qc.getSizeInMBs(M_validation) )

    # only remove SNPs from the training set once we have any potential extra removals in from the validation set
    qc_data = geno_qc.removeList(M, indicesToRemove)
    M = qc_data["X"]
    del qc_data; gc.collect() # overwrite
    M = geno_qc.standardise_Genotypes(M) ; gc.collect()
    print("After standardising, training data in MBs is: ",geno_qc.getSizeInMBs(M) )


    # if there are any prior weights on each SNP's importance
    if priors is not None :    
        print("multiply Genotype matrix by a weight vector (1 for each SNP)", flush=True )
        # remove the ones that were QCd out
        priors = np.delete(priors, indicesToRemove)
        M = M * priors # this us the same as M.dot(np.diag( priors)) , but does not require a pxp matrix
    
    



    # if we have AMBLUP external region data, we need to apply them now
    if amblup_regions is not None :
        print("reordering genotype matrix to match the AMBLUP region SNPs")
        regions_boundaries = list()
        
        allIndices = list()
        for i in range(len(amblup_regions["REGIONS"])) :
            amblupregion = amblup_regions["REGIONS"][i] # get AMBLUP region RSids for current region
            rsIds_qc_list = rsIds_qc.tolist()
            indices = ( [rsIds_qc_list.index(i) for i in amblupregion] ) # find the AMBLUP region's RSid's indices in the total M indices
            # AMBLUP regions are a all nonoverlapping: so bounderies defined as: [ length so far , new length ]
            regions_boundaries.append( np.array( [ len(allIndices) , len(allIndices) +len(indices)  ] ) )
            allIndices.extend(indices)                    
        
            
        
        indicesToDelete = list()
        learnRateMultiplier = 1
        if amblup_regions["CONV"] == 0 :
            print("AMBLUP fully connected regions with different deltas requested")
            # generate the 'Lambdas' 1 for each SNP based on the region datas/deltas
            hiddenShrinkage = convertDeltasIntoLambdas(amblup_regions["REGIONS"], amblup_regions["DELTAS"] )
            
            # now filter out '0 predictors', as SNPS that are regularized to the max are not useful
            MAXDELTA = np.exp(10)
            for i in range(len(hiddenShrinkage)) :
                if hiddenShrinkage[i] >= MAXDELTA :
                    indicesToDelete.append(i)
            print("we will remove ", len(indicesToDelete), " zero SNPs, out of total:" , len(hiddenShrinkage), ", number of SNPs left: ",   (len(hiddenShrinkage)- len(indicesToDelete) ) )
            
            # if removed 0, then 1
            #howMuchLeft = len(hiddenShrinkage) - len(indicesToDelete) # IE 1500 - 1000 = 500
            #learnRateMultiplier =  1 / (howMuchLeft / len(hiddenShrinkage))
            #learnRate = learnRate * learnRateMultiplier # if we rewmoved this many predictors, then we should increase the learningrate proportionately
            learnRate = learnRate  * 1.5
        
        
        else :
            print("AMBLUP locally-connected regions requested")
            regions = {"REGIONS":regions_boundaries, "DELTAS":amblup_regions["DELTAS"] }
            hiddenShrinkage = 0.0

        # rorder/extract both train/valid genotype matrices, so that the order of SNPs matches AMBLUP regions, IE the correct Deltas apply to the correct set of SNPs
        M = M[:,allIndices]
        if M_validation is not None : M_validation = M_validation[:,allIndices]
        
        # remove 0 SNPs and their Shrinkage params
        if len(indicesToDelete) > 0 :
                hiddenShrinkage = np.delete(hiddenShrinkage, indicesToDelete)[:, np.newaxis]
                M = np.delete(M, indicesToDelete, axis=1)
                if M_validation is not None : 
                    M_validation = np.delete(M_validation, indicesToDelete, axis=1)
                
        
        


    # 2. create a 'fake' minibatch list, with only 1 element: the whole data
    train_GWAS = list()
    train_y = list()
    train_GWAS.append(M)
    train_y.append(y)
    
    test_GWAS = None
    test_y = None
    if M_validation is not None : 
        test_GWAS = list()
        test_y = list()
        test_GWAS.append( M_validation )
        test_y.append(y_validation)
        

    # 3. initialise network params
    np.random.seed(randomSeed)
    numIndividuals = M.shape[0] 
    numSNPs = M.shape[1] # numSNPs = bed.get_nb_markers(), as we may have removed SNPs, we want to know how many are left
    minibatch_size = numSNPs # thereare no minibatches effectively
    
    

    outPutActivation = knet_main.k_linear
    if len( y.shape) > 1 : 
        print("output is discrete, so we use softmax")
        outPutActivation = knet_main.k_softmax
       
    # 4. setup network topology (with loaded weights if any)
    myNet = knet_main.knn() # no need to set any params
 
    # create layers
    Input = knet_main.knnLayer(np.array([numSNPs]), myNet, knet_main.LAYER_SUBTYPE_INPUT)
    
    # determine regularizer type: 
    hiddenREGULARIRIZER = knet_main.REGULARIZER_NONE
    if type(hiddenShrinkage) is np.ndarray or  hiddenShrinkage != 0.0 : # this needs to work if Lambda is either scalar OR an array (IE if it was an array and checked for != 0.0. then that would throw an error)
        hiddenREGULARIRIZER = knet_main.REGULARIZER_RIDGE

    hLayerActFunction = knet_main.k_softplus
    if hLayerActivation == 1 :  hLayerActFunction = knet_main.k_sigmoid
    elif hLayerActivation == 2 :  hLayerActFunction = knet_main.k_leakyRELU
    elif hLayerActivation == 3 :  hLayerActFunction = knet_main.k_linear
    
    
    # Convolutions
    if regions is not None :
        print("We got regions data for locally connected network")
        # learnRate = learnRate * 10 # conv networks should learn faster
        totalNumRegions = len( regions["REGIONS"] )
        regionLambdas = regions["DELTAS"]
        regionSizes = parseParams(   regions["REGIONS"]  )
        regionRegularizers = np.array( [knet_main.REGULARIZER_RIDGE] * totalNumRegions)
        P_Splitter_region_sizes = regionSizes # the number of regions in the input layer found by the region Scanner
        a_Joiner_region_sizes  =  np.array( [1] * totalNumRegions) # number of units in the conv layer, 1 for each region for now
        a = np.sum(a_Joiner_region_sizes) # number of units in the Convolutional layer   
        joinerParams = np.concatenate( ([a] , np.array(range(a)) ) ) # 1st element is the number of units in the conv layer (= number of regions), and the rest of the elements are the splits, this is just 1,2,3,... to the last if we have just conv unit for each region
        splitterParams = np.concatenate( ( [a], *regions["REGIONS"] ) ) # 1st element is the number of units in all INPUT regions combined

       # num_H1_units = totalNumRegions # 1
        #b = np.array([num_H1_units]) # number of units in the FC layer coming after the Conv layer
        Splitter = knet_main.knnSplitter(splitterParams, myNet)
        Conv1D = knet_main.knnConv1D(np.array([totalNumRegions]), myNet, a_Joiner_region_sizes, P_Splitter_region_sizes, regionRegularizers, regionLambdas) # NOTE: Conv layers are all LINEAR outputs...
        Joiner = knet_main.knnJoiner(joinerParams, myNet)
        
        # if there ARE  convolutions then the first hidden layer size is the number of regions we had
        lastLayerSize = 4096 # totalNumRegions

        # add a fully connected network, with the same number of units as regions
        # H_Layer1 = knet_main.knnLayer(b, myNet, knet_main.LAYER_SUBTYPE_HIDDEN, iactivation= hLayerActFunction, regularizer = knet_main.REGULARIZER_NONE, shrinkageParam = 0.0)
    # if there are no convolutions then the first hidden layer size is hardcoded
    else : lastLayerSize = 4096 # 5000
    
    
    # new logic for working out layersizes, start at half, and end up at 2
    lastLayerSize = int(numSNPs/2) # this is too big
    if lastLayerSize > lastLayerSize_MAX : lastLayerSize = lastLayerSize_MAX
    orig_lastLayerSize = lastLayerSize

    hLayerCount = 1
    while lastLayerSize >= 4 :
        print("lastLayerSize:", lastLayerSize)
        lastLayerSize = int(lastLayerSize/2)
        hLayerCount += 1
    print("determined to have hLayerCount;" , hLayerCount)
    
   
    lastLayerSize =orig_lastLayerSize  # reset this

    #if numSNPs < 1.5 * lastLayerSize :
    #    lastLayerSize = int(lastLayerSize /2)
    #    hLayerCount = int(hLayerCount -1)
    
    #lastLayerSize = int( np.sqrt(numSNPs) )

    for i in range(hLayerCount) :
        if i > 0 or regions is not None: # only add regularizer for first layer, subsequent layers will always have none
                    # OR if we had convolutions, as then the regularization happened there
            hiddenREGULARIRIZER  = knet_main.REGULARIZER_NONE
            hiddenShrinkage = 0.0
            print("forcing subsequent layers to have no regularization")
        H_Layer = knet_main.knnLayer([lastLayerSize], myNet, knet_main.LAYER_SUBTYPE_HIDDEN, iactivation= hLayerActFunction, regularizer = hiddenREGULARIRIZER, shrinkageParam = hiddenShrinkage)
        #lastLayerSize = int ( np.sqrt(lastLayerSize) )
        # lastLayerSize = int(lastLayerSize/ 2)
        if i == hLayerCount -2 : lastLayerSize  = int(2)
        else: lastLayerSize = int(lastLayerSize/ 2)
        
    print("added " , hLayerCount, " hidden layers with regularizers: ", hiddenREGULARIRIZER, " / activation:" , hLayerActFunction )
    
    outputShape = 1
    if len(y.shape) > 1 : outputShape = y.shape[1]
    Output = knet_main.knnLayer(np.array([ outputShape ]), myNet, knet_main.LAYER_SUBTYPE_OUTPUT, iactivation= outPutActivation, regularizer = knet_main.REGULARIZER_NONE, shrinkageParam = 0.0)
    
    
    # 5. init network by connecting up all layers ( so that the matrices are created the right size
    initWeights = True
    if loadedWeightsData is not None : initWeights = False
      
    myNet.connectLayers(initWeights) # note: this resets the Weight matrices
    
    if loadedWeightsData is not None :
         counter = 0
         # manually add weights
         for i in range(1, len(myNet.layers)) : # loop from 2nd, as input layer cannot have weights
                layer = myNet.layers[i]
                if type(layer).__name__ is 'knnLayer' :
                   # print(i, " is a knnLayer")
                    layer.Weights_W = loadedWeightsData[counter][0]
                    layer.Momentum = loadedWeightsData[counter][1]
                    layer.Weights_bias = loadedWeightsData[counter][2] # check for .biasEnabled, athough this is always enabled for now..
                    layer.Bias_Momentum = loadedWeightsData[counter][3]
                    counter = counter +1
                         
                elif type(layer).__name__ is 'knnConv1D':
                    #print(i, " is a knnConv1D")
                    
                    for j in range(0, len(layer.regions) ) : # loop through each mini layer
                        layer_mini = layer.regions[j]
                        layer_mini.Weights_W = loadedWeightsData[counter][0]
                        layer_mini.Momentum = loadedWeightsData[counter][1]
                        layer_mini.Weights_bias = loadedWeightsData[counter][2] # check for .biasEnabled, athough this is always enabled for now..
                        layer_mini.Bias_Momentum = loadedWeightsData[counter][3]
                        counter = counter +1
                   
         print(counter , " sets of weights loaded and added into network")

    
    evalResults = False
    if evalFreq != -1 : evalResults = True

    # epochs = 1
    # 6. TRAIN NETWORK
    results = myNet.learn(train_GWAS, train_y, test_GWAS, test_y, eval_test=evalResults,eval_train=evalResults, num_epochs=epochs, eta=learnRate,  eval_freq = evalFreq, friction = momentum)
    print("Knet Finished, total RAM used:", knet_main.getNetRAMUsage(myNet) )
    
    
     # (OPTIONAL) save final phenotype prediction for validation set 
    yhat = None
    if predictPheno : yhat = myNet.forward_propagate(test_GWAS[0])
    
    # release memory: wont work as we still have these inside the network itself
    # del M_validation ; del M; del train_GWAS; del test_GWAS; gc.collect() 
 

    # (OPTIONAL) save weights
    weights = None
    counter = 0
    if saveWeights is not None :
        weights = list()
        #W = list()
        #bias = list()
        #Mom = list()
       # Mom_bias = list()
        for i in range(1, len(myNet.layers)) : # loop from 2nd, as input layer cannot have weights
            layer = myNet.layers[i]
            if type(layer).__name__ is 'knnLayer' :
                weights.append(list())
                print(i, " is a knnLayer")
                weights[counter].append(layer.Weights_W) 
                weights[counter].append(layer.Momentum)
                weights[counter].append(layer.Weights_bias) # check for .biasEnabled, athough this is always enabled for now..
                weights[counter].append(layer.Bias_Momentum)

                counter = counter +1
                                  
            elif type(layer).__name__ is 'knnConv1D':
                print(i, " is a knnConv1D")
                
                for j in range(0, len(layer.regions) ) : # loop through each mini layer
                    layer_mini = layer.regions[j]
                    weights.append(list())
                    weights[counter].append(layer_mini.Weights_W) 
                    weights[counter].append(layer_mini.Momentum)
                    weights[counter].append(layer_mini.Weights_bias) # check for .biasEnabled, athough this is always enabled for now..
                    weights[counter].append(layer_mini.Bias_Momentum)
                    counter = counter +1
               
        print(counter , " sets of weights were collected disk")
                

    


    
    return( {"results":results, "yhat":yhat, "weights":weights, "rsIds":rsIds_qc } )
   
 #   inputData = train_GWAS[0]
#    outPutdata = train_y[0]
def performGradientCheck(myNet, inputData, outPutdata) :  # the net, Standardised SNPs, and y
    # Gradient Test
    grad_current = myNet.getCurrentWeightGradients(inputData, outPutdata)
    numgrad = myNet.gradientCheck(inputData, outPutdata)
    myNorm = norm(grad_current-numgrad)/norm(grad_current+numgrad)
    return(myNorm )




# we have 1 Deltas for each REGION, but we need one Lambda for each SNP
# IE we need to repeat the same Delta for each SNP within a region
def convertDeltasIntoLambdas(allRegions,allDeltas ) :
    allLambdas = list()
    for i in range( len(allRegions) ):
        regionLength = len( allRegions[i] ) # get  how many SNPs are in current regions
        allLambdas.extend(  [allDeltas[i]] * regionLength    ) # repeat the relevant Delta that many times
        
    # convert it into a column vector
    allLambdas = np.array(allLambdas)[:, np.newaxis] 
    return(allLambdas)

