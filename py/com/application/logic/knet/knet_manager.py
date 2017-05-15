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
import numpy as np
from numpy.linalg import norm 

def parseParams(iregions) :
    reginUnits  = np.zeros(0)
    for i in range( len(iregions) ): # go through each region
        # number of units in region = length of region
        numUnits = iregions[i][1] - iregions[i][0]
        # print(paste("region end:", iregions[[i]][2], " / regionStart:",  ))
        reginUnits = np.concatenate( (reginUnits, [numUnits]) )
        
    
    return(reginUnits)

def runKnet(irsIds, y,M, regions, epochs, learnRate, momentum, evalFreq = -1, y_validation = None, M_validation  = None, savFreq =-1, predictPheno = False, loadedWeightsData = None, saveWeights = None, randomSeed =1, flag = 0, hiddenShrinkage = 0.0) :

    # 1. standardise data
    qc_data = geno_qc.genoQC_all(M, rsIds = irsIds) # we MUST perform QC with the EXACT SAME settings as the 'region scanner' otherwise the region coordinates will be mismatched
    M = qc_data["X"]
    rsIds_qc = qc_data["rsIds"] # save away the surviving SNP list that we have used 
    indicesToRemove = qc_data["indicesToRemove"]
    Standardised_SNPs = geno_qc.standardise_Genotypes_01(M, 'd') # MUST convert it to doubles, othwerwise it will treat as int and won't get any fractions just 0s and 1s
    print("After standardising, training data in MBs is: ",geno_qc.getSizeInMBs(Standardised_SNPs) )
    
    if M_validation is not None : 
        qc_data = geno_qc.removeList(M_validation, indicesToRemove)
        M_validation = qc_data["X"]
        Standardised_SNPs_validation= geno_qc.standardise_Genotypes_01(M_validation, 'd')
        print("After standardising, validation data in MBs is: ",geno_qc.getSizeInMBs(Standardised_SNPs_validation) )

    # 2. create a 'fake' minibatch list, with only 1 element: the whole data
    train_GWAS = list()
    train_y = list()
    train_GWAS.append(Standardised_SNPs)
    train_y.append(y)
    
    test_GWAS = None
    test_y = None
    if M_validation is not None : 
        test_GWAS = list()
        test_y = list()
        test_GWAS.append( Standardised_SNPs_validation )
        test_y.append(y_validation)
        

    # 3. initialise network params
    np.random.seed(randomSeed)
    numIndividuals = M.shape[0] 
    numSNPs = M.shape[1] # numSNPs = bed.get_nb_markers(), as we may have removed SNPs, we want to know how many are left
    totalNumRegions = len( regions["REGIONS"] )
    regionLambdas = regions["DELTAS"]
    regionSizes = parseParams(   regions["REGIONS"]  )
    minibatch_size = numSNPs # thereare no minibatches effectively
    
    regionRegularizers = np.array( [knet_main.REGULARIZER_RIDGE] * totalNumRegions)
    P_Splitter_region_sizes = regionSizes # the number of regions in the input layer found by the region Scanner
    a_Joiner_region_sizes  =  np.array( [1] * totalNumRegions) # number of units in the conv layer, 1 for each region for now

    ##############
    num_H1_units = 1
    num_H2_units = 1
    c = np.array([num_H2_units])
    hiddenREGULARIRIZER = knet_main.REGULARIZER_NONE
    
    
    if flag == 0 :
        num_H1_units = 1
        print("flag = 0, 1 unit in 1 hidden layer")
    elif flag == 1:
        
        num_H1_units = int(totalNumRegions/2)
        print("flag = 1, ",num_H1_units," units in 1 hidden layer")

    elif flag == 2:
        num_H1_units = int(totalNumRegions/2)
        num_H2_units = int(num_H1_units/2)
        c = np.array([num_H2_units])
        print("flag = 2, ",num_H1_units," units in hidden layer 1, and ", num_H2_units , " units i nhidden layer 2")
    
    elif flag == 3 :
        num_H1_units = 1
        hiddenREGULARIRIZER = knet_main.REGULARIZER_RIDGE
        print("flag = 3, 1 unit in 1 hidden layer, with hidden Shrinkage:", hiddenShrinkage)
        
    elif flag == 4:
        num_H1_units = int(totalNumRegions/2)
        hiddenREGULARIRIZER = knet_main.REGULARIZER_RIDGE
        print("flag = 4, ",num_H1_units," units in 1 hidden layer, with hidden Shrinkage:", hiddenShrinkage)

    elif flag == 5:
        num_H1_units = int(totalNumRegions/2)
        num_H2_units = int(num_H1_units/2)
        c = np.array([num_H2_units])
        hiddenREGULARIRIZER = knet_main.REGULARIZER_RIDGE
        print("flag = 5, ",num_H1_units," units in hidden layer 1, and ", num_H2_units , " units i nhidden layer 2, with hidden Shrinkage:", hiddenShrinkage)
    
    
    
    
    ##############
    
    b = np.array([num_H1_units]) # number of units in the FC layer coming after the Conv layer
    a = np.sum(a_Joiner_region_sizes) # number of units in the Convolutional layer
         
    joinerParams = np.concatenate( ([a] , np.array(range(a)) ) ) # 1st element is the number of units in the conv layer (= number of regions), and the rest of the elements are the splits, this is just 1,2,3,... to the last if we have just conv unit for each region
    splitterParams = np.concatenate( ( [a], *regions["REGIONS"] ) ) # 1st element is the number of units in all INPUT regions combined


    
    # 4. setup network topology (with loaded weights if any)
    myNet = knet_main.knn() # no need to set any params
 
    # create layers
    Input = knet_main.knnLayer(np.array([numSNPs]), myNet, knet_main.LAYER_SUBTYPE_INPUT)
    Splitter = knet_main.knnSplitter(splitterParams, myNet)
    Conv1D = knet_main.knnConv1D(np.array([totalNumRegions]), myNet, a_Joiner_region_sizes, P_Splitter_region_sizes, regionRegularizers, regionLambdas)
    Joiner = knet_main.knnJoiner(joinerParams, myNet)
    H_Layer1 = knet_main.knnLayer(b, myNet, knet_main.LAYER_SUBTYPE_HIDDEN, iactivation= knet_main.k_softplus, regularizer = hiddenREGULARIRIZER, shrinkageParam = hiddenShrinkage)
    
    if flag == 2 or flag == 5: H_Layer2 = knet_main.knnLayer(c, myNet, knet_main.LAYER_SUBTYPE_HIDDEN, iactivation= knet_main.k_softplus, regularizer = hiddenREGULARIRIZER, shrinkageParam = hiddenShrinkage)
    
    Output = knet_main.knnLayer(np.array([1]), myNet, knet_main.LAYER_SUBTYPE_OUTPUT, iactivation= knet_main.k_linear)
    
    
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
    print("Knet Finished")
    



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
                
                
    # loop through all layers of the net
    # if it is knnlayer, save it
    
     # (OPTIONAL) save final phenotype prediction for validation set 
    yhat = None
    if predictPheno : yhat = myNet.forward_propagate(test_GWAS[0])

    
    return( {"results":results, "yhat":yhat, "weights":weights, "rsIds":rsIds_qc } )
   
 #   inputData = train_GWAS[0]
#    outPutdata = train_y[0]
def performGradientCheck(myNet, inputData, outPutdata) :  # the net, Standardised SNPs, and y
    # Gradient Test
    grad_current = myNet.getCurrentWeightGradients(inputData, outPutdata)
    numgrad = myNet.gradientCheck(inputData, outPutdata)
    myNorm = norm(grad_current-numgrad)/norm(grad_current+numgrad)
    return(myNorm )