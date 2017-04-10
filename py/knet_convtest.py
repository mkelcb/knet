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

import copy
import numpy as np
from numpy import genfromtxt
from numpy.linalg import norm

exec(open("./calKinship.py").read()) 
exec(open("./simphen.py").read()) 
exec(open("./regionScanner.py").read()) 

# Generate test Data:
np.random.seed(1)


validationPerc = 0.2
numIndividuals = 400  # 500 training, and 125 validation, at 20% validation
numSNPs = 1257
h2_true = 0.5 #heritability

def generateSNPs(numIndividuals,numSNPs, PLINKgenotypes = False):
    M = np.zeros((numIndividuals, numSNPs))
    for i in range(0, numIndividuals): 
        if PLINKgenotypes == False :   # if we only want genotypes as -1 and +1 
            M[i,] = np.ones(numSNPs)  # generate all 1s
            M[i,] = M[i,] * [np.sign( random.uniform(-1, 1) ) for j in M[i,]] # randomly flip the sign
        else  :
            M[i,] =[random.randint(0,2) for j in M[i,]]
    

    return(M)



# generate SNPs:
M = generateSNPs(numIndividuals,numSNPs,True)  # , TRUE)


percCausals = 0.2

# alternative way of getting phenotypes:
# indices = np.array(range(numSNPs)) # basic ' all SNPs equally causals setup'

# only subset of SNPs are causal
#indices =pickCausalIndices(percCausals * numSNPs,numSNPs)

# specify, local h2 substructure
indices =pickCausalIndices(percCausals * numSNPs,numSNPs, [0.2,0.8], [0.9,0.1])




resultsh2 = generatePhenotypes(h2_true, indices, standardise_Genotypes(M))
y = resultsh2["PHENO"]



# now partition
From = int(numIndividuals - validationPerc * numIndividuals )
to = numIndividuals 
y_validation = y[From:to]
M_validation =  M[From:to,:]

to = From 
y = y[0:to]
M =  M[0:to,:]                  
        

##############################################################

# load Training/Test data from dis
M = genfromtxt('train_GWAS.txt', delimiter='\t')
M_validation = genfromtxt('test_GWAS.txt', delimiter='\t')
y = genfromtxt('train_y.txt', delimiter='\t')
y_validation= genfromtxt('test_y.txt', delimiter='\t')

##############################################################


                
                        
# Create kinship matrices
Standardised_SNPs_validation= standardise_Genotypes(M_validation)
Standardised_SNPs= standardise_Genotypes(M)





K = calc_Kinship( Standardised_SNPs ) # calculate kinship matrix, the standard way XXt/p


# create a 'fake' minibatch list, with only 1 element: the whole data
train_GWAS = list()
train_y = list()
train_GWAS.append(Standardised_SNPs)
train_y.append(y)

test_GWAS = list()
test_y = list()
test_GWAS.append( Standardised_SNPs_validation )
test_y.append(y_validation)

##############################################################

# alternatively load data from disk:

#convW_bias = genfromtxt("Wbias_Conv1.txt", delimiter='\t')
#convW_bias.shape[0]
#convW_bias

def loadNetworkWeightsFromDisk(network, numRegions = 4) :

    deltas = genfromtxt('deltas.txt', delimiter='\t')  # interWeaved["ALLDELTAS"]
    convW = list()
    convW_bias = list()
    for i in range(numRegions) :
        filename = "W_Conv" + str(i + 1)+ ".txt"
        convW.append( genfromtxt(filename, delimiter='\t') )
        filename = "Wbias_Conv" + str(i + 1)+ ".txt"
        convW_bias.append( genfromtxt(filename, delimiter='\t') )
        
        convW[i] = convW[i].reshape(convW[i].shape[0],-1)
        #convW_bias[i] = convW_bias[i].reshape(convW_bias[i].shape[0],-1)
        
        myNet.layers[2].regions[i].Weights_W = convW[i]
        myNet.layers[2].regions[i].Weights_bias = convW_bias[i]
        myNet.layers[2].regions[i].Lamda = deltas[i]
    
    W5 = genfromtxt('W5.txt', delimiter='\t') 
    W5_bias = genfromtxt('W5_bias.txt', delimiter='\t') 
    W5 = W5.reshape(W5.shape[0],-1)
   # W5_bias = W5_bias.reshape(W5_bias.shape[0],-1)
    myNet.layers[4].Weights_W = W5
    myNet.layers[4].Weights_bias = W5_bias
    
    W6 = genfromtxt('W6.txt', delimiter='\t') 
    W6_bias = genfromtxt('W6_bias.txt', delimiter='\t') 
    #W6 = W6.reshape(W6.shape[0],-1)
   # W6_bias = W6_bias.reshape(W6_bias.shape[0],-1)
    
    myNet.layers[5].Weights_W = W6
    myNet.layers[5].Weights_bias = W6_bias


##############################################################


# look at what  true h2 architecture looks like
# visualiseH2(resultsh2["EFFECTS"], indices,numSNPs )

exec(open("./regionScanner.py").read()) 
# detect regions
h2Architecture = findRegions(y, M, blockSize = 100, stride = 50)


# visualise it
#visualiseRegions(h2Architecture[[0]], numSNPs) # red: h2 detected
#visualiseRegions(h2Architecture[[0]], numSNPs, True) # blue: 'background' regions

#h2Architecture

interWeaved = interweaveRegions( h2Architecture["REGIONS"], h2Architecture["BACKGROUND"], h2Architecture["REGIONDELTAS"], h2Architecture["BG_DELTAS"])
interWeaved["ALLREGIONS"]
interWeaved["ALLDELTAS"]




###############################################################################################################


# parse the inferred architecture into convolutional regions
regions = np.zeros(0) # stores the end position of each region 
for i in range( len(interWeaved["ALLREGIONS"]) ):
    regions = np.concatenate( (regions, [ interWeaved["ALLREGIONS"][i][1] ] ) )


totalNumRegions = len(interWeaved["ALLREGIONS"])
regionLambdas = interWeaved["ALLDELTAS"]





def parseParams(regions) :
    reginUnits  = np.zeros(0)
    for i in range( len(regions) ): # go through each region
        # number of units in region = length of region
        numUnits = regions[i][1] - regions[i][0]
        # print(paste("region end:", regions[[i]][2], " / regionStart:",  ))
        reginUnits = np.concatenate( (reginUnits, [numUnits]) )
        
    
    return(reginUnits)
    

regionSizes = parseParams(interWeaved["ALLREGIONS"])

# np.sum(regionSizes) # should sum to numSNPs
#regionSizes
#regionLambdas
#regions


###############################################################################################################




numEpochs = 2001 # 501  #  2001  # +1 as we will only display training results every 100th iteration
minibatch_size = numSNPs # thereare no minibatches effectively
learnRate = 0.00005 # = eta, if its any larger than this, then the weights get blown up
freqTest = 1 # how often the fit is evaluated, IE a 100 means every 100th evaluation
friction = 0.9


# Knet
exec(open("./knet_bias5.py").read())  # the main NN framework
np.random.seed(1)


regionRegularizers = np.array( [REGULARIZER_RIDGE] * totalNumRegions)
P_Splitter_region_sizes = regionSizes # the number of regions in the input layer found by the region Scanner
a_Joiner_region_sizes  =  np.array( [1] * totalNumRegions) # number of units in the conv layer, 1 for each region for now


b = np.array([1]) # number of units in the FC layer coming after the Conv layer
a = np.sum(a_Joiner_region_sizes) # number of units in the Convolutional layer


        
joinerParams = np.concatenate( ([a] , np.array(range(a)) ) ) # 1st element is the number of units in the conv layer (= number of regions), and the rest of the elements are the splits, this is just 1,2,3,... to the last if we have just conv unit for each region
splitterParams = np.concatenate( ( [a], regions ) ) # 1st element is the number of units in all INPUT regions combined

myNet = knn() # no need to set any params



# create layers
Input = knnLayer(np.array([numSNPs]), myNet, LAYER_SUBTYPE_INPUT)
Splitter = knnSplitter(splitterParams, myNet)
Conv1D = knnConv1D(np.array([totalNumRegions]), myNet, a_Joiner_region_sizes, P_Splitter_region_sizes, regionRegularizers, regionLambdas)
Joiner = knnJoiner(joinerParams, myNet)
H_Layer1 = knnLayer(b, myNet, LAYER_SUBTYPE_HIDDEN, iactivation=k_linear, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0)
Output = knnLayer(np.array([1]), myNet, LAYER_SUBTYPE_OUTPUT, iactivation=k_linear)


# init network by connecting up all layers ( so that the matrices are created the right size
myNet.connectLayers() # note: this resets the Weight matrices

 # overwrite network weights from Disk                  
loadNetworkWeightsFromDisk(myNet)


                      
myNet.learn(train_GWAS, train_y, test_GWAS, test_y, eval_test=True,eval_train=True, num_epochs=numEpochs, eta=learnRate,  eval_freq = freqTest, friction = friction)

# Results

# Gradient Test
inputData = train_GWAS[0]
outPutdata = train_y[0]
grad_current = myNet.getCurrentWeightGradients(inputData, outPutdata)
numgrad = myNet.gradientCheck(inputData, outPutdata)
myNorm = norm(grad_current-numgrad)/norm(grad_current+numgrad)
myNorm 



######################################


yhat = myNet.forward_propagate(train_GWAS[0])
myNet.backpropagate(yhat, train_y[0]) # compare it to the same minibatch's true Y
myNet.update(eta = learnRate, minibatch_size = len(train_y[0]), friction = 0)


################################


# Knet, neuralNetwork MME
exec(open("./knet_bias3.py").read()) 
numEpochs = 2001 #  +1 as we will only display training results every 100th iteration
minibatch_size = numSNPs # thereare no minibatches effectively
learnRate = 0.00005
freqTest = 10 # how often the fit is evaluated, IE a 100 means every 100th evaluation
friction = 0.9  # DISABLE for Gradient checking


np.random.seed(1)
nn_bias = knn( layer_config = [numSNPs, 1], iminibatch_size = minibatch_size , hidden_activation = HIDDEN_ACT_SPLUS, output_type = OUT_REGRESSION , regularizer = REGULARIZER_NONE,shrinkageParam = 0.0, ifriction = friction) # REGULARIZER_LASSO , regularizer = REGULARIZER_RIDGE,shrinkageParam = Lambda
nn_bias.learn(train_GWAS, train_y, test_GWAS, test_y, eval_test=True,eval_train=True, num_epochs=numEpochs, eta=learnRate,  eval_freq = freqTest)

# Results


#W1_ = nn_bias.layers[0].Weights_W

                    
grad_current = nn_bias.getCurrentWeightGradients(inputData, outPutdata,numIndividuals)
numgrad = nn_bias.gradientCheck(inputData, outPutdata)
myNorm = norm(grad_current-numgrad)/norm(grad_current+numgrad)
myNorm 


########################
exec(open("./knet_bias3.py").read()) 
K = calc_Kinship( Standardised_SNPs  )
results = emma.REMLE_GWAS(y, K)
h2_REMLE_emma = results["vg"] / ( results["vg"] + results["ve"])
h2_REMLE_emma

# Knet with L2 ( IE BLUP)
np.random.seed(1)
nn_bias = knn( layer_config = [numSNPs, 1], iminibatch_size = minibatch_size , hidden_activation = HIDDEN_ACT_SPLUS, output_type = OUT_REGRESSION , regularizer = REGULARIZER_RIDGE,shrinkageParam = (results["delta"] * numIndividuals), ifriction = friction) # REGULARIZER_LASSO , regularizer = REGULARIZER_RIDGE,shrinkageParam = Lambda
nn_bias.learn(train_GWAS, train_y, test_GWAS, test_y, eval_test=True,eval_train=True, num_epochs=numEpochs, eta=learnRate,  eval_freq = freqTest)

# Results



##########################################
#Trash:
    
#
#fos = prevLayerOutputs[0]
#fos2 = prevLayerOutputs[1]
#
#fos = np.array([ prevLayerOutputs[0] ], )
#fos2 = prevLayerOutputs[1]
#
#
#fos = prevLayerOutputs[0].reshape(320,1)
#fos2 = prevLayerOutputs[1].reshape(320,1)
#
#fos3 = np.append(fos, fos2, axis=1)
#
#fos3 = np.vstack( (fos, fos2) ).T
#fos3 = np.column_stack( (fos, fos2) )
#          
#fos4 = np.stack( (fos, fos2), axis=1 )
#fos5 = np.stack( (fos, fos2), axis=0 )
#
#               
#
#
#fos4 = np.stack( (fos, fos2), axis=1 )          
#fos5 = np.stack( (fos, fos2), axis=0 )    
#
#
#
#fos4 = np.stack( (fos, fos2), axis=1 )          
#fos5 = np.stack( (fos, fos2), axis=0 )
#
#
#
#
#c = np.concatenate([fos[np.newaxis,:],fos2[np.newaxis,:]],axis = 1)
#
#
#fos = np.ones( (3,4)) 
#fos2 = np.ones( (3,4))    +1   
#              
#fos = prevLayerOutputs[0]
#fos2 = prevLayerOutputs[1]
#
#fos4 = np.column_stack( (fos, fos2) )     
#fos5 = np.row_stack( (fos, fos2) )   
#
#fos = np.ones( (3,4)) 
#
#fosZ = np.array(fos,ndmin=2)
#
#fosZ = np.array(fos,ndmin=2).T
#               
#fosZ2 = np.atleast_2d(fos)
#
#fos[0:2,:]
#
#
#
#myList = list()
#myList.append(np.ones( (3,4)) )
#myList.append(fos )
#
#
#myList2 = [np.ones( (3,4)),fos ]
#
#OneD = np.array(range(12)).reshape(3,4)
#OneD = prevLayerOutputs[0][0:5]
#
#OneD_Force = np.atleast_2d(OneD)
#
#
#OneD_Force2 =OneD.reshape(OneD.shape[0],-1)
#
#
#           mat = Input[:,regionStart:regionEnd] # want to force 2D array structure, and at the same time keep orientation (IE atleast_2d().T would transpose 2D arrays )
#            mat.reshape(mat.shape[0],-1)
#
#
#W5 = myNet.layers[4].Weights_W
#Conv_W1 = myNet.layers[2].regions[0].Weights_W
#Conv_D1 =  myNet.layers[2].regions[0].Error_D
