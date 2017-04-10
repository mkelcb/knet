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


# Generate test Data:
source("calKinship.R")
source("simphen.R")
source("regionScanner.R")

set.seed(1)

validationPerc = 0.2
numIndividuals = 400  # 500 training, and 125 validation, at 20% validation
numSNPs = 1257
h2_true <- 0.5 #heritability

generateSNPs = function(numIndividuals,numSNPs, PLINKgenotypes = FALSE) {
  M <- matrix(rep(0,numIndividuals*numSNPs),numIndividuals,numSNPs)
  for (i in 1:numIndividuals) {
    if(PLINKgenotypes == FALSE) {  M[i,] <- ifelse(runif(numSNPs)<0.5,-1,1) } else { # if we only want genotypes as -1 and +1 
      M[i,] <-sample(0:2, numSNPs, replace = TRUE)
    }
    
  }
  rownames(M) <- 1:numIndividuals
  return(M)
}



# generate SNPs:
M = generateSNPs(numIndividuals,numSNPs,TRUE)  # , TRUE)

# generate Y:  phenotypes based on heritability
#u <- rnorm(numSNPs) # effect sizes for SNPs
#g <- as.vector(crossprod(t(M),u)) # calculate pure breeding values
#y <- g + rnorm(numIndividuals,mean=0,sd=sqrt((1-h2_true)/h2_true*var(g))) # Y is g + noise
# y =scale(y)

percCausals = 0.2

# alternative way of getting phenotypes:
# indices = ( 1:numSNPs ) # basic ' all SNPs equally causals setup'

# only subset of SNPs are causal
#indices =pickCausalIndices(percCausals * numSNPs,numSNPs)

# specify, local h2 substructure
#indices =pickCausalIndices(percCausals * numSNPs,numSNPs, c(0.1,0.4,0.1,0.3,0.1), c(0.4,0.05,0.3,0.05,0.2))
#indices =pickCausalIndices(percCausals * numSNPs,numSNPs, c(0.5,0.5), c(0.9,0.1))
indices =pickCausalIndices(percCausals * numSNPs,numSNPs, c(0.2,0.8), c(0.9,0.1))

resultsh2 = generatePhenotypes(h2_true, indices, standardise_Genotypes(M))
y = resultsh2$PHENO



# now partition
from = numIndividuals - validationPerc * numIndividuals +1
to = numIndividuals
y_validation = y[from:to]
M_validation =  M[from:to,]

to = from -1
y = y[1:to]
M =  M[1:to,]

# Standardise Genotypes: ( this may make XtX singular....)
Standardised_SNPs_validation= standardise_Genotypes(M_validation)
Standardised_SNPs= standardise_Genotypes(M)




# create a 'fake' minibatch list, with only 1 element: the whole data
train_GWAS = list()
train_y = list()
train_GWAS[[1]] = Standardised_SNPs
train_y[[1]] = y

test_GWAS = list()
test_y = list()
test_GWAS[[1]] = Standardised_SNPs_validation
test_y[[1]] = y_validation



##############################################################

writeDataToDisk = function () {
  
  # need to write deltas onto disk too, as those depend on an eigendecomposition, that differns between R/Python
  write.table(interWeaved$ALLDELTAS, "py/deltas.txt", sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE) 
  
  # write data Training/Test data onto disk
  write.table(M, "py/train_GWAS.txt", sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE) 
  write.table(train_y[[1]], "py/train_y.txt", sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE) 
  write.table(M_validation, "py/test_GWAS.txt", sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE) 
  write.table(test_y[[1]], "py/test_y.txt", sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE) 
  
  # write Weights onto disk
  
  # input layer has no weights
  # splitter has no weights
  
  
  # Conv: has a number of nested knnLayers inside
  for(i in 1:myNet$layers[[3]]$numRegions ) {
    filename = paste("py/W_Conv", i, ".txt" , sep="")
    write.table(myNet$layers[[3]]$regions[[i]]$Weights_W, filename, sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
    
    filename = paste("py/Wbias_Conv", i, ".txt" , sep="")
    write.table(myNet$layers[[3]]$regions[[i]]$Weights_bias, filename, sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  }
  
  # write wiehgts of the FC layer
  write.table(myNet$layers[[5]]$Weights_W, "py/W5.txt", sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE) 
  write.table(myNet$layers[[5]]$Weights_bias, "py/W5_bias.txt", sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE) 
  
  # write wiehgts of the Output layer
  write.table(myNet$layers[[6]]$Weights_W, "py/W6.txt", sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE) 
  write.table(myNet$layers[[6]]$Weights_bias, "py/W6_bias.txt", sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE) 
  
}


loadWeights = function () {
  
  deltas = unlist ( as.vector( read.table("py/deltas.txt") ) )
  
  for(i in 1:myNet$layers[[3]]$numRegions ) {
    filename = paste("py/W_Conv", i, ".txt" , sep="")
    myNet$layers[[3]]$regions[[i]]$Weights_W = as.matrix( read.table(filename) )
    
    filename = paste("py/Wbias_Conv", i, ".txt" , sep="")
    myNet$layers[[3]]$regions[[i]]$Weights_bias=  unlist ( as.vector(read.table(filename) ) )
    myNet$layers[[3]]$regions[[i]]$Lambda = deltas[i]
  }
  
  myNet$layers[[5]]$Weights_W = as.matrix( read.table("py/W5.txt") )
  myNet$layers[[5]]$Weights_bias = unlist ( as.vector( read.table("py/W5_bias.txt") ) )
  
  
  myNet$layers[[6]]$Weights_W = as.matrix( read.table("py/W6.txt") )
  myNet$layers[[6]]$Weights_bias = unlist ( as.vector( read.table("py/W6_bias.txt") ) )
  
}



##############################################################


# look at what  true h2 architecture looks like
visualiseH2(resultsh2$EFFECTS, indices,numSNPs )

source("regionScanner.R")
# detect regions
h2Architecture = findRegions(y, M, blockSize = 100, stride = 50)


# visualise it
visualiseRegions(h2Architecture[[1]], numSNPs) # red: h2 detected
visualiseRegions(h2Architecture[[2]], numSNPs, TRUE) # blue: 'background' regions

#h2Architecture

interWeaved = interweaveRegions( h2Architecture$REGIONS, h2Architecture$BACKGROUND, h2Architecture$REGIONDELTAS, h2Architecture$BG_DELTAS)

interWeaved$ALLREGIONS
interWeaved$ALLDELTAS

###############################################################################################################

# parse the inferred architecture into convolutional regions
regions = vector() # stores the end position of each region

for(i in 1:length(interWeaved$ALLREGIONS)) { 
  regions = c(regions, interWeaved$ALLREGIONS[[i]][2]) 
}

totalNumRegions = length(interWeaved$ALLREGIONS)
regionLambdas = interWeaved$ALLDELTAS


parseParams = function(regions  ) {


  reginUnits  = vector()
  for(i in 1:length(regions)) { # go through each region
    # number of units in region = length of region
    numUnits = (regions[[i]][2] - regions[[i]][1]) + 1
   # print(paste("region end:", regions[[i]][2], " / regionStart:",  ))
    reginUnits = c(reginUnits, numUnits)
  }

  return(reginUnits)
}

regionSizes = parseParams(interWeaved$ALLREGIONS)


#sum(regionSizes) # should sum to numSNPs
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
debugSource("knet_bias5.R") # the main NN framework
set.seed(1)

regionRegularizers = rep(REGULARIZER_RIDGE,totalNumRegions)
P_Splitter_region_sizes = regionSizes # the number of regions in the input layer found by the region Scanner
a_Joiner_region_sizes  = rep(1,totalNumRegions) # number of units in the conv layer, 1 for each region for now



b = 1 # number of units in the FC layer coming after the Conv layer
a = sum(a_Joiner_region_sizes) # number of units in the Convolutional layer

joinerParams = c(a , 1:a ) # 1st element is the number of units in the conv layer (= number of regions), and the rest of the elements are the splits, this is just 1,2,3,... to the last if we have just conv unit for each region
splitterParams = c( a, regions ) # 1st element is the number of units in all INPUT regions combined

myNet = knn() # no need to set any params

# create layers
Input = knnLayer(numSNPs, myNet, LAYER_SUBTYPE_INPUT)
Splitter = knnSplitter(splitterParams, myNet)
Conv1D = knnConv1D(totalNumRegions, myNet, a_Joiner_region_sizes, inumUnitsInPrevLayer_Pi = P_Splitter_region_sizes, regionRegularizers, regionLambdas)
Joiner = knnJoiner(joinerParams, myNet)
H_Layer1 = knnLayer(b, myNet, LAYER_SUBTYPE_HIDDEN, iactivation=k_linear, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0)
Output = knnLayer(1, myNet, LAYER_SUBTYPE_OUTPUT, iactivation=k_linear)


# init network by connecting up all layers ( so that the matrices are created the right size
myNet$connectLayers() # note: this resets the Weight matrices

writeDataToDisk()
loadWeights() # load it so that weights ar 100% identical (otherwise we would have rounding errors)

myNet$learn(train_GWAS, train_y, test_GWAS, test_y, eval_test=T,eval_train=T, num_epochs=numEpochs, eta=learnRate,  eval_freq = freqTest, friction = friction)

# [1] "it: 1000  / Training prediction (r^2) : 0.17455  / Test  prediction (r^2) : 1.387e-05"

# [1] "it: 20000  / Training prediction (r^2) : 0.89878  / Test  prediction (r^2) : 0.14717"


######################################################

# Gradient Test
inputData = train_GWAS[[1]]
outPutdata = train_y[[1]]
grad_current = myNet$getCurrentWeightGradients(inputData, outPutdata)
numgrad = myNet$gradientCheck(inputData, outPutdata)
myNorm = norm(grad_current-numgrad)/norm(grad_current+numgrad)
grad_current_mat = matrix(grad_current)
numgrad_mat = matrix(numgrad)
myNorm # 0.0004139675,  1.34703e-12

######################################################

# individual iteration test
#yhat = myNet$forward_propagate(train_GWAS[[1]])
#myNet$backpropagate(yhat, train_y[[1]]) # compare it to the same minibatch's true Y
#myNet$update(eta = learnRate, minibatch_size = length(train_y[[1]]), friction = 0)


################################



# Knet, neuralNetwork MME
debugSource("knet_bias3.R") # the main NN framework
numEpochs = 2001 #  +1 as we will only display training results every 100th iteration
minibatch_size = numSNPs # thereare no minibatches effectively
learnRate = 0.00005
#numTraining = 500 # how many training samples we use to train network
freqTest = 10 # how often the fit is evaluated, IE a 100 means every 100th evaluation
friction = 0.9  # DISABLE for Gradient checking


set.seed(1)
nn_bias = knn( layer_config = c(numSNPs, 1), iminibatch_size = minibatch_size , hidden_activation = HIDDEN_ACT_SPLUS, output_type = OUT_REGRESSION , regularizer = REGULARIZER_NONE,shrinkageParam = 0.0, ifriction = friction) # REGULARIZER_LASSO , regularizer = REGULARIZER_RIDGE,shrinkageParam = Lambda
nn_bias$learn(train_GWAS, train_y, test_GWAS, test_y, eval_test=TRUE,eval_train=TRUE, num_epochs=numEpochs, eta=learnRate,  eval_freq = freqTest)


W1_ = nn_bias$layers[[1]]$Weights_W

nn_bias$layers[[1]]$Weights_bias

grad_current = nn_bias$getCurrentWeightGradients(inputData, outPutdata,numIndividuals)

numgrad = nn_bias$gradientCheck(inputData, outPutdata)
myNorm = norm(grad_current-numgrad)/norm(grad_current+numgrad)
grad_current_mat = matrix(grad_current)
numgrad_mat = matrix(numgrad)
myNorm  # 0.0004139675, WITHOUT friction: 1.34703e-12



# [1] "it: 20000  / Training prediction (r^2) : 1  / Test  prediction (r^2) : 3.3598e-05" ] 625/4000/
# [1] "it: 20000  / Training prediction (r^2) : 0.85501  / Test  prediction (r^2) : 0.39671"




########################
debugSource("knet_bias3.R") # the main NN framework
K <- calc_Kinship( Standardised_SNPs  )
results = emma.REMLE_GWAS(y, K)
h2_REMLE_emma = results$vg / ( results$vg + results$ve)
h2_REMLE_emma # 0.4661182
# 0.4698294


# Knet with L2 ( IE BLUP)
set.seed(1)
nn_bias = knn( layer_config = c(numSNPs, 1), iminibatch_size = minibatch_size , hidden_activation = HIDDEN_ACT_SPLUS, output_type = OUT_REGRESSION , regularizer = REGULARIZER_RIDGE,shrinkageParam = (results$delta * numIndividuals), ifriction = friction) # REGULARIZER_LASSO , regularizer = REGULARIZER_RIDGE,shrinkageParam = Lambda
nn_bias$learn(train_GWAS, train_y, test_GWAS, test_y, eval_test=TRUE,eval_train=TRUE, num_epochs=numEpochs, eta=learnRate,  eval_freq = freqTest)
# [1] "it: 20000  / Training prediction (r^2) : 0.97335  / Test  prediction (r^2) : 0.048375"


