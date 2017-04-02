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
numIndividuals = 200  # 500 training, and 125 validation, at 20% validation
numSNPs = 100
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


# look at what  true h2 architecture looks like
visualiseH2(resultsh2$EFFECTS, indices,numSNPs )

# detect regions
h2Architecture = findRegions(y, M, blockSize = 10, stride = 5)


# visualise it
visualiseRegions(h2Architecture[[1]], numSNPs) # red: h2 detected
visualiseRegions(h2Architecture[[2]], numSNPs, TRUE) # blue: 'background' regions



###############################################################################################################

# parse the inferred architecture into convolutional regions
regions_wh2 = h2Architecture$REGIONS
regions_background = h2Architecture$BACKGROUND


# we need to order the  genotype matrix so that it is ordered according to the regions found (probably inefficient)
Standardised_SNPs_ordered = NULL
Standardised_SNPs_validation_ordered = NULL
regions = vector()

for(i in 1:length(regions_wh2)) { # first go through all regions with h2, as those will come first
  startPos = regions_wh2[[i]][1]
  endPos = regions_wh2[[i]][2]
  
  Standardised_SNPs_ordered = cbind(Standardised_SNPs_ordered, Standardised_SNPs[,startPos:endPos])
  Standardised_SNPs_validation_ordered = cbind(Standardised_SNPs_validation_ordered, Standardised_SNPs_validation[,startPos:endPos])
  regions = c(regions, ncol(Standardised_SNPs_ordered)) # we want to find out ACTUALLY where the region ends now that we have reordered the design matrix
  
}

for(i in 1:length(regions_background)) { # first go through all regions with h2, as those will come first
  startPos = regions_background[[i]][1]
  endPos = regions_background[[i]][2]
  
  Standardised_SNPs_ordered = cbind(Standardised_SNPs_ordered, Standardised_SNPs[,startPos:endPos])
  Standardised_SNPs_validation_ordered = cbind(Standardised_SNPs_validation_ordered, Standardised_SNPs_validation[,startPos:endPos])
  regions = c(regions, ncol(Standardised_SNPs_ordered)) # we want to find out ACTUALLY where the region ends now that we have reordered the design matrix
}

numRegions_wh2 = length(regions_wh2)
numRegions_bg = length(regions_background)
totalNumRegions = numRegions_wh2 + numRegions_bg


region_wh2_deltas = h2Architecture$REGIONDELTAS
region_bg_deltas = h2Architecture$BG_DELTA
regionLambdas = c(region_wh2_deltas, rep(region_bg_deltas,numRegions_bg)) # the lambdas are all the with H2 region's deltas, and then repeat the same delta for all the BG regions ( probably not 100% accurate)

parseParams = function(regions, pastResults = NULL  ) {

  if( is.null(pastResults)[1] == FALSE ) { # if we have called this function before and have passed in a results object
    reginUnits  = pastResults
  } else {
    reginUnits  = vector()
  }
  
  for(i in 1:length(regions)) { # go through each region
    # number of units in region = length of region
    numUnits = (regions[[i]][2] - regions[[i]][1]) + 1
   # print(paste("region end:", regions[[i]][2], " / regionStart:",  ))
    reginUnits = c(reginUnits, numUnits)
  }

  return(reginUnits)
}

regionSizes = parseParams(regions_wh2)
regionSizes = parseParams(regions_background, regionSizes)

#sum(regionSizes) # should sum to numSNPs
#regionSizes
#regionLambdas
#regions



###############################################################################################################

numEpochs = 201 # 501  #  2001  # +1 as we will only display training results every 100th iteration
minibatch_size = numSNPs # thereare no minibatches effectively
learnRate = 0.0005 # = eta, if its any larger than this, then the weights get blown up
freqTest = 10 # how often the fit is evaluated, IE a 100 means every 100th evaluation
friction = 0.8


# Knet
debugSource("knet_bias4.R") # the main NN framework
set.seed(1)

regionRegularizers = rep(REGULARIZER_RIDGE,totalNumRegions)
P_Splitter_region_sizes = regionSizes # the number of regions in the input layer found by the region Scanner
a_Joiner_region_sizes  = rep(1,totalNumRegions) # number of units in the conv layer, 1 for each region for now



b = 1 # number of units in the FC layer coming after the Conv layer
a = sum(a_Joiner_region_sizes)

joinerParams = c(a , 1:a ) # 1st element is the number of units in the conv layer (= number of regions), and the rest of the elements are the splits, this is just 1,2,3,... to the last if we have just conv unit for each region
splitterParams = c( a, regions ) # 1st element is the number of units in all INPUT regions combined

myNet = knn() # no need to set any params

# create layers
Input = knnLayer(numSNPs, LAYERTYPE_INPUT, myNet)

Splitter = knnSplitter(splitterParams, LAYERTYPE_SPLITTER, myNet)

Conv1D = knnConv1D(totalNumRegions, LAYERTYPE_CONV1, myNet, a_Joiner_region_sizes, inumUnitsInPrevLayer_Pi = P_Splitter_region_sizes, regionRegularizers, regionLambdas)
Joiner = knnJoiner(joinerParams, LAYERTYPE_JOINER, myNet)
H_Layer1 = knnLayer(b, LAYERTYPE_HIDDEN, myNet, iactivation=k_linear, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0)
Output = knnLayer(1, LAYERTYPE_OUTPUT, myNet, iactivation=k_linear)


# init network by connecting up all layers ( so that the matrices are created the right size
myNet$connectLayers() # note: this resets the Weight matrices


myNet$learn(train_GWAS, train_y, test_GWAS, test_y, eval_test=T,eval_train=T, num_epochs=numEpochs, eta=learnRate,  eval_freq = freqTest)

# [1] "it: 20000  / Training prediction (r^2) : 0.52109  / Test  prediction (r^2) : 0.20056"
# [1] "it: 1000  / Training prediction (r^2) : 0.1233  / Test  prediction (r^2) : 0.082512"

# with no regularization
# [1] "it: 1000  / Training prediction (r^2) : 0.028853  / Test  prediction (r^2) : 0.00061633"

#yhat = myNet$forward_propagate(train_GWAS[[1]])
#myNet$backpropagate(yhat, train_y[[1]]) # compare it to the same minibatch's true Y
#myNet$update(num_samples_total = length(train_y[[1]]), eta = learnRate, minibatch_size = length(train_y[[1]]))

