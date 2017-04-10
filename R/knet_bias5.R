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

# STATIC vars
OUT_BINARY = "OUT_BINARY"
OUT_MULTICLASS = "OUT_MULTICLASS"
OUT_REGRESSION = "OUT_REGRESSION"

HIDDEN_ACT_SIG = "HIDDEN_ACT_SIG"
HIDDEN_ACT_RELU = "HIDDEN_ACT_RELU"
HIDDEN_ACT_SPLUS= "HIDDEN_ACT_SPLUS"

REGULARIZER_NONE = "REGULARIZER_NONE"
REGULARIZER_RIDGE = "REGULARIZER_RIDGE"
REGULARIZER_LASSO = "REGULARIZER_LASSO"



GRAD = "GRAD"
WEIGHT = "WEIGHT"
REG_COST = "REG_COST"


## GLOBAL functions


# activation functions
k_sigmoid = function (X, deriv=FALSE) {
  if ( deriv == FALSE) { 
    return( 1.0 / (1.0 + exp(-X)) )
    
  } else {
    return( k_sigmoid(X)*(1.0 - k_sigmoid(X)) )
  }
}

# R doesn't allow 'null' to be set as default for arguments which are meant to be functions
k_no_activation = function () { print("should never be called!")}

k_linear = function(X, deriv=FALSE) {
  if ( deriv == FALSE) {  return(X ) } else {
    
    # the derivative of a linear function is 1, but we need to make sure we return it in the right dimensions
    if (is.matrix(X)) { 
      return ( matrix(1.0, ncol = ncol(X), nrow = nrow(X)) )
      } else if ( is.vector(X) ) {
        return ( rep(1.0, length(X) ) )
    } else {return( 1.0 )  } # if its a scalar
  }
  
}

k_softmax_buggy = function (X, deriv=FALSE) {
  if ( deriv == FALSE) { 
  Output_Z = sum( exp(X) ) # the normalizing constant
  return ( exp(X) / Output_Z )
  # this is NEVER used
  } else { # the derivative of a softmax function is 1: http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
    if (is.matrix(X)) { 
      return ( matrix(1.0, ncol = ncol(X), nrow = nrow(X)) )
    } else if ( is.vector(X) ) {
      return ( rep(1.0, length(X) ) )
    } else {return( 1.0 )  } # if its a scalar
  }
  
}

logsumexp <- function (x) {
  y = max(x)
  y + log(sum(exp(x - y)))
}


k_softmax_apply <- function (x) {
  exp(x - logsumexp(x))
}

k_softmax = function (X, deriv=FALSE) {
  return ( t( apply(X, 1, k_softmax_apply ) ) )  # apply idiotically transposes its results: http://stackoverflow.com/questions/4140371/why-does-sapply-return-a-matrix-that-i-need-to-transpose-and-then-the-transpose
}




k_softplus = function(X, deriv=FALSE) {
  if ( deriv == FALSE) {  return( log( 1.0 + exp(X) ) ) } else {   
    return( k_sigmoid(X) ) }
 
}

k_leakyRELU = function (X, deriv=FALSE) {
  if ( deriv == FALSE) { 
  EX = (0.001 * X)
  return( pmax(X, EX) ) } else { 
    if (is.matrix(X)) { 
      return ( matrix(1.0, ncol = ncol(X), nrow = nrow(X)) )
    } else if ( is.vector(X) ) {
      return ( rep(1.0, length(X) ) )
    } else {return( 1.0 )  } # if its a scalar
  }

}

# given the true and predicted values in matrix format, it returns the mean % difference between them
calc_MAE = function (yhat, y) {
  total = 0.0
  
  for(i in 1: nrow(yhat)) { # go row by row
    prediction = yhat[i,]
    true = y[i,]
    index = which(true == max(true)) # find index of true value (IE in a 1 of K scenario this is the index of the highest value eg: (0,1,0) -> it is 2 )
    error = abs( prediction[index] - max(true) ) # absolut edifference between our predicted value and the truth
    error_perc = abs ( error / max(true) ) # express this as a % of the ideal value
    total = total + error_perc
  }
  return ( total  )
  # return ( total / nrow(yhat) )
}

# calculates multiclass classification accuracy (use for softmax output)
calc_Accuracy = function (yhat, y) {
  num_matches = 0.0
  for(i in 1:nrow(yhat)  ) {  # go row by row
    prediction = yhat[i,]
    true = y[i,]
    
    index_truth = which(true == max(true)) # find index of true value (IE in a 1 of K scenario this is the index of the highest value eg: (0,1,0) -> it is 2 )
    index_guess = which(prediction == max(prediction)) # find index of the guess
    if(index_truth == index_guess) { num_matches = num_matches+1 }
  }

  return(num_matches)
}

norm <- function(x) sqrt(sum(x^2))


# _______________________________________________________________


LAYER_SUBTYPE_INPUT = "LAYER_SUBTYPE_INPUT"
LAYER_SUBTYPE_OUTPUT = "LAYER_SUBTYPE_OUTPUT"
LAYER_SUBTYPE_HIDDEN = "LAYER_SUBTYPE_HIDDEN"



# creates an Artifical Neural Network
# layer_config: the architecture of the network specifing the number of nodes in each layer: [num_inputs,num_hidden1,..., num_hiddenlast,num_output]
# iminibatch_size: the number observations per iteration
knn <- setRefClass("knn",
                   
                   fields = list( 
                     layers =  "list",
                     #DEBUG_WeightGrads = "list",
                     DEBUG = "logical",
                     CAN_ADD_LAYERS= "logical",
                     num_layers =  "numeric"
                     
                   ),
                   
                   methods = list(
                     initialize = function () {
                       layers <<- list()
                       num_layers <<- 0
                       CAN_ADD_LAYERS <<- TRUE
                       
                       DEBUG <<- FALSE
                       #DEBUG_WeightGrads <<- list()

                     },
                     
                     addLayer = function (layer) {  # adds a layer to the neural networks list of layers
                       if(CAN_ADD_LAYERS) {
                         num_layers <<- num_layers+1
                         layers[[num_layers]] <<- layer 
                         print( paste("layer ",class(layer)[1]," is added to Network " ))
                       }
                     }, 
                     
                     connectLayers = function () {  # connects up each layer to its next, and initialises the Weighted layers with the correct dimension matrices
                       
                       # the first layer only has a next layer
                       layers[[1]]$initConnections( list(NEXT = layers[[2]]) )
                      
                       if(num_layers > 2)  { # if there  are any hidden layers
                         for(i in 2:(num_layers-1) ) { # go through all layers except last
                           
                           layers[[i]]$initConnections(  list(PREV= layers[[i-1]],NEXT = layers[[i+1]]) )
                         }

                       }
                       
                       
                       # connect Output
                       layers[[num_layers]]$initConnections( list(PREV = layers[[num_layers-1]])  ) # the last layer only has a prev layer
                       
                       print("All layers created, Neural Network ready")
                     },
                     
                     
                     # Main forward propagation algo: it usually return Sigma(XB), and feeds it int the next layer's input
                     forward_propagate = function (data) {  # data is the design matrix
                       
                       return(layers[[1]]$forward_propagate(data)) # this recursively calls all subsequent layers' forward propagate until the last (the output, which produces the yhat)
                     },
                     
                     
                     # calculates the cumulative errors (Delta) at each node, for all layers
                     backpropagate =  function(yhat, y) {
                       
                       # calculate first Error Delta directly from output
                       Error_Delta_Output = t( yhat - y ) 
                       layers[[num_layers]]$backpropagate(Error_Delta_Output) # call this on the last layer, which then will recursively call all the rest all the way back to the first
                     },
                     
                     
                     # goes through and updates each layer (for ones that have weights this updates the weights, for others this may do something else or nothing)
                     update = function(eta, minibatch_size, friction) {
                       # print(paste("friction is:", friction))
                       for (i in 2:(num_layers)) { # loop from the 2nd layer to the last ( the 1st layer is the input data, and that cannot be updated)
                         layers[[i]]$update(eta, minibatch_size, friction)
                       }
                     },
                     
                     
                     # train_X/Y (and test_X/Y): a list() of matrices, each element on the list being a minibatch matrix
                     learn = function(train_X, train_Y, test_X, test_Y, num_epochs=500, eta=0.05, eval_train=FALSE, eval_test=FALSE, eval_freq = 100,  friction = 0.0) {
                       print ( paste("Starting to train Neural Network for for num iterations:", num_epochs) )
                       
                       minibatch_size = nrow(  as.matrix(train_Y[[1]]) )
                       # initMatrices(minibatch_size)
                       
                       # num_samples_total = length(train_Y) * nrow(  as.matrix(train_Y[[1]]) )
                       
                      
                       if(ncol(  as.matrix(train_y[[1]]) ) == 1) { # if the thing we are trying to predict has only 1 column, then it is a regression problem
                         outPutType = OUT_REGRESSION
                       } else { outPutType = OUT_MULTICLASS }
                       
                       out_str = ""
                       # cosmetics: if we are classifying then we are evaluating via 'accuracy' where as if we are regressing, then we care about error
                       evaluation = "prediction (r^2)"
                       if( outPutType != OUT_REGRESSION ) { evaluation = "accuracy (%)" }
                       
                       
                       for (t in 1:num_epochs) { # iterate through all epochs
                         out_str = paste("it:", sep =" ", t)
                         
                         # 1) Complete an entire training cycle: Forward then Backward propagation, then update weights, do this for ALL minibatches in sequence
                         for ( i in 1:length(train_X)) { # go through all minibatches
                           Yhat = forward_propagate(train_X[[i]]) # grab current minibatch's X, and feed it through the system
                           backpropagate(Yhat, train_Y[[i]]) # compare it to the same minibatch's true Y
                           update(eta, minibatch_size, friction)
                         }
     
                         
                         if ( t %% eval_freq == 0) { # only evaluation fit every 100th or so iteration, as that is expensive
                           
                           # 2) Diagnostics (optional)
                           # if we want to compare our error against the training values
                           if (eval_train == TRUE ) {
                             # in case y is a col vector (like for regression), then nrow would return 0, so we have to cast it as a matrix
                             N_train = length(train_Y) * nrow(  as.matrix( train_Y[[1]]) )  # the number of total training observations, is the number of minibatches (length of the list of matrices itself), * the number in each minibatch
                             totals = 0                                      # we assume that each minibatch has the same length here
                             
                             for ( i in 1:length(train_X)) { # go through all minibatches
                               yhat = forward_propagate(train_X[[i]]) # grab current minibatch's X, and calculate the prediction
                               
                               # depending on if we are in a classification or regression problem, we evaluate performance differently
                               if( outPutType != OUT_REGRESSION ) {  currentRate = calc_Accuracy(yhat, train_Y[[i]] ) 
                               } else {
                                 # evaluate via mean average error
                                 # residualSQ = (yhat - train_Y[[i]])^2  # residual Squared (Yhat - Y)^2
                                 #currentRate = as.numeric( sqrt( sum(residualSQ) )  ) # total error is the error so far plus the sum of the above
                                 
                                 # evaluate via r^2
                                 currentRate = cor(yhat , train_Y[[i]])^2
                                 N_train = length(train_Y) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
                                 
                               }
                               
                               totals = totals +currentRate # sum in all minibatches
                             }
                             
                             out_str =  paste(out_str, " / Training",evaluation,":", format(totals/N_train, digits=5), sep =" ")
                           }
                           
                           # if we want to compare our predictions against the validation set
                           if (eval_test == TRUE) {
                             
                             N_test = length(test_Y) * nrow(  as.matrix(  test_Y[[1]]  )   ) # in case y is a col vector (like for regression), then nrow would return 0, so we have to cast it as a matrix
                             totals = 0
                             
                             for ( i in 1:length(test_X)) { # go through all minibatches
                               yhat = forward_propagate(test_X[[i]]) # grab current minibatch's X, and calculate the output
                               
                               # depending on if we are in a classification or regression problem, we evaluate performance differently
                               if( outPutType != OUT_REGRESSION ) {  currentRate = calc_Accuracy(yhat,  test_Y[[i]] ) 
                               } else {
                                 
                                 # evaluate via mean average error
                                 #residualSQ = (yhat - test_Y[[i]])^2  # residual Squared (Yhat - Y)^2
                                 #currentRate =  as.numeric( sqrt( sum(residualSQ) ) ) # total error is the error so far plus the sum of the above
                                 
                                 # evaluate via r^2
                                 currentRate = cor(yhat , test_Y[[i]])^2
                                 N_test = length(test_Y) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
                               }
                               
                               totals = totals +currentRate
                               
                             }
                             out_str =  paste(out_str, " / Test ",evaluation,":", format(totals/N_test, digits=5), sep =" ")
                           }
                         }
                         print ( out_str )

                       }
                       
                     },
                     
                     
                     
                     ####### Debugger functions: used for numerical Gradient checking
                     # replaces the weights in the network from an external source: a 1D array (IE the same format as that is returned by getWeightsAsVector() )
                     setWeights = function(allWeights) {

                       for (i in 1:(num_layers)) { # go through all layers except first, as that cannot have weights
                         allWeights = layers[[i]]$addDebugData(allWeights) # if layer has weights, then this function takes as many as needed to fill up layer weight matrix, adds them , and removes them from the list before passing back what is left
                       }
                     },
                     
                     # returns the Sum of Squared errors ( IE this is the Cost function)
                     regressionCost_SumSquaredError = function (data, y) {   #
                       #Compute cost for given X,y, use weights already stored in class.
                       yHat = forward_propagate(data)
                       batch_num_samples = nrow(layers[[1]]$Output_Z)
                       J =  0.5*  sum((y-yHat)^2) / batch_num_samples + getRegularizerCostTerms()
                       return(J)
                     },
                     
                     multiclassCost_softMax = function(data, y) {
                       yHat = forward_propagate(data)
                       batch_num_samples = nrow(layers[[1]]$Output_Z)
                       J =  -sum(  y * log(yHat))  / batch_num_samples + getRegularizerCostTerms()    
                       return(J)
                     },
                     
                     # goes through and retreives the regularizer cost terms from all layers (Which have it)
                     getRegularizerCostTerms = function () {
                       
                       allTerms = vector()
                       for (i in 1:(num_layers)) { # go through all layers, and get their weights 
                         
                         allTerms = layers[[i]]$getDebugInfo(allTerms, type = REG_COST) # if layer has weights, it adds it into the array, if not just simply returns the orig
                       } 
                       return(sum (allTerms) )
                     },
                     
                     
                     getCurrentWeightGradients = function(data, y) {
                       minibatch_size = nrow(  as.matrix(y) )
                       yHat = forward_propagate(data)
                       backpropagate(yHat, y) # we need of backpropagate also, as we need to know the CURRENT error, IE that is resulted from the weights that we have now (as the errors we have atm are the errors due to the previous iteration)
                       origDEBUG = DEBUG
                       DEBUG <<- TRUE # temporarily set this, so that the next function saves the weight gradients
                       update(eta=0, minibatch_size, 0.0) # this calculates the current weight GRADIENTS into an array, without actually updating them ( as eta is 0)
                                                                        # must disable Friction, otherwise gradient checks will always fail
 
                       allWeightGrads = vector()  # allWeightGrads = c(1,1)
                       for (i in 1:(num_layers)) { # go through all layers, and get their weight gradients
                         allWeightGrads = layers[[i]]$getDebugInfo(allWeightGrads, type = GRAD) # if layer has weight grads, it adds it into the array, if not just simply returns the orig
                      
                         } 
                       DEBUG <<- origDEBUG # reset
                       return(allWeightGrads)
                     },
                     
                     
                     # gets the current weights across all layers in a 1D vector
                     getWeightsAsVector = function () {
                       
                       allWeights = vector()
                       for (i in 1:(num_layers)) { # go through all layers, and get their weights 
                         
                         allWeights = layers[[i]]$getDebugInfo(allWeights, type = WEIGHT ) # if layer has weights, it adds it into the array, if not just simply returns the orig
                       } 
                       return(allWeights)
                     },
                     
                     
                     gradientCheck = function (data, y) {
                       if(ncol(  as.matrix(y) ) == 1) { # if the thing we are trying to predict has only 1 column, then it is a regression problem
                         outPutType = OUT_REGRESSION
                       } else { outPutType = OUT_MULTICLASS }
                       
                       
                       weightsOriginal = getWeightsAsVector()  # gets all weights
                       # init some empty vectors same length as weights
                       
                       numgrad = c( rep( 0, length(weightsOriginal))) # the numerical approximation for the derivatives of the weights
                       perturb = c( rep( 0, length(weightsOriginal))) # perturbations: these are #1-of-k' style, where we have 0 for all else, except for the current
                       e = 1e-4  # the perturbation
                       
                      # num_samples_total = nrow(data) # the total number of samples are the same as the minibatch size, as gradient checks are only performed on signle minibatches
                       # the costfunction differs based on if it is a regression NN or a multiclass classification
                       costFunction= NULL
                       if( outPutType == OUT_REGRESSION) {
                         costFunction =  .self$regressionCost_SumSquaredError # must use the '.self' otherwise we cannot get a reference to the function as a variable
                       } else {
                         costFunction =  .self$multiclassCost_softMax
                       }
                       
                       for (  p in 1:length(weightsOriginal) ) { # go through each original weight
                         #Set perturbation vector
                         perturb[p] = e   # add a slight difference at the position for current weight
                         
                         # here we slightly change the current weight, and recalculate the errors that result ( vec1 +1, creates a new vector instance without modifying the original)
                         setWeights(weightsOriginal + perturb) # add the changed weights into the neural net (positive offset)
                         loss2 = costFunction(data, y) # get squared error: IE X^2  (IE this is the 'cost')
                         setWeights(weightsOriginal - perturb) # add the changed weights into the neural net (negative offset)
                         loss1 = costFunction(data, y) # get squared error: IE X^2  (IE this is the 'cost')
                         
                         #print(paste("num_samples_total is: ", num_samples_total, " // lengths are for weightsOriginal:", length(weightsOriginal), "/ loss2:", length(loss2), "/ loss1", length(loss1) )) 
                         
                         #Compute Numerical Gradient
                         numgrad[p] = (loss2 - loss1) / (2*e) # apply 'manual' formula for getting the derivative of X^2 -> 2X
                         
                         #Return the value we changed to zero:
                         perturb[p] = 0 # we do this as in the next round it has to be 0 everywhere again
                       }
                       #Return Weights to original value we have saved earlier:
                       setWeights(weightsOriginal)
                       
                       return(numgrad)
                     }
                   )
                   
)




#This only exists in R, as we need class fields that reference 'knnBaseLAyer' in the knnBaseLayer class itself and R doesn't like that
knnDummyLayer <- setRefClass("knnDummyLayer", 
         fields=list(
           params="numeric",
           isNull="logical",
           parentNetwork="knn"
         ),
         
         methods=list(
           initialize = function (iparams = -1, iparentNetwork = knn()) {
             params <<- iparams
             isNull <<- TRUE  # R is retarded, it instantiates this layer, whenever it is just referenced in prevLayer = "knnDummyLayer" (with default values), so in order to check i instance really we need to have this dummy variable...
             parentNetwork <<- iparentNetwork
           },
           
           generateOutput = function(input, ...) {},
           forward_propagate = function(input, ...) {},
           calcError = function(input,...) {},
           backpropagate = function(input,...) {},
           update = function(...) {},
           initConnections = function( prevNext = list() ) {  },
           getDebugInfo = function (dataSoFar, type,...) { return(dataSoFar) }, # the base method still needs to return the passed in data if nothing else
           addDebugData = function(allWeights) { return(allWeights) }  # the base method still needs to return the passed in data if nothing else
         )
)

# base class for all knn layers: should never be directly instantiated
knnBaseLayer <- setRefClass("knnBaseLayer",
      fields=list(
        #params="numeric",
        #parentNetwork="knn",
        prevLayer = "knnDummyLayer", # this should be 'knnBaseLayer', but R cannot reference this type of class in the class definition, so we have to use a 'dummy' class
        nextLayer = "knnDummyLayer"
        ),
      
       contains="knnDummyLayer",

      methods=list(
        initialize = function (iparams, iparentNetwork) {
          callSuper(iparams, iparentNetwork)
          # params <<- iparams
          # parentNetwork <<- iparentNetwork
          
          isNull <<- FALSE # R: only for an instantiated layer, we set this to FALSE to let others know this really exists...

          
          # add this layer to its parent Neural Network's list of layers
          parentNetwork$addLayer(.self)
        },
        
        # produces output from the layer's data, this is used by forward_propagate
        generateOutput = function(input, ...) { 
        },
        
        # passes along the output of this layer to the next, if any 
        forward_propagate = function(input, ...) {
          output = generateOutput(input)
          
          if( nextLayer$isNull ) { # if there is no next layer, IE this is the last one the Output
            return(output)
          }else { # if there are any next layers still, we recursively call them
            # send output from this layer to the next
            
            return( nextLayer$forward_propagate(output) )
          }
        },
        
        
        # computes the Error Delta of the current layer, based off from the error passed back from the layer after this during backpropagation
        calcError = function(input,...) { 
        },
        
        # passes errors backwards from the output onto preceding layers
        backpropagate = function(input,...) { # this receives the Error_Delta from the layer after this
          
          if( prevLayer$isNull == FALSE) {# if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            # generate output rom this layer
            error_D = calcError(input) # this is USUALLY really just the Error_D, but the Joiner layer for instance, passes along the weights of the next layer too
            return( prevLayer$backpropagate(error_D) )
          } # else: if there are no prev layers then stop backpropagating ... IE we reached the INPUT layer
        },
        
        
        # generic update function, (it usually updates weights for layers that have them)
        update = function(...) {
        },
        
        # Lets each layer know about its neighbours: the previous and next layer in the stack: this is called once the entire network has been set up
        initConnections = function( prevNext = list() ) {
          # check previous layer
          if( is.null(prevNext$PREV) == FALSE) { prevLayer <<- prevNext$PREV }  # R is retarded, you cannot assign a NULL value to a var that expects it to be a class (even thoug it is initialised as NULL...)
         
          # check next layer
          if( is.null(prevNext$NEXT) == FALSE) { nextLayer <<- prevNext$NEXT }
        }
        
      )
)


# specialised layer type used by Convolutional topologies:  takes input, and splits it into a set of predefined regions, and passes these along to the next layer
# params: [totalNumUnitsinLayer, lastUnitInRegion1, lastUnitInRegion2,...,, lastUnitInRegionn]
knnSplitter <- setRefClass("knnSplitter",
          fields=list(
            regions="list", # list of arrays describing the regions: where [0] is the start, and [1] is the last unit in a region
            numRegions="numeric",
            Error_D = "matrix",
            Input_S = "matrix"
          ),
          
          contains="knnBaseLayer",
          
          methods=list(
            initialize = function (iparams, iparentNetwork) {
              callSuper(iparams, iparentNetwork) # call init on parent

              # parse params into regions
              numRegions <<- length(params) -1  # 1st element sets the number of regions we will need, so the number of regions will be length of the params-1
                                      # params[1] = a = # units in current layer;  need to keep the 1st element being the total number of units in this layeer (including in all regions), so that other layers can access this information the same way as for any other layers)
              regions <<- list()
              regionStart = 1
              for( i in 2:length(params)) { # loop from 2nd element, where each element is the LAST predictor in matrix
                regionEnd = params[i]
                
                regions[[(i-1)]] <<- c(regionStart, regionEnd)
                regionStart = regionEnd+1 #the next region's start is just the one after this ended
              }
            },
            
            
            # Splitter splits an incoming Input data into the parameters' predefined set of regions
            generateOutput = function(input, ...) {
             
              Input_S <<- input # save input
              
              subMatrices = list()
              for (i in 1:numRegions) { # go through all regions we are meant to be having
                regionStart = regions[[i]][1]
                regionEnd = regions[[i]][2]
                subMatrices[[i]] = input[,regionStart:regionEnd] # cut the Input, into submatrices in their cols (as we are splitting by predictors)
                
              }
              return(subMatrices) # return the splits
            },
            
            
            # Splitter performs the opposite function for backpropagation: it Joins an incoming list error Delta submatrices into a single matrix
            calcError = function(input, ...) {
              
              # input here is a list()  of all the Error_D s from the next layer (the Convolution1D)
              new_Error_D = NULL
              for(i in 1:length(input)) {
                new_Error_D = rbind(new_Error_D, input[[i]] ) # as the Di submatrices s are transposed, so we join them by adding them under each other
              }
              
              Error_D <<-new_Error_D
              return(Error_D)
            }
          )
)


# specialised layer type used by Convolutional topologies:  takes input, and joins them up, and passes these along to the next layer
# params: [totalNumRegions, lastUnitInRegion1, lastUnitInRegion2,...,, lastUnitInRegionn]
knnJoiner <- setRefClass("knnJoiner",
                           fields=list(
                             regions="list", # list of arrays describing the regions: where [0] is the start, and [1] is the last unit in a region
                             numRegions="numeric",
                             Error_D = "matrix", # this refers to the ENTIRE Error_D of the layer after this 
                             nextLayerWeights = "list", # list of the weights that were split into regions of the layer after this
                             # nextLayerWeight_bias = "numeric", # this is unused, we don't transfer this, although could be sent forward to Splitter, which could then add this into the D it joined... (but as bias weights are generally not used to calculate the Error Deltas this is fine)
                             Output_Z = "matrix" , # 
                             prevLayerOutputs = "list" # the outputs of a Conv layer
                             
                           ),
                         
                         contains="knnBaseLayer",
                           
                           methods=list(
                             initialize = function (iparams, iparentNetwork) {
                               callSuper(iparams, iparentNetwork) # call init on parent
                               
                               # parse params into regions
                               numRegions <<- length(params) -1 # the number of regions, is the length of the array minus the 1st element: 
                                                        # params[1] total number of units in the previous layer (this may NOT be the number of regions, if there are multiple units that represent each region), need to keep this reflecting his info, so that we can access it outside the same way as any other layer
                               
                               regions <<- list()
                               regionStart = 1
                               for( i in 2:length(params)) { # loop from 2nd element, where each element is the LAST predictor in matrix
                                 regionEnd = params[i]
                                 
                                 regions[[(i-1)]] <<- c(regionStart, regionEnd)
                                 regionStart = regionEnd+1 #the next region's start is just the one after this ended
                               }
                               
                             },
                             
                             
                             # Joiner joins an incoming list of submatrices; outputs from a convolutional layer, into a single output
                             generateOutput = function(input, ...) { # Joiner gets a list of outputs, that it will merge
                               prevLayerOutputs <<- input # save input
                               
                               output = NULL
                               for(i in 1:length(prevLayerOutputs)) {
                                 output = cbind(output, prevLayerOutputs[[i]] ) # as we split by predictors (cols) we join them by placing them next to each other
                               }
                               Output_Z <<- output # save output too
                               return(Output_Z)
                             },
                             
                             
                             # Joiner performs the opposite function for backpropagation: it splits the incoming Weights, and passes along the entire Error Delta
                             calcError = function(input, ...) {

                               Error_D <<- input #  save the entie Error_D  of next layer, which is NOT split, but will be used to multiply each weight by
 
                               # get the Weight of the next layer, this will be needed by the Conv1D layer that follows this
                               weightOfNextLayer = nextLayer$Weights_W
                               
                               nextLayerWeights <<- list()
                               for (i in 1:numRegions) {
                                 regionStart = regions[[i]][1]
                                 regionEnd = regions[[i]][2]
                                 nextLayerWeights[[i]] <<- weightOfNextLayer[regionStart:regionEnd,] # cut the submatrix, as the Error Deltas are transposed at this stage we spit by the rows, to get the effecot of splitting by the predictors
                                 
                               }
                               return( list(Error_D, nextLayerWeights) ) # the error itself has not changed, but we also pass along the next layer's weights split into a list of submatrices for each region
                               
                             }
                           )
)


# specialised layer type used by Convolutional topologies:  takes input, and joins them up, and passes these along to the next layer
# params: [totalNumRegions, lastUnitInRegion1, lastUnitInRegion2,...,, lastUnitInRegionn]
# [numSamplesinBatch(n), currentLayerUnits(a), nextLayerUnits(b)]
knnConv1D <- setRefClass("knnConv1D",
                         fields=list(
                           regions="list", # list of knn layers, 1 for each region
                           numRegions="numeric",
                           Error_D = "list", # this refers to a list of 
                           numUnitsInregions ="numeric", # how many units are in each region
                           numUnitsInPrevLayer_Pi ="numeric",  # vector of each of the number of units,  (IE columns/predictors/weights) in previous layer's regions
                           regionRegularizers = "character", # list of the type of regularizers for each region (layer)
                           regionShrinkageParams ="numeric", # each region is allowed to have its own lambda
                           Input_S = "list" # list of matrices of outputs of the previous layer
                         ),
                         
                         contains="knnBaseLayer",
                         
                         methods=list(
                           initialize = function (iparams, iparentNetwork, inumUnitsInregions, inumUnitsInPrevLayer_Pi, iregionRegularizers, iregionShrinkageParams) {
                             callSuper(iparams, iparentNetwork) # call init on parent
                             
                             # parse params into regions
                             numRegions <<- params[1] # params only has 1 element for Conv1D
                             regions <<- list()
                           
                             numUnitsInregions <<- inumUnitsInregions
                             numUnitsInPrevLayer_Pi <<- inumUnitsInPrevLayer_Pi
                             regionRegularizers <<- iregionRegularizers
                             regionShrinkageParams <<- iregionShrinkageParams
 
                          #dummyKnn = knn() # as all layers must belong to a Knet, we need to create a dummy one here (this is a Hacky fix...)
                             parentNetwork$CAN_ADD_LAYERS <<- FALSE # as all layers must belong to a Knet, (as they derive overall network wide properties such as DEBUG), but nested layers inside Conv1Ds should not be added to the main network flow, so we disable that
                             for(i in 1:numRegions){ 
                              
                               # create layer
                               regions[[i]] <<-  knnLayer(numUnitsInregions[i], parentNetwork, LAYER_SUBTYPE_HIDDEN, iactivation=k_linear, regularizer = regionRegularizers[i], shrinkageParam = regionShrinkageParams[i])
                              
                              # need to set up Connections of each layer that represents a region, but without having to rely on prevLayer. as there won't be', so we directly set the weight matrix' dims
                               # will need to make sure that prev/next layer won't exist
                               regions[[i]]$initWeightMatrix(numUnitsInPrevLayer_Pi[i])

                             }
                             parentNetwork$CAN_ADD_LAYERS <<- TRUE # re enable this
                           },
                           
                          
                           # goes through a list of mini Weighted layers, and collects their output, which is then sent forward
                           generateOutput = function(input, ...) { # Convolution1D gets a list of inputs (Xi) from a Splitter
                             Input_S <<- input # save away the list of subMatrices
                             
                             convOutput = list()
                             for(i in 1:numRegions) { # go through each region's layer, and make each output its bit
                               convOutput[[i]] = regions[[i]]$forward_propagate(input[[i]])
                             }

                             return(convOutput)
                           },
                           
                          
                          # update here delegates the updating of the weights to each of its regions' layers
                          update = function(eta, num_samples_minibatch, friction) {
                            for(i in 1:numRegions) { # go through each region's layer, and force them to 
                              regions[[i]]$update(eta, num_samples_minibatch, friction)
                            }
                          },
                          
                          
                           # Conv1D gets a single matrix for Error_D and a list of weights, which come from the next FC(Fully Connected) layer (split into matching parts via the Joiner)
                          calcError = function(input, ...) {
                             
                            Error = input[[1]]
                            nextLayerWeights = input[[2]]
        
                            
                            convOutput = list()
                             for (i in 1:numRegions) { 

                              # each region's layer calculates its D, from the previous D, and the relevant submatrix of W
                               convOutput[[i]] = regions[[i]]$calcError(Error, nextLayerWeights[[i]] ) # this performs Di_part = W_NEXT * t(D_FC_all) * Fp
                                                                          # here we override the weights, by passing one in directly (this is kindof hacky)
                             }

                             Error_D <<- convOutput # save away ALL the Error_Ds of all the regions
                             return(Error_D) # return the splits
                             
                           },
                           
                           
                           # returns all weights in ALL the region's layers into a 1D array
                           getWeightsAsVector = function () { # a Conv layer just goes through ALL of its mini layer's weights
                             allWeights = vector()
                             for (i in 1:numRegions) { 
                               allWeights = c(  allWeights, regions[[i]]$getWeightsAsVector() )
                             }

                             return(allWeights)
                           },
                          
                          # returns all weight Gradients in ALL the region's layers into a 1D array
                          getWeightsGradsAsVector = function () { # a Conv layer just goes through ALL of its mini layer's weights
                            allWeightGrads = vector()
                            for (i in 1:numRegions) { 
                              allWeightGrads = c(  allWeightGrads, regions[[i]]$DEBUG_WeightGrads )
                            }
                            
                            return(allWeightGrads)
                          },
                          
                          
                          getRegularizerCostTerm = function () { # a Conv layer just goes through ALL of its mini layer's regularizer cost terms
                            allCostTerms = vector()
                            for (i in 1:numRegions) { 
                              allCostTerms = c(  allCostTerms, regions[[i]]$getRegularizerCostTerm() )
                            }
                            
                            return(allCostTerms)
                          },
                          
                          # concats the weights of this layer (if it has any), into the ones we got so far
                          getDebugInfo = function (dataSoFar, type = WEIGHT) {

                              if(type == WEIGHT ) {  #if we are NOT looking for weight gradients, then we want the weights
                                return( c(dataSoFar, getWeightsAsVector()) )
                              } else if (type == GRAD  ) { # if we do want the gradients then we use the ones saved during debug
                                return( c(dataSoFar, getWeightsGradsAsVector() ) )
                              } else { # if we want  the regularizer cost terms
                                return( c(dataSoFar, getRegularizerCostTerm() ) ) 
                              }
                            
                          },
                          
                          
                          
                          addDebugData = function(allWeights) {
                            
                            for (i in 1:(numRegions)) { # go through all layers/regions
                              allWeights = regions[[i]]$addDebugData(allWeights) # if layer has weights, then this function takes as many as needed to fill up layer weight matrix, adds them , and removes them from the list before passing back what is left
                            }
                            
                            return(allWeights)
                          }
                          
                         )
)


# the main 'Weighted' neural network layer, used by all FC (Fully Connected) layers
# params: a scalar, it is simply just the number of units in his layer
knnLayer <- setRefClass("knnLayer",
     fields = list(
       subtype="character", # as we have 3 different subtypes, for Input/Output and Hidden, we need to differentiate it via this flag
       layer_numUnits= "numeric", # number of neurons in layer
       biasEnabled = "logical",
       activation  = "function", 
       Weights_W = "matrix",   # weights are stored as 'incoming' IE they refer to the weights between this layer and the one before
       Weights_bias = "numeric", 
       Output_Z = "matrix", 
       Error_D = "matrix", # moved to the baseclass
       Derivative_Fp = "matrix",
       Momentum = "matrix",
       Bias_Momentum = "numeric",
       Input_S = "matrix",
       regularizer = "character",
       DEBUG_WeightGrads = "matrix",
       Lambda =  "numeric"
       ),
     
     contains="knnBaseLayer",
     
     methods = list(
       initialize = function (iparams, iparentNetwork, isubtype, iactivation=k_no_activation, ibiasEnabled = TRUE, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0) { 
         callSuper(iparams, iparentNetwork) # call init on parent
         regularizer <<- regularizer
         subtype <<- isubtype
         Lambda <<- shrinkageParam
         biasEnabled <<- ibiasEnabled
         layer_numUnits <<- params[1] # here params has just 1 element
         
         # The activation function is an externally defined function (with a derivative)
         activation <<- iactivation

         Output_Z <<-  matrix(NA) # Z is the matrix that holds output values
         Weights_W <<- matrix(NA) # W is the INCOMING weight matrix for this layer
         Input_S <<- matrix(NA) # S is NULL matrix that holds the inputs to this layer
         Error_D <<- matrix(NA) # D is the matrix that holds the deltas for this layer
         Derivative_Fp <<- matrix(NA) # Fp is the matrix that holds the derivatives of the activation function applied to the input
         DEBUG_WeightGrads <<- matrix(NA) # holds the debug weights for this layer (only used if parent layer has DEBUG == TRUE)
         
         print( paste("layer ",subtype," is regularized by: " , regularizer, "/ its params are", params))
         }, 
       
       
      # sets up relationships to neighbouring layers, and if we have previous layer (IE it is not an input or a nested minilayer in a Conv), then we can init the weight matrix to the correct dimension
       initConnections = function( prevNext = list() ) { 
         callSuper(prevNext)
    
        # Weights can be initialised only by knowing the number of units in the PREVIOUS layer (but we do NOT need to know the size of the minibatches (IE n), as Weight matrix' dimension does not depend on that
         if ( prevLayer$isNull == FALSE) { # IE the we are testing if, subtype != LAYER_SUBTYPE_OUTPUT
           prevLayer_size = prevLayer$params[1] # find out how big the next layer is 
           initWeightMatrix(prevLayer_size)
         }
       },
     
     
       # (this is usually called from initConnections(), but this might also be called directly from outside, by the conv layer for example)
       initWeightMatrix = function(prevLayer_size) {
 #  print(paste("initWeightMatrix for subtype:", subtype, " / prev layer has size:",prevLayer_size))
         # Weights are sized as: rows: number of units in previous layer, cols: number of units in current layer (IE the minibatch size doesnt matter)
         Weights_W  <<- matrix( rnorm(prevLayer_size * layer_numUnits, sd=sqrt(2.0 / prevLayer_size) ),  nrow = prevLayer_size, ncol=layer_numUnits  ) # modified xavier init of weights for RELU/softplus (this also has the effect of basic L2 regularization)
         Momentum <<- matrix(0.0, nrow = nrow(Weights_W),  ncol = ncol(Weights_W) ) # stores a 'dampened' version of past weights (IE its an older version of the above with the same dimensions)
      
         if (biasEnabled == TRUE) { # if we have bias/intercept, we add a row of Weights that we keep separate
           Weights_bias <<- rnorm(ncol(Weights_W)                    , sd=sqrt(2.0 / prevLayer_size) )
           Bias_Momentum <<- rep(0, length(Weights_bias)) # stores a 'dampened' version of past weights  for intercepts
         }
       },
     
     
       # performs the addition of the bias terms effect  onto the output
       add_bias = function(ZW) {
         if(biasEnabled == FALSE) { return(ZW)} # if bias wasn't enabled in the first place we just return the orig
         
         ZWb = sweep(ZW,2,Weights_bias,"+") # we simply add the bias to every row, the bias is implicitly multipliedby 1
         return(ZWb) 
       },
       
     
     # as knn layer can be 1 of 3 subtypes, we have to produce an output for forward propagation differently
       generateOutput = function(input,...) {
         # if its the first layer, we just return the data that was passed in
         if (subtype == LAYER_SUBTYPE_INPUT) {
           Output_Z <<- input
           return( Output_Z  ) 
       
          # if its NOT an input, then all FC layers will have weights (even Output, as weights are now stored as 'incoming' weights between this and prev layer)
         } else { 
           Input_S  <<- input # save this away for later access
           Output_Z <<- Input_S %*% Weights_W  # the output is constructed by first multiplying the input by the weights
           }
         
         Output_Z <<- add_bias(Output_Z) # we then add the intercept (the liner predictor is now complete)
         
         # non Output subtype layers need to figure out the rate of change in the output (this is redundant, if we are just making predictions)
         if (subtype != LAYER_SUBTYPE_OUTPUT) {  Derivative_Fp <<- t( activation(Output_Z, deriv=TRUE) ) } # this is transposed, as D is also transposed during back propagation
         
         # output is completed by passing the linear predictor through an activation (IE we squash it through a sigmoid)
         Output_Z <<- activation(Output_Z) # if its a hidden layer (or output), we need to activate it
         
         return ( (Output_Z) )
         
       },
     
     
       # computes the accumulated error up until this layer, this usually happens by 
     # each layer's Delta, is calculated from the Delta of the Layer +1 outer of it, and this layer's Weights scaled by the anti Derivative Gradient
       calcError = function(input, customWeight = NULL,...) { 
         if (subtype == LAYER_SUBTYPE_OUTPUT) { 
           #print(paste("Output Error row/col", nrow(input), "/", ncol(input) ))
          Error_D <<-  input # this is stored as Transposed D ()
        } else { # 'input' here basically refers to 'nextLayer$Error_D'
   
          #    to get the current layer's ErrorDelta, we need the NEXT layer's weight, this can be usually directly accessed
          # , except if this is a 'mini' layer nested in a convolutional layer, in that case it had to be directly passed in as a 'customWeight'
          if( is.null(customWeight) ) { weightToUse = nextLayer$Weights_W} else {weightToUse = customWeight}
         # print(paste("weightToUse row/col:", nrow(weightToUse), "/", ncol(weightToUse), "// input row/col", nrow(input), "/", ncol(input) ))
          Error_D <<- (weightToUse %*% input) * Derivative_Fp # the current layer's Delta^T = (NextWeights_W * D_next^T) *Schur* F^T  
          }
         return(Error_D)
       },
     
     
       # updates the Bias Weights separately ( if enabled)
       update_bias_weights = function(num_samples_minibatch, eta, friction) {
         if(biasEnabled == TRUE) {
           
           W_bias_grad =  rowSums(Error_D)  # via implicit multiplication of D by 1 ( IE row sums)
           W_bias_grad = W_bias_grad / num_samples_minibatch
           
           # add Mometum (velocity)
           Bias_Momentum <<- friction * Bias_Momentum - (eta*W_bias_grad)
           
           # bias gradients are NOT regularized, so we just apply them as is
           Weights_bias <<-  Weights_bias + Bias_Momentum
         }
       },
       
     
       # updates the weights (including intercept) for current layer by calculating a 'change in weights': this is basically a scaled version of Error*Input
       # we scale by: learning rate(eta) and number of samples in minibatch ( to ensure consistent updates between different minibatch sizes)
       # this is the stage where we add regularizers too: we scale those by the 'total number of samples in ALL minibatches' (kindof redundant)
       # finally entire update is applied via 'momentums' we build up acceleration from previous updates, so if we keep moving int othe same direction then we can go(IE learn) faster and faster
       update = function(eta, num_samples_minibatch, friction) {
         
         if(is.na(Weights_W)[1] == FALSE) { # if it has any outgoing weights (IE not an input that feeds into a splitter)
           W_grad = t(Error_D %*% Input_S) # the weight changes are (D_next^t * Input_current) and transposed back so dims match
           W_grad = W_grad / num_samples_minibatch # normalise by N, to keep the same as the cost function
           
           regularizers = getRegularizerGrad()
           
           W_grad = W_grad + regularizers
           
           if (parentNetwork$DEBUG == TRUE) { # if we are in Debug mode, we want to save away the current Weight gradients
             DEBUG_WeightGrads <<- W_grad
           }
           
           # add Mometum (velocity)
           Momentum <<- friction * Momentum - (eta*W_grad) # total update is: (dampened) pastUpdates + current update
           
           Weights_W <<- Weights_W +  Momentum
           
           # update bias/intercept terms separately (if needed)
           update_bias_weights(num_samples_minibatch, eta,friction)  # bias weights are not trained ...for now (I think this has to do with the fact that NNs have universal approx ability, as long as the function doesn't go through 0)
         }
       },
     
     
     # the derivative of the regularizer term: used when updating weights 
     # (have to normalise by the total number of samples in the total dataset, otherwise regularization would grow with smaller minibatches)
   #  getRegularizerGrad = function(num_samples_total) {
       
   #    if ( regularizer == REGULARIZER_NONE ) {
   #      return(0.0)
   #    } else if (regularizer == REGULARIZER_RIDGE ) {
  #       return ( Lambda/num_samples_total * Weights_W )
   #    } else { # LASSO
   #      return ( Lambda/num_samples_total * sign(Weights_W ) ) }  # this will cause the numerical check to fail.. it will still be small but much bigger than with L2 or without, this is probably due to the funky L1 derivative at 0
   #  },
   
     getRegularizerGrad = function() { # no longer normalise by total number of sampels, as we assume that the L2 norm has been found by EMMA to be precisely correct for the given sample size
       
       if ( regularizer == REGULARIZER_NONE ) {
         return(0.0)
       } else if (regularizer == REGULARIZER_RIDGE ) {
         return ( Lambda * Weights_W )
       } else { # LASSO
         return ( Lambda * sign(Weights_W ) ) }  # this will cause the numerical check to fail.. it will still be small but much bigger than with L2 or without, this is probably due to the funky L1 derivative at 0
     },
     
     
       # the cost of the regularizer term: only used for  gradient checking
       getRegularizerCostTerm = function() { # no longer normalise by total number of sampels, as we assume that the L2 norm has been found by EMMA to be precisely correct for the given sample size
         #num_samples = layers[0].Output_Z.shape[0]
         
         if (regularizer == REGULARIZER_NONE) {
           return(0.0) 
         } else if (regularizer == REGULARIZER_RIDGE) {
           return ( (Lambda * 0.5) * sum( getWeightsAsVector()^2 ) ) 
         } else { # LASSO
           return ( (Lambda * 0.5) * sum( abs( getWeightsAsVector() ) ) ) }
       },
       
      # the cost of the regularizer term: only used for  gradient checking
    #  getRegularizerCostTerm = function(num_samples_total) {
        #num_samples = layers[0].Output_Z.shape[0]
        
    #    if (regularizer == REGULARIZER_NONE) {
    #      return(0.0) 
    #    } else if (regularizer == REGULARIZER_RIDGE) {
    #      return ( (Lambda/num_samples_total * 0.5) * sum( getWeightsAsVector()^2 ) ) 
    #    } else { # LASSO
    #      return ( (Lambda/num_samples_total * 0.5) * sum( abs( getWeightsAsVector() ) ) ) }
    #  },
  

       # returns the weights in in the current layer concated into a 1D array (used oly by getRegularizerCostTerm()  )
       getWeightsAsVector = function () {
         return( c(Weights_W) )
       },
       
     
     # concats the weights of this layer (if it has any), into the ones we got so far
     getDebugInfo = function (dataSoFar, type = WEIGHT) {

       
       if(is.na(Weights_W)[1] == FALSE) {
         if(type == WEIGHT) {  #if we are NOT looking for weight gradients, then we want the weights
           return( c(dataSoFar, getWeightsAsVector()) )
         } else if (type == GRAD){ # if we do want the gradients then we use the ones saved during debug
           return( c(dataSoFar, DEBUG_WeightGrads) )
         } else {
           return( c(dataSoFar,  getRegularizerCostTerm()) )
         }
         
       } else {
         return(dataSoFar)
         } # if we dont have weight, then we wont have weight gradients either so we will just return the data passed along so far (it wont matter which case it was)
     },
     
     # removes an equal number of weights from a vector as this layer had, and then replaces the matching values in its Weight matrix
     addDebugData = function(allWeights) {
       if(is.na(Weights_W)[1] == FALSE) {
         
         numWeightsInLayer = length(Weights_W)
         
       
           Weights_W <<- matrix(  allWeights[1:numWeightsInLayer] , nrow = nrow(Weights_W) , ncol = ncol(Weights_W))
           
           totalLength = length(allWeights)
           allWeights = allWeights[(numWeightsInLayer+1):totalLength] # subset the original vector, in a way that leaves off an equal number of elements
         
       }
       
       return(allWeights) # return what is left of weights
     }
  
    ) 
)

