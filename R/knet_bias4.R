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


# myVec = c(1,2,3,4)
# norm(myVec)^2


# sum(myVec^2)

# pmax( c(1,2, -1, -3) , c(0,0, 0, 0))
# _______________________________________________________________

LAYERTYPE_SPLITTER = "LAYERTYPE_SPLITTER"
LAYERTYPE_JOINER = "LAYERTYPE_JOINER"
LAYERTYPE_CONV1 = "LAYERTYPE_CONV1"
LAYERTYPE_INPUT = "LAYERTYPE_INPUT"
LAYERTYPE_OUTPUT = "LAYERTYPE_OUTPUT"
LAYERTYPE_HIDDEN = "LAYERTYPE_HIDDEN"



# creates an Artifical Neural Network
# layer_config: the architecture of the network specifing the number of nodes in each layer: [num_inputs,num_hidden1,..., num_hiddenlast,num_output]
# iminibatch_size: the number observations per iteration
knn <- setRefClass("knn",
                   
                   fields = list( 
                     layers =  "list",
                     DEBG_WeightGrads = "list",
                     DEBUG = "logical",
                     #minibatch_size =  "numeric",
                     num_layers =  "numeric"
                     
                   ),
                   
                   methods = list(
                     initialize = function () {
                       layers <<- list()
                       num_layers <<- 0
                       
                       DEBUG <<- FALSE
                       DEBG_WeightGrads <<- list()

                     },
                     
                     addLayer = function (layer) {  # adds a layer to the neural networks list of layers
                       num_layers <<- num_layers+1
                       layers[[num_layers]] <<- layer 
                       print( paste("layer ",layer$type," is added to Network " ))
                     },
                     
                     connectLayers = function () {  # connects up each layer to its next, and initialises the Weighted layers with the correct dimension matrices
                       
                       # the first layer only has a next layer
                       layers[[1]]$initConnections( list(NEXT = layers[[2]]) )
                      
                       if(num_layers > 2)  { # if there  are any hidden layers
                         for(i in 2:(num_layers-1) ) { # go through all layers except last
                           
                           layers[[i]]$initConnections(  list(PREV= layers[[i-1]],NEXT = layers[[i+1]]) )
                         }
                         
                         # connect Output
                         layers[[num_layers]]$initConnections( list(PREV = layers[[num_layers-1]])  ) # the last layer only has a prev layer
                       }
                       

                       
                       print("All layers created, Neural Network ready")
                     },
                     
                     
                     # Main forward propagation algo: it usually return Sigma(XB), and feeds it int the next layer's input
                     forward_propagate = function (data) {  # data is the design matrix *X()
                       
                       return(layers[[1]]$forward_propagate(data)) # this recursively calls all subsequent layers' forward propagate until the last (the output, which produces the yhat)
                     },
                     
                     
                     # calculates the cumulative errors (Delta) at each node, for all layers
                     backpropagate =  function(yhat, y) {
                       
                       # calculate first Error Delta directly from output
                       Error_Delta_Output = t( yhat - y ) 
                       layers[[num_layers]]$backpropagate(Error_Delta_Output) # call this on the last layer, which then will recursively call all the rest all the way back to the first
                     },
                     
                     
                     # goes through and updates each layer (for ones that have weights this updates the weights, for others this may do something else or nothing)
                     update = function(num_samples_total, eta, minibatch_size) {
                       
                       for (i in 2:(num_layers)) { # loop from the 2nd layer to the last ( the 1st layer is the input data, and that cannot be updated)
                         layers[[i]]$update(num_samples_total, eta, minibatch_size, friction)
                       }
                     },
                     
                     
                     # train_X/Y (and test_X/Y): a list() of matrices, each element on the list being a minibatch matrix
                     learn = function(train_X, train_Y, test_X, test_Y, num_epochs=500, eta=0.05, eval_train=FALSE, eval_test=FALSE, eval_freq = 100,  friction = 0.0) {
                       print ( paste("Starting to train Neural Network for for num iterations:", num_epochs) )
                       
                       minibatch_size = nrow(  as.matrix(train_Y[[1]]) )
                       # initMatrices(minibatch_size)
                       
                       num_samples_total = length(train_Y) * nrow(  as.matrix(train_Y[[1]]) )
                       
                      
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
                           update(num_samples_total, eta, minibatch_size)
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
                             #out_str =  paste(out_str, " / Test ",evaluation,":", format(totals/N_test, digits=5), sep =" ")
                           }
                         }
                         print ( out_str )
                         
                         # if(t == 895) { break}
                       }
                       
                     },
                     
                     
                     
                     ####### Debugger functions: used for numerical Gradient checking
                     
                     
                     
                     # replaces the weights in the network from an external source: a 1D array (IE the same format as that is returned by the above)
                     setWeights = function(allWeights) {
                       #Set W1 and W2 using single paramater vector.
                       W_start = 1 # 1st weight's indices start at 0
                       for (i in 1:(num_layers-1)) {
                         W_end = W_start + length(layers[[i]]$Weights_W) -1
                         layers[[i]]$Weights_W <<- matrix(  allWeights[W_start:W_end] , nrow = nrow(layers[[i]]$Weights_W) , ncol = ncol(layers[[i]]$Weights_W))
                         W_start = W_end  +1 # update the start position for next round
                       }
                     },
                     
                     # returns the Sum of Squared errors ( IE this is the Cost function)
                     regressionCost_SumSquaredError = function (data, y, num_samples_total) {   #
                       #Compute cost for given X,y, use weights already stored in class.
                       yHat = forward_propagate(data)
                       batch_num_samples = nrow(layers[[1]]$Output_Z)
                       J =  0.5*  sum((y-yHat)^2) / batch_num_samples + getRegularizerCostTerm(num_samples_total)
                       return(J)
                     },
                     
                     multiclassCost_softMax = function(data, y, num_samples_total) {
                       yHat = forward_propagate(data)
                       batch_num_samples = nrow(layers[[1]]$Output_Z)
                       J =  -sum(  y * log(yHat))  / batch_num_samples + getRegularizerCostTerm(num_samples_total)    
                       return(J)
                     },
                     
                     getCurrentWeightGradients = function(data, y, num_samples_total) {
                       yHat = forward_propagate(data)
                       backpropagate(yHat, y) # we need o backpropagate also, as we need to know the CURRENT error, IE that is resulted from the weights that we have now (as the errors we have atm are the errors due to the previous iteration)
                       origDEBUG = DEBUG
                       DEBUG <<- TRUE # temporarily set this, so that the next function saves the weight gradients
                       update_weights(num_samples_total, eta=0) # this calculates the current weight GRADIENTS into an array, without actually updating them ( as eta is 0)
                       
                       allWeightGrads = vector()
                       for (i in 1:(num_layers-1)) { # go through all layers before last (as that doesnt have outgoing weights)
                         allWeightGrads = c(  allWeightGrads, c( DEBG_WeightGrads[[i]])  )
                       } 
                       DEBUG <<- origDEBUG # reset
                       return(allWeightGrads)
                     },
                     
                     gradientCheck = function (data, y) {
                       weightsOriginal = getWeightsAsVector()  # gets all weights
                       # init some empty vectors same num as weights
                       
                       numgrad = c( rep( 0, length(weightsOriginal))) # the numerical approximation for the derivatives of the weights
                       perturb = c( rep( 0, length(weightsOriginal))) # perturbations: these are #1-of-k' style, where we have 0 for all else, except for the current
                       e = 1e-4  # the perturbation
                       
                       
                       num_samples_total = nrow(data) # the total number of samples are the same as the minibatch size, as gradient checks are only performed on signle minibatches
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
                         loss2 = costFunction(data, y,num_samples_total) # get squared error: IE X^2  (IE this is the 'cost')
                         setWeights(weightsOriginal - perturb) # add the changed weights into the neural net (negative offset)
                         loss1 = costFunction(data, y,num_samples_total) # get squared error: IE X^2  (IE this is the 'cost')
                         
                         #print( cat("loss2 is: ", loss2))
                         # print( cat("loss1 is: ", loss1))
                         #print("num_samples_total is: ")
                         #print(  num_samples_total)
                         
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




# params: holds numeric parameters for current layer, eg, how many regions to split input
knnDummyLayer <- setRefClass("knnDummyLayer", 
         fields=list(
           type="character",
           params="numeric",
           isNull="logical",
           # Error_D = "matrix",
           parentNetwork="knn"
         ),
         methods=list(
           initialize = function (iparams = -1, itype = "character", iparentNetwork = knn()) {
             params <<- iparams
             type <<- itype
             isNull <<- TRUE  # R is retarded, it instantiates this layer, whenever it is just referenced i nprevLayer = "knnDummyLayer" (with default values)
             parentNetwork <<- iparentNetwork
             

           },
           generateOutput = function(input, ...) {},
           forward_propagate = function(input, ...) {},
           calcError = function(input,...) {},
           backpropagate = function(input,...) {},
           update = function(...) {},
           initConnections = function( prevNext = list() ) {  }
         )
)

# base class for all knn layers: should never be directly instantiated
knnBaseLayer <- setRefClass("knnBaseLayer",
      fields=list(
       # type="character",
      #  params="numeric",
      #  parentNetwork="knn",
        prevLayer = "knnDummyLayer", # this should be 'knnBaseLayer', but R cannot reference this type of class in the class definition, so we have to use a 'dummy' class
        nextLayer = "knnDummyLayer"
        ),
      
       contains="knnDummyLayer",

      methods=list(
        initialize = function (iparams, itype, iparentNetwork) {
          callSuper(iparams, itype, iparentNetwork)
          #params <<- iparams
          #type <<- itype
         # parentNetwork <<- iparentNetwork
          
          isNull <<- FALSE # for an instantiated layer, we set this to FALSE to let others know this really exists

          
          # add this layer to its parent Neural Network's list of layers
          parentNetwork$addLayer(.self)
        },
        
        
        generateOutput = function(input, ...) { 
        },
        
        
        forward_propagate = function(input, ...) {
          output = generateOutput(input)
          
          if( nextLayer$isNull ) { # if there is no next layer, IE this is the last one the Output
            return(output)
          }else { # if there are any next layers still, we recursively call them
            # send output from this layer to the next
            
            return( nextLayer$forward_propagate(output) )
          }
        },
        
        calcError = function(input,...) { # this receives the Error_Delta from the layer after this
        },
        backpropagate = function(input,...) { # this receives the Error_Delta from the layer after this
          
          
          if( prevLayer$isNull == FALSE) {# if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            # generate output rom this layer
            error_D = calcError(input) # this is USUALLY really just the Error_D, but the Joiner layer for instance, passes along the weights of the next layer too
            return( prevLayer$backpropagate(error_D) )
          } # else: if there are no prev layers then stop backpropagating ... IE we reached the INPUT layer
 
          
          
        },
        
        # generic update function, only does anything on weighted layers
        update = function(...) {
        },
        
        initConnections = function( prevNext = list() ) { # this is called once the entire network has been set up, IE once each layer is connected to the next
          # print(paste("prevNext is:", prevNext))
          # if( length(prevNext) == 0 ) {return()}# this is another R-hack, as R shits itself if we try to access elements of a list with 0 elements
          
          # check previous layer
          if( is.null(prevNext$PREV) == FALSE) { prevLayer <<- prevNext$PREV }  # R is retarded, you cannot assign a NULL value to a var that expects it to be a class (even thoug it is initialised as NULL...)
         
          # check next layer
          if( is.null(prevNext$NEXT) == FALSE) { nextLayer <<- prevNext$NEXT }
        }
        
      )
)

knnSplitter <- setRefClass("knnSplitter",
          fields=list(
            regions="list",
            numRegions="numeric",
            Error_D = "matrix",
            Input_S = "matrix"
          ),
          
          contains="knnBaseLayer",
          
          methods=list(
            initialize = function (iparams, itype, iparentNetwork) {
              callSuper(iparams, itype, iparentNetwork) # call init on parent

              # parse params into regions
              numRegions <<- length(params) -1  # 1st element sets the number of regions we will need
                                      # params[1] = a = # units in current layer;  need to keep the 1st element being the total number of units in this layeer (including in all regions), so that other layers can access this information the same way as for any other layers)
              regions <<- list()
              regionStart = 1
              for( i in 2:length(params)) { # loop from 2nd element, where each element is the LAST predictor in matrix
                regionEnd = params[i]
                
                regions[[(i-1)]] <<- c(regionStart, regionEnd)
                regionStart = regionEnd+1 #the next region's start is just the one after this ended
              }
              
            },
            
            # Splitter splits an incoming Input data into a predefined set of regions
            generateOutput = function(input, ...) {
             
              Input_S <<- input # save input
              
              subMatrices = list()
              for (i in 1:numRegions) {
                regionStart = regions[[i]][1]
                regionEnd = regions[[i]][2]
                subMatrices[[i]] = input[,regionStart:regionEnd] # cut the submatrix
                
              }
              return(subMatrices) # return the splits
            },
            
            # Splitter performs the opposite function for backpropagation: it Joins an incoming list error Dis into a single matrix
            calcError = function(input, ...) {
              # input here is a list()  of all the Error_D s from the next layer (the Convolution1D)
              new_Error_D = NULL
              for(i in 1:length(input)) {
                new_Error_D = rbind(new_Error_D, input[[i]] ) # Ds are transposed, so we join them by adding them under each other
              }
              
              Error_D <<-new_Error_D
              return(Error_D)
            }
          )
)


knnJoiner <- setRefClass("knnJoiner",
                           fields=list(
                             regions="list",
                             numRegions="numeric",
                             Error_D = "matrix", # this refers to the ENTIRE Error_D of the layer after this 
                             nextLayerWeights = "list", # list of the weights that were split into regions of the layer after this
                             # nextLayerWeight_bias = "numeric", # this is unused, we don't transfer this, although could be sent forward to Splitter, which could then add this into the D it joined...
                             Output_Z = "matrix" , # 
                             prevLayerOutputs = "list" # the outputs of a Conv layer
                             
                           ),
                         
                         contains="knnBaseLayer",
                           
                           methods=list(
                             initialize = function (iparams, itype, iparentNetwork) {
                               callSuper(iparams, itype, iparentNetwork) # call init on parent
                               
                               # parse params into regions
                               numRegions <<- length(params) -1 # 1st element sets the number of regions we will need ( need to keep the 1st element being the layer size, so that other layers can access this information the same way as for any other layers)
                                                        # params[1]   need to keep the 1st element being the tota number of units in this layeer (including in all regions), so that other layers can access this information the same way as for any other layers)
                               
                               regions <<- list()
                               regionStart = 1
                               for( i in 2:length(params)) { # loop from 2nd element, where each element is the LAST predictor in matrix
                                 regionEnd = params[i]
                                 
                                 regions[[(i-1)]] <<- c(regionStart, regionEnd)
                                 regionStart = regionEnd+1 #the next region's start is just the one after this ended
                               }
                               
                             },
                             
                             # Joiner joins an incoming list out outputs from a convolutional layer
                             generateOutput = function(input, ...) { # Joiner gets a list of outputs, that it will merge
                               prevLayerOutputs <<- input # save input
                               
                               output = NULL
                               for(i in 1:length(prevLayerOutputs)) {
                                 output = cbind(output, prevLayerOutputs[[i]] )
                               }
                               
                               
                               Output_Z <<- output # save output too
                               return(Output_Z)
                             },
                             
                             # Joiner performs the opposite function for backpropagation: it splits the incoming Weights, and passes along the entire D
                             calcError = function(input, ...) {
                               # input is a single Error matrix, that we wish to then split
                               Error_D <<- input #  save the entie Error_D  of next layer
 
                               # get the Weight of the next layer, this will be needed by the Conv1D layer that follows this
                               weightOfNextLayer = nextLayer$Weights_W
                               # could get the bias here too..
                               
                               nextLayerWeights <<- list()
                               
                               subMatrices = list()
                               for (i in 1:numRegions) {
                                 regionStart = regions[[i]][1]
                                 regionEnd = regions[[i]][2]
                                 #print(paste("regionStat:" , regionStart , "/ regionEnd:", regionEnd))
                                 #print(paste("weightOfNextLayer dimensions:" , nrow(weightOfNextLayer) , "/ :", ncol(weightOfNextLayer)))
                                 nextLayerWeights[[i]] <<- weightOfNextLayer[regionStart:regionEnd,] # cut the submatrix, as Ds are transposed at this stage we spit the rows
                                 
                               }
                               #print(paste("the length of nextLayerWeights is:" , length(nextLayerWeights)))
                               return( list(Error_D, nextLayerWeights) ) # the error itself has not changed, but we also pass along the next layer's weights split into a list of submatrices for each region
                               
                             }
                           )
)

# [numSamplesinBatch(n), currentLayerUnits(a), nextLayerUnits(b)]
knnConv1D <- setRefClass("knnConv1D",
                         fields=list(
                           regions="list",
                           numRegions="numeric",
                           numSamplesInBatch ="numeric",
                           Error_D = "list", # this refers to a list of 
                           numUnitsInregions ="numeric",
                           numUnitsInPrevLayer_Pi ="numeric",  # vector of each of the number of columns (IE predictors or weights) in previous layer's regions
                           regionRegularizers = "character",
                           regionShrinkageParams ="numeric",
                           Input_S = "list" 
                           
                         ),
                         
                         contains="knnBaseLayer",
                         
                         methods=list(
                           initialize = function (iparams, itype, iparentNetwork, inumUnitsInregions, inumUnitsInPrevLayer_Pi, iregionRegularizers, iregionShrinkageParams) {
                             callSuper(iparams, itype, iparentNetwork) # call init on parent
                             
                             # parse params into regions
                             numRegions <<- params[1] # 1st element sets the number of regions we will need
                             # numSamplesInBatch <<- params[2] # 2nd element sets the number of total samples in a minibatch (IE = n)
                             regions <<- list()
                           
                             numUnitsInregions <<- inumUnitsInregions
                             numUnitsInPrevLayer_Pi <<- inumUnitsInPrevLayer_Pi
                             regionRegularizers <<- iregionRegularizers
                             regionShrinkageParams <<- iregionShrinkageParams
 
                          dummyKnn = knn() # as all layers must belong to a Knet, we need to create a dummy one here (this is a Hacky fix...)
                             for(i in 1:numRegions){ 
                              
                               # create layer
                               regions[[i]] <<-  knnLayer(numUnitsInregions[i], LAYERTYPE_HIDDEN, dummyKnn, iactivation=k_linear, regularizer = regionRegularizers[i], shrinkageParam = regionShrinkageParams[i])
                              
                               # init layer 
                               
                              # need to call initConnections, but without having to rely on prevLayer. as there won't be', as we will use this:
                               # will need to make sure that prev/next layer won't exist
                               regions[[i]]$initWeightMatrix(numUnitsInPrevLayer_Pi[i])

                             }
                           },
                           
                           # Convolution1D, goes through a list of mini Weighted layers, and collects their output, which is then sent forward
                           generateOutput = function(input, ...) { # Convolution1D gets a list of inputs (Xi) from a Splitter
                             Input_S <<- input # save away the list of subMatrices
                             
                             convOutput = list()
                             for(i in 1:numRegions) { # go through each region's layer, and make each output its bit
                               convOutput[[i]] = regions[[i]]$forward_propagate(input[[i]])
                             }

                             return(convOutput)
                           },
                           
                          
                          update = function(num_samples_total, eta, num_samples_minibatch, friction) {
                            for(i in 1:numRegions) { # go through each region's layer, and force them to 
                              regions[[i]]$update(num_samples_total, eta, num_samples_minibatch, friction)
                            }
                          },
                          
                          
                           # Conv1D gets a single Error_D, which comes from the next FC layer, and which was passed along by the Joiner intact
                           # Joiner also passes along its 
                          calcError = function(input, ...) {
                             
                            Error = input[[1]]
                            nextLayerWeights = input[[2]]
                            #print(paste("the length of nextLayerWeights is:" , length(nextLayerWeights)))
                            #print(paste("the length of input is:" , length(input)))
                            
                            convOutput = list()
                             for (i in 1:numRegions) { 
                               # need to force the region layer, to use the submatrix of the next FC layer, so we 
                               # use the 'dummy nextLayer layer', that R creates as a mule, by overriding its 'Weights_W'. with the submatrix
  
                               # regions[[i]]$nextLayer$Weights_W <<- nextLayerWeights[[i]] 
                               # also we don't want to call backpropagate, as there is no real 'prevLayer'
                              
                               convOutput[[i]] = regions[[i]]$calcError(Error, nextLayerWeights[[i]] ) # this performs Di_part = W_NEXT * t(D_FC_all) * Fp
                             }

                             Error_D <<- convOutput # save away ALL the Error_Ds of all the regions
                             return(Error_D) # return the splits
                             
                           },
                           
                           
                           # returns all weights in the NN concated into a 1D array
                           getWeightsAsVector = function () { # a Conv layer just goes through ALL of its mini layer's weights
                             allWeights = vector()
                             for (i in 1:numRegions) { 
                               allWeights = c(  allWeights, regions[[i]]$getWeightsAsVector() )
                             }

                             return(allWeights)
                           }
                         )
)

## LAYER

# size is a vector that has 2 entries: [0] is the number of units in the current layer, [1] number of units in the next layer (needed for the weights matrix), 
# numInputs: is the number of units in previous layer ( needed to init the weight variances) 
knnLayer <- setRefClass("knnLayer",
     fields = list(
       layer_numUnits= "numeric", 
       biasEnabled = "logical",
       activation  = "function", 
       Weights_W = "matrix", 
       Weights_bias = "numeric", 
       Output_Z = "matrix", 
       Error_D = "matrix", # moved to the baseclass
       Derivative_Fp = "matrix",
       Momentum = "matrix",
       Bias_Momentum = "numeric",
       Input_S = "matrix",
       regularizer = "character",
       Lambda =  "numeric"
       ),
     
     contains="knnBaseLayer",
     
     methods = list(
       initialize = function (iparams, itype, iparentNetwork,iactivation=k_no_activation, ibiasEnabled = TRUE, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0) { 
         callSuper(iparams, itype, iparentNetwork) # call init on parent
         regularizer <<- regularizer
         Lambda <<- shrinkageParam
         biasEnabled <<- ibiasEnabled
         layer_numUnits <<- params[1]
         
         # The activation function is an externally defined function (with a derivative) that is stored here )
         activation <<- iactivation
         
         # Z is the matrix that holds output values
         Output_Z <<-  matrix(NA) # matrix(0.0, nrow = minibatch_size, ncol = layer_numUnits)
         
         
         # W is the INCOMING weight matrix for this layer
         Weights_W <<- matrix(NA)
         
         
         # S is NULL matrix that holds the inputs to this layer
         Input_S <<- matrix(NA)
         # D is the matrix that holds the deltas for this layer
         Error_D <<- matrix(NA)
         # Fp is the matrix that holds the derivatives of the activation function applied to the input
         Derivative_Fp <<- matrix(NA) 

         
         print( paste("layer ",type," is regularized by: " , regularizer, "/ its params are", params))
         }, 
       
       # called BEFORE starting the learning process, when we find out the size of the minibatches
       # dont need this, as none of these matrices are ever used, they are all overriden at each trainin cycle
      # initMatrices = function( minibatch_size ) { 
      #   if ( type != LAYERTYPE_INPUT &&  type != LAYERTYPE_OUTPUT) {
      #     Derivative_Fp <<- matrix(0.0, nrow = layer_numUnits, ncol = minibatch_size)  # this is in fact inited transposed
      #   }
      #   Output_Z <<-  matrix(0.0, nrow = minibatch_size, ncol = layer_numUnits) 
      #   
     #    
      #   if (type != LAYERTYPE_INPUT) {
     #     Input_S <<- matrix(0.0, nrow = minibatch_size, ncol = layer_numUnits)  # we init S with +1 cols, but then assign them to be -1 size
      #     Error_D <<- matrix(0.0, nrow = minibatch_size, ncol = layer_numUnits)  # we init D with +1 cols, but then assign them to be -1 size
      #   }
      # },
       
       initConnections = function( prevNext = list() ) { 
         callSuper(prevNext)
         
         #  the 1st element in the params array refers to the size of this layer( IE how many units)
         #size = c(params[1], params[2]) # the 2nd element refers to the number of rows in the datamatrix,
        # layer_numUnits <<- params[1]
        # minibatch_size = params[2]
         

        # Weights can be initialised only by knowing the number of units in the PREVIOUS layer (but we do NOT need to know the size of the minibatches (IE n), as Weight matrix' dimension does not depend on that
         if ( prevLayer$isNull == FALSE) { # type != LAYERTYPE_OUTPUT
           
           # TODO: need to add additional checks here, if the next layer is NOT a 'fully connected' type,
           # IE it is possible, that this is an Input layer, and the next layer is a splitter
           # in that case there shouldn't be any weights initted
           # if (nextLayer$type == weighted layer) ... then do the following
           
           prevLayer_size = prevLayer$params[1] # find out how big the next layer is 
           print(paste("prevLayer_size is:", prevLayer_size, " / and layer_numUnits in this layer is:", layer_numUnits))
           
           initWeightMatrix(prevLayer_size)

         }
       },
     
     initWeightMatrix = function(prevLayer_size) {
       
       numInputs = layer_numUnits # numInputs is the number of inputs that a Layer gets, IE the number of neurons in the previous layer. However as we store Weights of W1->2, on Layer 1, the 'previous' layer is this one (IE each layer stores the weights matrix for the NEXT layer)
       Weights_W  <<- matrix( rnorm(prevLayer_size * layer_numUnits, sd=sqrt(2.0 / numInputs) ),  nrow = prevLayer_size, ncol=layer_numUnits  ) # modified xavier init of weights for RELU/softplus (this also has the effect of basic L2 regularization)
       Momentum <<- matrix(0.0, nrow = nrow(Weights_W),  ncol = ncol(Weights_W) ) # stores a 'dampened' version of past weights 
       
       if (biasEnabled == TRUE) { # if we have bias/intercept, we add a row of Weights that we keep separate
         Weights_bias <<- rnorm(ncol(Weights_W)                    , sd=sqrt(2.0 / numInputs) )
         Bias_Momentum <<- rep(0, length(Weights_bias)) # stores a 'dampened' version of past weights  for intercepts
       }
       
     },
     
       # performs the addition of the bias terms effect  onto the output
       add_bias = function(ZW) {
         if(biasEnabled == FALSE) { return(ZW)} # if bias wasn't enabled in the first place we just return the orig
         
         ZWb = sweep(ZW,2,Weights_bias,"+") # we simply add the bias to every row, the bias is implicitly multipliedby 1
         return(ZWb) 
       },
       
       generateOutput = function(input,...) {
         # if its the first layer, we just return the linear predictor
         if (type == LAYERTYPE_INPUT) {
           Output_Z <<- input
     
           # ZW = Output_Z %*% Weights_W # this assumes that there ARE weights on this layer..
           #return( add_bias(ZW)  ) 
           
           return( Output_Z  ) 
         } else { # if its NOT an input, then all FC layers will have weights (even Output)
           Input_S  <<- input # save this away
           #print(paste("Input_S dim is:", nrow(Input_S), "/" , ncol(Input_S) ))
           #print(paste("Weights_W dim is:", nrow(Weights_W), "/" , ncol(Weights_W) ))

           Output_Z <<- Input_S %*% Weights_W  # so we just multiply the incoming data, by the weights
           }
         
         Output_Z <<- add_bias(Output_Z)
         
         if (type != LAYERTYPE_OUTPUT) {  Derivative_Fp <<- t( activation(Output_Z, deriv=TRUE) ) } # this is transposed, as D is also transposed during back propagation
         
         
         Output_Z <<- activation(Output_Z) # if its a hidden layer (or output), we need to activate it
         
         return ( (Output_Z) )
         
       },
     
       calcError = function(input, customWeight = NULL,...) { 
         if (type == LAYERTYPE_OUTPUT) { 
          Error_D <<-  input # this is stored as Transposed D ()
        } else { # 'input' here basically refers to 'nextLayer$Error_D', except for minilayers in a Conv1D
          # each layer's Delta, is calculated from the Delta of the Layer +1 outer of it, and this layer's Weights scaled by the anti Derivative Gradient
          #Error_D <<- (Weights_W %*% input) * Derivative_Fp # the current layer's Delta^T = (Weights_W * D_prev^T) *Schur* F^T   # 'rev' actually refers to the 'next' layer closer to the poutput as we are going backwards
                                    # this is OK, as a hidden layer will never have output into a conv layer
                                    # but what about conv1D's minilayers??
          # we assume that calcError() is NEVER called on the Input layer, as that doesn't even have an error
          #    to get the current layer's ErrorDelta, we need the NEXT layer's weight
           
          if( is.null(customWeight) ) { weightToUse = nextLayer$Weights_W} else {weightToUse = customWeight}
          
          Error_D <<- (weightToUse %*% input) * Derivative_Fp # the current layer's Delta^T = (Weights_W * D_prev^T) *Schur* F^T   # 'rev' actually refers to the 'next' layer closer to the poutput as we are going backwards
          # this is OK, as a hidden layer will never have output into a conv layer
          # but what about conv1D's minilayers??     
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
       
       
       update = function(num_samples_total, eta, num_samples_minibatch, friction) {
          #print(paste("update on type:", type ))
         
         if(is.na(Weights_W)[1] == FALSE) { # if it has any outgoing weights (IE not the output, or an input that feeds into a splitter)
         # print(paste("for type:",type," / nextLayer$Error_D dims are:", dim(nextLayer$Error_D)," / Output_Z dims are:", dim(Output_Z)))
           #print("EXECUTED")
           #print(paste("for","uga"))
           W_grad = t(Error_D %*% Input_S) # the weight changes are (D_next^t * Input_current) and transposed back so dims match
           W_grad = W_grad / num_samples_minibatch # normalise by N, to keep the same as the cost function
           
           regularizers = getRegularizerGrad(num_samples_total) 
           
           W_grad = W_grad + regularizers
           
           #if (DEBUG == TRUE) { # if we are in Debug mode, we want to save away the current Weight gradients
           #  DEBG_WeightGrads[[i]] <<- W_grad
           #}
           
           # add Mometum (velocity)
           #print(paste(" Output_Z dim:", dim(Output_Z)))
           #print(paste(" W_grad dim:", dim(W_grad), " / Momentum dim:" , dim(Momentum)))
           Momentum <<- friction * Momentum - (eta*W_grad) 
           
           Weights_W <<- Weights_W +  Momentum
           
           # update bias/intercept terms separately (if needed)
           update_bias_weights(num_samples_minibatch, eta,friction)  # bias weights are not trained ...for now (I think this has to do with the fact that NNs have universal approx ability, as long as the function doesn't go through 0)
           
           #  print(layers[[i]]$Weights_W)
         }
       },
       # the cost of the regularizer term
       getRegularizerCostTerm = function(num_samples_total) {
         #num_samples = layers[0].Output_Z.shape[0]
         
         if (regularizer == REGULARIZER_NONE) {
           return(0.0) 
         } else if (regularizer == REGULARIZER_RIDGE) {
           return ( (Lambda/num_samples_total * 0.5) * sum( getWeightsAsVector()^2 ) ) 
         } else { # LASSO
           return ( (Lambda/num_samples_total * 0.5) * sum( abs( getWeightsAsVector() ) ) ) }
       },
       
       # the derivative of the regularizer term: have to normalise by the total number of samples in the total dataset, otherwise regularization would grow with smaller minibatches
       getRegularizerGrad = function(num_samples_total) {
         # num_samples = layers[0].Output_Z.shape[0]
         
         if ( regularizer == REGULARIZER_NONE ) {
           return(0.0)
         } else if (regularizer == REGULARIZER_RIDGE ) {
           return ( Lambda/num_samples_total * Weights_W )
         } else { # LASSO
           return ( Lambda/num_samples_total * sign(Weights_W ) ) }  # this will cause the numerical check to fail.. it will still be small but much bigger than with L2 or without, this is probably due to the funky L1 derivative at 0
       },
       # returns the weights in in the current layer concated into a 1D array
       getWeightsAsVector = function () {
         return( c(Weights_W) )
       }
       
       
    ) 
)

