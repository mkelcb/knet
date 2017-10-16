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

## LAYER

# size is a vector that has 2 entries: [0] is the number of units in the current layer, [1] number of units in the next layer (needed for the weights matrix), 
# numInputs: is the number of units in previous layer ( needed to init the weight variances) 
knnLayer <- setRefClass("knnLayer",
     fields = list(
       is_input  = "logical",
       is_output   = "logical",
       biasEnabled = "logical",
       activation  = "function", 
       Weights_W = "matrix", 
       Weights_bias = "numeric", 
       Output_Z = "matrix", 
       # Z_bias = "numeric", 
       Error_D = "matrix",
       Derivative_Fp = "matrix",
       Momentum = "matrix",
       Bias_Momentum = "numeric",
       Input_S = "matrix"
       ),
     methods = list(
       initialize = function (size, minibatch_size, iis_input=FALSE, iis_output=FALSE,iactivation=k_sigmoid, ibiasEnabled = TRUE) {  
         is_input <<- iis_input
         is_output <<- iis_output
         biasEnabled <<- ibiasEnabled
         
         # Z is the matrix that holds output values
         Output_Z <<- matrix(0.0, nrow = minibatch_size, ncol = size[1])
         #if (ibiasEnabled == TRUE) { # if we have bias/intercept, we add a col of 1s
         #  Z_bias <<- rep(1,nrow(Output_Z))
         #  print("adding bias for Layer")
         #}
         
         # The activation function is an externally defined function (with a derivative) that is stored here )
         activation <<- iactivation
         
         # W is the outgoing weight matrix for this layer
         Weights_W <<- matrix(NA)
         
         
         # S is NULL matrix that holds the inputs to this layer
         Input_S <<- matrix(NA)
         # D is the matrix that holds the deltas for this layer
         Error_D <<- matrix(NA)
         # Fp is the matrix that holds the derivatives of the activation function applied to the input
         Derivative_Fp <<- matrix(NA)
         
         if (is_input == FALSE) {
           Input_S <<- matrix(0.0, nrow = minibatch_size, ncol = size[1])  # we init S with +1 cols, but then assign them to be -1 size
           Error_D <<- matrix(0.0, nrow = minibatch_size, ncol = size[1])  # we init D with +1 cols, but then assign them to be -1 size
         }
         if (is_output == FALSE) {
           numInputs = size[1] # numInputs is the number of inputs that a Layer gets, IE the number of neurons in the previous layer. However as we store Weights of W1->2, on Layer 1, the 'previous' layer is this one (IE each layer stores the weights matrix for the NEXT layer)
           Weights_W  <<- matrix( rnorm(size[1] * size[2], sd=sqrt(2.0 / numInputs) ),  nrow = size[1], ncol=size[2]  ) # modified xavier init of weights for RELU/softplus (this also has the effect of basic L2 regularization)
           Momentum <<- matrix(0.0, nrow = nrow(Weights_W),  ncol = ncol(Weights_W) ) # stores a 'dampened' version of past weights 
           
           if (ibiasEnabled == TRUE) { # if we have bias/intercept, we add a row of Weights that we keep separate
             Weights_bias <<- rnorm(ncol(Weights_W)                    , sd=sqrt(2.0 / numInputs) )
             Bias_Momentum <<- rep(0, length(Weights_bias)) # stores a 'dampened' version of past weights  for intercepts
           }
           
         }
         if ( is_input == FALSE &&  is_output == FALSE) {
           Derivative_Fp <<- matrix(0.0, nrow = size[1], ncol = minibatch_size)  # we init F with +1 rows, but then assign them to be -1 size
            }
         }, 
       # performs the addition of the bias terms effect  onto the output
       add_bias = function(ZW) {
         if(biasEnabled == FALSE) { return(ZW)} # if bias wasn't enabled in the first place we just return the orig
         #print("Z bias dimensions")
         #print(length(Z_bias))
         
         #print("Weights_bias dimensions")
         #print(length(Weights_bias))
         #Sum = Z_bias * Weights_bias  # this assumes that Z is the same dimensions as when we first initted it (IE all minibatches have same n)
         #ZWb = sweep(ZW,2,Sum,"+") # add to each column the Bias vector
         
         #Sum = matrix(Z_bias) %*% Weights_bias  # this assumes that Z is the same dimensions as when we first initted it (IE all minibatches have same n)
         #ZWb = ZW+Sum # add to each column the Bias vector 
         
         ZWb = sweep(ZW,2,Weights_bias,"+") # we simply add the bias to every row, the bias is implicitly multipliedby 1
         return(ZWb) 
       },
       
       forward_propagate = function() {
         # if its the first layer, we just return the linear predictor
         if (is_input == TRUE) { 
           #prints( dim(Output_Z))
           #prints( dim(Weights_W))
           ZW = Output_Z %*% Weights_W
           return( add_bias(ZW)  ) 
           #return( ZW  ) 
           }
         
         
         Output_Z <<- activation(Input_S) # if its a hidden layer (or output), we need to activate the 'input'
         
         if (is_output == TRUE) {  return(Output_Z) } else { # if its output, then we just return the result
           # For hidden layers, we add the bias/Intercept here
          # intercept =  matrix( rep(1, ncol(Output_Z) ) , ncol = 1, nrow = ncol(Output_Z) )
           #Output_Z <<- cbind(Output_Z , (rep(1.0, nrow(Output_Z) ))  )  # this adds a column of 1s to the END, IE intercept is the last not the first
           Derivative_Fp <<- t( activation(Input_S, deriv=TRUE) ) # this is transposed, as D is also transposed during back propagation
           
           ZW = Output_Z %*% Weights_W
           return ( add_bias(ZW) )
          }
       }
    ) 
)


# creates an Artifical Neural Network
# layer_config: the architecture of the network specifing the number of nodes in each layer: [num_inputs,num_hidden1,..., num_hiddenlast,num_output]
# iminibatch_size: the number observations per iteration
knn <- setRefClass("knn",
                   
          fields = list( 
            layers =  "list",
            DEBG_WeightGrads = "list",
            DEBUG = "logical",
            outPutType = "character",
            num_layers =  "numeric",
            minibatch_size =  "numeric",
            regularizer = "character",
            biasEnabled = "logical",
            friction = "numeric",
            Lambda =  "numeric"
            ),
          
          methods = list(
            initialize = function (layer_config, hidden_activation = HIDDEN_ACT_SPLUS, output_type = OUT_BINARY, iminibatch_size=100, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0, ibiasEnabled = TRUE,  ifriction = 0.0) {
              layers <<- list()
              num_layers <<- length(layer_config)
              minibatch_size <<- iminibatch_size
              actfunct = NULL
              DEBUG <<- FALSE
              outPutType <<- output_type # this is used to determine for debugging the cost function to be used
              regularizer <<- regularizer
              biasEnabled <<- ibiasEnabled
              Lambda <<- shrinkageParam
              friction <<- ifriction

              
              DEBG_WeightGrads <<- list()
              
              print( cat("Net is regularized by: " , regularizer))
              
              # determine the activation function
              if(hidden_activation == HIDDEN_ACT_SPLUS) {
                actfunct = k_softplus
              } else if (hidden_activation == HIDDEN_ACT_SIG ) {
                actfunct = k_sigmoid
              } else  { actfunct = k_leakyRELU }
              
              for (i in 1:(num_layers-1)) { # go through all layers except last
                if (i == 1) { # init the first layer as the Input, need to treat this special
                print ( paste("Initializing INPUT layer with size", layer_config[i], " / activation:", hidden_activation) )
                  
                # Here, we add an additional unit at the input for the bias weight.
                layers[[i]] <<- knnLayer( c(layer_config[i], layer_config[i+1]) ,minibatch_size, iactivation=actfunct,iis_input=TRUE, ibiasEnabled = biasEnabled)
                                       # the 'size' of each layer is an array, with only the first value being
                                       # the number of nodes in this layer, the 2nd entry being the NEXT layer's
                                       # size, which we need to know to get the right dimensions for the Weights matrix
                
              } else {  # all hidden layers are treated as normal
                
                print ( paste("Initializing hidden layer with size", layer_config[i], " / activation:", hidden_activation) )
                # Here we add an additional unit in the hidden layers for the bias weight.
                layers[[i]] <<- knnLayer( c(layer_config[i], layer_config[i+1]), minibatch_size, iactivation=actfunct, ibiasEnabled = biasEnabled)
                }
              }
              
              # the last layer is the Output, also need to be treated separately
              print (  paste("Initializing output layer with size",layer_config[length(layer_config)], " / activation:", output_type) )
              # determine the activation function
              if(output_type == OUT_MULTICLASS) {
                actfunct = k_softmax
              } else if (output_type == OUT_REGRESSION  ) {
                actfunct = k_linear
              } else  { actfunct = k_softmax } # if Binary, we still use Softmax as that has a linear derivative { actfunct = k_sigmoid }
              
              layers[[num_layers]] <<- knnLayer( c(length(layer_config), -1),minibatch_size,iis_output=TRUE,iactivation=actfunct, ibiasEnabled = FALSE) # -1 as it doesnt have outgoing weights,  # the output layer NEVER has any bias
             
               print("All layers created, Neural Network ready")
            },
            
            # Main forward propagation algo: it usually return Sigma(XB), and feeds it int the next layer's input
            forward_propagate = function (data) {  # data is the design matrix *X()
              # We need to be sure to add bias/intercept column to the input ( this has to be done here, all the rest of the layers have initialised internall to already include this)
              # intercept = matrix(  , ncol = 1, nrow = ncol(data) )
               #layers[[1]]$Output_Z <<- cbind( data , (rep(1.0, nrow(data) )) )  # this adds a column of 1s to the END, IE intercept is the last not the first
              layers[[1]]$Output_Z <<- data  # add the data WITHOUT the intercept
              
              for (i in 1:(num_layers-1)) { # go through all layers before last (as that has its output calculated separately)
                layers[[i +1]]$Input_S <<- layers[[i]]$forward_propagate() # feed the next layer with current layer's output
              }
              
              return(layers[[num_layers]]$forward_propagate() ) # return the output of the final layer
            },
            
            # calculates the cumulative errors (Delta) at each node, for all layers
            backpropagate =  function(yhat, y) {
              
              # calculate first Delta directly from output !!WARNING!! ( I think this only works for Softmax/Linear output layers, but not yes/no classifications)
              residuals = t( yhat - y ) # store this locally
              layers[[num_layers]]$Error_D <<-  residuals # this creates a Transposed D ()
              
              if(num_layers >= 3) { # only do this if we have at least 1 hidden layer, otherwise we would attempt to loop forwards
                for (i in (num_layers-1):2) { # loop backwards from the last, to the 2nd
                  #print("backpropping layer")
                  #print(i)
                  # We do not calculate deltas for the bias values (another local var)
                  #W_nobias =  ( layers[[i]]$Weights_W[1:( (nrow(layers[[i]]$Weights_W)-1) ), ] ) # get all but the last row of the Weight matrix, which stores the weights for the Intercept terms
                  
  
  
                  # each layer's Delta, is calculated from the Delta of the Layer +1 outer of it, and this layer's Weights scaled by the anti Derivative Gradient
                  layers[[i]]$Error_D <<- (layers[[i]]$Weights_W %*% layers[[i+1]]$Error_D) * layers[[i]]$Derivative_Fp # the current layer's Delta^T = (Weights_W * D_prev^T) *Schur* F^T   # 'rev' actually refers to the 'next' layer closer to the poutput as we are going backwards
                }
              }
            },
            
            
            # updates the Bias Weights separately ( if enabled)
            update_bias_weights = function(i, num_samples_minibatch, eta) {
              if(biasEnabled == TRUE) {
            
               
                #W_bias_grad = t( layers[[i+1]]$Error_D %*% layers[[i]]$Z_bias)
                W_bias_grad =  rowSums(layers[[i+1]]$Error_D)  # via implicit multiplication of D by 1 ( IE row sums)
                W_bias_grad = W_bias_grad / num_samples_minibatch
                
                # add Mometum (velocity)
                layers[[i]]$Bias_Momentum <<- friction * layers[[i]]$Bias_Momentum - (eta*W_bias_grad)

                
                # bias gradients are NOT regularized, so we just apply them as is
                layers[[i]]$Weights_bias <<-  layers[[i]]$Weights_bias + layers[[i]]$Bias_Momentum
              }
            },
            
            
             update_weights = function(num_samples_total, eta) {
               num_samples_minibatch = nrow(layers[[1]]$Output_Z)
               
               for (i in 1:(num_layers-1)) { # go through all layers before last (as that doesnt have outgoing weights)
                W_grad = t( layers[[i+1]]$Error_D %*% layers[[i]]$Output_Z) # the weight changes are (D_next^t * Z_current) and transposed back so dims match
                W_grad = W_grad / num_samples_minibatch # normalise by N, to keep the same as the cost function

                regularizers = getRegularizerGrad(i, num_samples_total) 

                W_grad = W_grad + regularizers

                if (DEBUG == TRUE) { # if we are in Debug mode, we want to save away the current Weight gradients
                  DEBG_WeightGrads[[i]] <<- W_grad
                }
                
                #print(paste(" Output_Z dim:", dim(layers[[i]]$Output_Z)))
                #print(paste(" W_grad dim:", dim(W_grad), " / Momentum dim:" , dim(layers[[i]]$Momentum)))
                
                # add Mometum (velocity)
                layers[[i]]$Momentum <<- friction * layers[[i]]$Momentum - (eta*W_grad) 
                
                layers[[i]]$Weights_W <<- layers[[i]]$Weights_W +  layers[[i]]$Momentum
                
                # update bias/intercept terms separately (if needed)
                update_bias_weights(i,num_samples_minibatch, eta)  # bias weights are not trained ...for now (I think this has to do with the fact that NNs have universal approx ability, as long as the function doesn't go through 0)
                
              #  print(layers[[i]]$Weights_W)
               }
            },
             
            # train_X/Y (and test_X/Y): a list() of matrices, each element on the list being a minibatch matrix
            learn = function(train_X, train_Y, test_X, test_Y, num_epochs=500, eta=0.05, eval_train=FALSE, eval_test=FALSE, eval_freq = 100) {
              print ( paste("Starting to train Neural Network for for num iterations:", num_epochs) )
              

              num_samples_total = length(train_Y) * nrow(  as.matrix(train_Y[[1]]) )
              
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
                    update_weights(num_samples_total,eta)
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
            getRegularizerGrad = function(i, num_samples_total) {
              # num_samples = layers[0].Output_Z.shape[0]
              
              if ( regularizer == REGULARIZER_NONE ) {
                return(0.0)
              } else if (regularizer == REGULARIZER_RIDGE ) {
                return ( Lambda/num_samples_total * layers[[i]]$Weights_W )
              } else { # LASSO
                return ( Lambda/num_samples_total * sign(layers[[i]]$Weights_W ) ) }  # this will cause the numerical check to fail.. it will still be small but much bigger than with L2 or without, this is probably due to the funky L1 derivative at 0
            },
            
            
            ####### Debugger functions: used for numerical Gradient checking
            
            # returns all weights in the NN concated into a 1D array
            getWeightsAsVector = function () {
              allWeights = vector()
              for (i in 1:(num_layers-1)) {
                #allWeights = c(  allWeights, c(t(layers[[i]]$Weights_W))  ) # this would generate a vector by COLS. IE [1,3,2,4] , problem is that we cannot turn that back into the original matrix
                allWeights = c(  allWeights, c( layers[[i]]$Weights_W )  ) # so we just use it without the transpose
              }
            return(allWeights)
            },
            
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


