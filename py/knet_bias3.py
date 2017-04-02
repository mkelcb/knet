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

import numpy as np

def k_sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        # return np.multiply(k_sigmoid(X),(1 - k_sigmoid(X)))
        return k_sigmoid(X)*(1 - k_sigmoid(X))


def k_softmax(X):
    Output_Z = np.sum(np.exp(X), axis=1)
    Output_Z = Output_Z.reshape(Output_Z.shape[0], 1)
    return np.exp(X) / Output_Z


def k_softplus(X, deriv=False):
    if not deriv:
        return np.log( 1 + np.exp(X)  )
    else:
        return k_sigmoid(X)
    
def k_linear(X):
    return X



def k_leakyRELU(X, deriv=False):
    if not deriv:
        EX = (0.001 * X)
        return np.maximum(X, EX)
    else:
        return np.ones((X.shape[0], X.shape[1]))


def calc_MAE(yhat, y): 
  total = 0.0

  for i in range( len(yhat) ):  # go row by row
    prediction = yhat[i,:]
    true = y[i,:]
    index =  true.argmax() # find index of true value (IE in a 1 of K scenario this is the index of the highest value eg: (0,1,0) -> it is 2 )
    error = abs( prediction[index] - true.max() ) # absolut edifference between our predicted value and the truth
    error_perc = abs ( error / true.max() ) # express this as a % of the ideal value
    total = total + error_perc
    
  return ( total )

# calculates multiclass classification accuracy  (use for softmax output)
def calc_Accuracy(yhat, y): 
    num_matches = 0.0
    for i in range( len(yhat) ):  # go row by row
        if  np.argmax(yhat[i,:]) ==  np.argmax(y[i,:]) : 
            num_matches = num_matches+1
         
    #perc = num_matches/len(yhat)
    return(num_matches)


def l2norm(x):
    return ( np.sqrt(np.sum(x**2)) )
#  return ( total / len(yhat) )


# elementwise RELU with zeroes
# np.maximum(X, np.zeros((X.shape[0],X.shape[1])))

class knnLayer:
    def __init__(self, size, minibatch_size, is_input=False, is_output=False,
                 activation=k_sigmoid, biasEnabled = True):
        self.is_input = is_input
        self.is_output = is_output
        self.biasEnabled = biasEnabled
     
    
        # Z is the matrix that holds output values
        self.Output_Z = np.zeros((minibatch_size, size[0]))
        #if biasEnabled :  # if we have bias/intercept, we add a col of 1s
        #    self.Z_bias = np.ones( (self.Output_Z.shape[0],1) )
        
        # The activation function is an externally defined function (with a
        # derivative) that is stored here
        self.activation = activation

        # W is the outgoing weight matrix for this layer
        self.Weights_W = None
        
        
        # S is the matrix that holds the inputs to this layer
        self.Input_S = None
        # D is the matrix that holds the deltas for this layer
        self.Error_D = None
        # Fp is the matrix that holds the derivatives of the activation function applied to the input
        self.Derivative_Fp = None
        

        if not is_input:
            self.Input_S = np.zeros((minibatch_size, size[0]))
            self.Error_D = np.zeros((minibatch_size, size[0]))

        if not is_output:
            numInputs = size[0] # numInputs is the number of inputs that a Layer gets, IE the number of neurons in the previous layer. However as we store Weights of W1->2, on Layer 1, the 'previous' layer is this one (IE each layer stores the weights matrix for the NEXT layer)
            self.Weights_W = np.random.normal(size=size, scale=np.sqrt(2.0 / numInputs )) # modified xavier init of weights (this also has the effect of basic L2 regularization)
            self.Momentum = np.zeros((self.Weights_W.shape[0], self.Weights_W.shape[1])) # stores a 'dampened' version of past weights 
                    

            if biasEnabled :  # if we have bias/intercept, we add a row of Weights that we keep separate
                self.Weights_bias = np.random.normal(size=(1,self.Weights_W.shape[1]), scale=np.sqrt(2.0 / numInputs ))  
                self.Bias_Momentum = np.zeros(self.Weights_bias.shape[0]) # stores a 'dampened' version of past weights  for intercepts
                
        if not is_input and not is_output:
            self.Derivative_Fp = np.zeros((size[0], minibatch_size))

    # performs the addition of the bias terms effect  onto the output
    def add_bias(self, ZW) :

        if self.biasEnabled == False:  
            return(ZW) # if bias wasn't enabled in the first place we just return the orig
         
       # Sum = self.Z_bias.T * self.Weights_bias  # this assumes that Z is the same dimensions as when we first initted it (IE all minibatches have same n)
        #ZWb = ZW + Sum  # add to each column the Bias vector

        ZWb = ZW + self.Weights_bias # we simply add the bias to every row, the bias is implicitly multipliedby 1
        return(ZWb) 
       
    def forward_propagate(self):
        if self.is_input:
            ZW = self.Output_Z.dot(self.Weights_W)
            return ( self.add_bias(ZW) )

        self.Output_Z = self.activation(self.Input_S)
 
        
        if self.is_output:
            return self.Output_Z
        else:
            # For hidden layers, we add the bias values here
            # self.Output_Z = np.append(self.Output_Z, np.ones((self.Output_Z.shape[0], 1)), axis=1)
            self.Derivative_Fp = self.activation(self.Input_S, deriv=True).T
            ZW = self.Output_Z.dot(self.Weights_W)

            return( self.add_bias(ZW) )


class knn:
    def __init__(self, layer_config, minibatch_size=100, hidden_activation = HIDDEN_ACT_SPLUS, output_type = OUT_BINARY, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0, biasEnabled = True, friction = 0.0):
        self.layers = []
        self.num_layers = len(layer_config)
        self.minibatch_size = minibatch_size
        self.DEBUG = False
        self.DEBG_WeightGrads = [None] * (self.num_layers -1) # initialise an empty list with the same number of elements as Weight matrices
        self.outPutType = output_type # this is used to determine for debugging the cost function to be used
        self.regularizer = regularizer
        self.biasEnabled =biasEnabled
        self.Lambda = shrinkageParam
        self.friction = friction
        
        actfunct = None
        print("Net is regularized by: " + self.regularizer)
       
        # determine the activation function
        if hidden_activation == HIDDEN_ACT_SPLUS :
            actfunct = k_softplus
        elif hidden_activation == HIDDEN_ACT_SIG : 
            actfunct = k_sigmoid
        else :
            actfunct = k_leakyRELU 

        for i in range(self.num_layers-1):
            if i == 0:
                print("Initializing input layer with size {0}.".format( layer_config[i] ) + " / num inputs {0}.: ".format(layer_config[i]) + " / activation: "+ hidden_activation )
                # Here, we add an additional unit at the input for the bias
                # weight.
                self.layers.append(knnLayer([layer_config[i], layer_config[i+1] ], minibatch_size,activation=actfunct,is_input=True, biasEnabled = self.biasEnabled))
            else:
                print("Initializing hidden layer with size {0}.".format(layer_config[i]) + " / num inputs {0}.: ".format(layer_config[i]) + " / activation: "+ hidden_activation)
                # Here we add an additional unit in the hidden layers for the
                # bias weight.
                self.layers.append(knnLayer([layer_config[i], layer_config[i+1]],minibatch_size,activation=actfunct, biasEnabled = self.biasEnabled))

        print("Initializing output layer with size {0}.".format(layer_config[-1]) + " / num inputs {0}.: ".format(layer_config[-1]) + " / activation: "+ output_type )
        # determine the activation function
        if output_type == OUT_MULTICLASS :
                actfunct = k_softmax  # k_softmax # k_sigmoid
        elif output_type == OUT_REGRESSION :
            actfunct = k_linear
        else :
            actfunct = k_softmax  # if Binary, we still use Softmax as that has a linear derivative { actfunct = k_sigmoid }
      
        self.layers.append(knnLayer( [layer_config[-1], None],minibatch_size,is_output=True,activation=actfunct, biasEnabled = False)) # the output layer NEVER has any bias
        print("Done!")

    def forward_propagate(self, data):
        # We need to be sure to add bias values to the input
        # self.layers[0].Output_Z = np.append(data, np.ones((data.shape[0], 1)), axis=1)
        self.layers[0].Output_Z = data  # add the data WITHOUT the intercept

        for i in range(self.num_layers-1):
            self.layers[i+1].Input_S = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, yhat, labels):
        self.layers[-1].Error_D = (yhat - labels).T
        if self.num_layers >= 3 : # only do this if we have at least 1 hidden layer, otherwise we would attempt to loop forwards
            for i in range(self.num_layers-2, 0, -1):
                # We do not calculate deltas for the bias values
                #W_nobias = self.layers[i].Weights_W[0:-1, :]
                # self.layers[i].Error_D =  np.multiply(W_nobias.dot(self.layers[i+1].Error_D),self.layers[i].Derivative_Fp)
                self.layers[i].Error_D = self.layers[i].Weights_W.dot(self.layers[i+1].Error_D) * self.layers[i].Derivative_Fp


    # updates the Bias Weights separately ( if enabled)
    def update_bias_weights(self, i, num_samples_minibatch, eta):
        if(self.biasEnabled):
           # W_bias_grad = (self.layers[i+1].Error_D.dot(self.layers[i].Z_bias)).T
                          
            W_bias_grad = np.sum(self.layers[i+1].Error_D, axis =1) # via implicit multiplication of D by 1 ( IE row sums)
            W_bias_grad = W_bias_grad / num_samples_minibatch

            # add Mometum (velocity)
            self.layers[i].Bias_Momentum = self.friction * self.layers[i].Bias_Momentum - (eta*W_bias_grad)

            # bias gradients are NOT regularized, so we just apply them as is
            self.layers[i].Weights_bias += self.layers[i].Bias_Momentum 
             
        
    def update_weights(self, num_samples_total, eta ):
        num_samples_minibatch = self.layers[0].Output_Z.shape[0]
        for i in range(0, self.num_layers-1):
            W_grad = (self.layers[i+1].Error_D.dot(self.layers[i].Output_Z)).T
            W_grad = W_grad / num_samples_minibatch # normalise by number in minibatch, to keep the same as the cost function
            regularizers = self.getRegularizerGrad(i, num_samples_total) 
            W_grad = W_grad + regularizers # add regularizer term to gradients
            
          
            
            if self.DEBUG:  # if we are in Debug mode, we want to save away the current Weight gradients
                  self.DEBG_WeightGrads[i] = W_grad
              
            # add Mometum (velocity)
            self.layers[i].Momentum = self.friction * self.layers[i].Momentum - (eta*W_grad) 
           
            
            #self.layers[i].Weights_W = something -eta*W_grad                     
            self.layers[i].Weights_W += self.layers[i].Momentum
            
            # update bias/intercept terms separately (if needed)
            self.update_bias_weights(i,num_samples_minibatch, eta) # bias weights are not trained ...for now (I think this has to do with the fact that NNs have universal approx ability, as long as the function doesn't go through 0)

        
    def learn(self, train_data, train_labels, test_data, test_labels,num_epochs=500, eta=0.05, eval_train=False, eval_test=True, eval_freq = 100):

        num_samples_total = len(train_labels)*len(train_labels[0])
        
        print("Training for {0} epochs...".format(num_epochs) )
        for t in range(0, num_epochs):
            out_str = "[{0:4d}] ".format(t)
            
            # cosmetics: if we are classifying then we are evaluating via 'accuracy' where as if we are regressing, then we care about error
            evaluation =  "prediction (r^2)"
            if self.outPutType != OUT_REGRESSION :  
                evaluation = "accuracy (%)"


            for b_data, b_labels in zip(train_data, train_labels):
                output = self.forward_propagate(b_data)
                self.backpropagate(output, b_labels)
                self.update_weights(num_samples_total, eta=eta)

            if t % eval_freq == 0: # only evaluation fit every 100th or so iteration, as that is expensive
                if eval_train:
                    N_train = len(train_labels)*len(train_labels[0])
                    totals = 0
                    for b_data, b_labels in zip(train_data, train_labels):
                        output = self.forward_propagate(b_data)
      
                        # depending on if we are in a classification or regression problem, we evaluate performance differently
                        if self.outPutType != OUT_REGRESSION :  
                            currentRate = calc_Accuracy(output,b_labels )   # calculate accuracy, this is ROUNDED   
                        else :
                            # evaluate via mean average error
                            # residualSQ = np.sum ( (output - b_labels)**2 )  # residual Squared (Yhat - Y)^2
                            #currentRate =  np.sqrt(  residualSQ) # total error is the error so far plus the sum of the above
                              
                            # evaluate via r^2
                            currentRate = np.corrcoef(output,b_labels)[1,0]**2
                            N_train = len(train_labels) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
                        
                           
                        totals = totals +currentRate # sum in all minibatches
                        # sum of squares
                       # residuals =
                      #  totals += residuals
                      ### rounded prediction 
                      #  yhat = np.argmax(output, axis=1)
                      #  errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])
    
                    out_str = "{0} Training "+evaluation+": {1:.5f}".format(out_str,float(totals)/N_train)
    
    
                if eval_test:
                    N_test = len(test_labels)*len(test_labels[0])
                    #print("N_test is:" + str(N_test))
                    totals = 0
                    for b_data, b_labels in zip(test_data, test_labels):
                        output = self.forward_propagate(b_data)

                        # depending on if we are in a classification or regression problem, we evaluate performance differently
                        if self.outPutType != OUT_REGRESSION :  
                            currentRate = calc_Accuracy(output,b_labels )    # calculate accuracy, this is ROUNDED 
                        else :
                            # evaluate via mean average error
                            # residualSQ = np.sum ( (output - b_labels)**2 )  # residual Squared (Yhat - Y)^2
                            #currentRate =  np.sqrt(  residualSQ) # total error is the error so far plus the sum of the above
                              
                            # evaluate via r^2
                            currentRate = np.corrcoef(output,b_labels)[1,0]**2
                            N_test = len(test_labels) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
                          
                                                   
                        totals = totals +currentRate # sum in all minibatches

    
                    out_str = "{0} Test "+evaluation+": {1:.5f}".format(out_str,float(totals)/N_test)



            print(out_str)
            
    # the cost of the regularizer term  
    def getRegularizerCostTerm(self, num_samples_total):
        #num_samples = self.layers[0].Output_Z.shape[0]
        
        if self.regularizer == REGULARIZER_NONE: 
            return(0.0)
        elif self.regularizer == REGULARIZER_RIDGE: 
            return ( (self.Lambda/num_samples_total * 0.5) * np.sum( self.getWeightsAsVector()**2 ) )
        else : # LASSO
            return ( (self.Lambda/num_samples_total * 0.5) * np.sum( np.abs( self.getWeightsAsVector() ) ) )
        
        
    # the derivative of the regularizer term: have to normalise by the total number of samples in the total dataset, otherwise regularization would grow with smaller minibatches
    def getRegularizerGrad(self,i, num_samples_total):
        # num_samples = self.layers[0].Output_Z.shape[0]
        
        if self.regularizer == REGULARIZER_NONE: 
            return(0.0)
        elif self.regularizer == REGULARIZER_RIDGE: 
            return ( self.Lambda/num_samples_total * self.layers[i].Weights_W )
        else : # LASSO
            return ( self.Lambda/num_samples_total * np.sign(self.layers[i].Weights_W ) )  # this will cause the numerical check to fail.. it will still be small but much bigger than with L2 or without, this is probably due to the funky L1 derivative at 0
        
          
    ####### Debugger functions: used for numerical Gradient checking
    
    # returns all weights in the NN concated into a 1D array
    def getWeightsAsVector(self):
        allWeights = np.zeros(0)
        for i in range(0, self.num_layers-1):
           allWeights = np.concatenate( ( allWeights, self.layers[i].Weights_W.ravel() ) )
        return(allWeights)
    
    # replaces the weights in the network from an external source: a 1D array (IE the same format as that is returned by the above)
    def setWeights(self, allWeights):
        #Set W1 and W2 using single paramater vector.
        W_start = 0 # 1st weight's indices start at 0
        for i in range(0, self.num_layers-1): 
            W_end = W_start + self.layers[i].Weights_W.size
            self.layers[i].Weights_W = np.reshape(allWeights[W_start:W_end], (len(self.layers[i].Weights_W) , len(self.layers[i].Weights_W[0])))
            W_start = W_end # update the start position for next round

#    # returns the Sum of Squared errors ( IE this is the Cost function)
#    def getSumSquaredError(self, data, y):   #
#        #Compute cost for given X,y, use weights already stored in class.
#        yHat = self.forward_propagate(data)
#        J =  0.5*  np.sum((y-yHat)**2) 
#        return J
    
    def regressionCost_SumSquaredError(self, batch_data, y, num_samples_total):   #
        #Compute cost for given X,y, use weights already stored in class.
        yHat = self.forward_propagate(batch_data)
        batch_num_samples = batch_data.shape[0]  # normalise by the batch size
        J =  0.5*  np.sum((y-yHat)**2) / batch_num_samples + self.getRegularizerCostTerm(num_samples_total)
        return J
    

    def multiclassCost_softMax(self,batch_data, y, num_samples_total):
        yHat = self.forward_propagate(batch_data)
        batch_num_samples = batch_data.shape[0] # normalise by the batch size
        J = -np.multiply(y, np.log(yHat)).sum() / batch_num_samples + self.getRegularizerCostTerm(num_samples_total)
        return  J


    # returns the concated Weight Gradients from last learning iteration
    def getLastWeightGradients(self):
        allWeightGrads = np.zeros(0)
        for i in range(0, self.num_layers-1):
           allWeightGrads = np.concatenate( ( allWeightGrads, self.DEBG_WeightGrads[i].ravel() ) )
        return(allWeightGrads)
    

    def getCurrentWeightGradients(self, data, y, num_samples_total):
       yHat = self.forward_propagate(data)
       self.backpropagate(yHat, y) # we need o backpropagate also, as we need to know the CURRENT error, IE that is resulted from the weights that we have now (as the errors we have atm are the errors due to the previous iteration)
       origDEBUG = self.DEBUG
       self.DEBUG = True # temporarily set this, so that the next function saves the weight gradients
       self.update_weights(num_samples_total,eta=0) # this calculates the current weight GRADIENTS into an array, without actually updating them ( as eta is 0)
       
       allWeightGrads = np.zeros(0)
       for i in range(0, self.num_layers-1):
          allWeightGrads = np.concatenate( ( allWeightGrads, self.DEBG_WeightGrads[i].ravel() ) )
          
       self.DEBUG = origDEBUG # reset
       return(allWeightGrads)
    
    def gradientCheck(self, batch_data, y):
            weightsOriginal = self.getWeightsAsVector()  # gets all weights
            # init some empty vectors same num as weights
            
            numgrad = np.zeros(weightsOriginal.shape) # the numerical approximation for the derivatives of the weights
            perturb = np.zeros(weightsOriginal.shape) # perturbations: these are #1-of-k' style, where we have 0 for all else, except for the current
            e = 1e-4  # the perturbation
    
            num_samples_total = batch_data.shape[0] # the total number of samples are the same as the minibatch size, as gradient checks are only performed on signle minibatches
            # the costfunction differs based on if it is a regression NN or a multiclass classification
            costFunction = None
            if self.outPutType == OUT_REGRESSION :
                costFunction =  self.regressionCost_SumSquaredError # must use the '.self' otherwise we cannot get a reference to the function as a variable
            else :
                costFunction =  self.multiclassCost_softMax
              
    
            for p in range(len(weightsOriginal)): # go through each original weight
                #Set perturbation vector
                perturb[p] = e   # add a slight difference at the position for current weight
                
                # here we slightly change the current weight, and recalculate the errors that result
                self.setWeights(weightsOriginal + perturb) # add the changed weights into the neural net (positive offset)
                loss2 = costFunction(batch_data, y,num_samples_total) # get squared error: IE X^2  (IE this is the 'cost')
                self.setWeights(weightsOriginal - perturb) # add the changed weights into the neural net (negative offset)
                loss1 = costFunction(batch_data, y,num_samples_total) # get squared error: IE X^2  (IE this is the 'cost')
                #Compute Numerical Gradient
                numgrad[p] = (loss2 - loss1) / (2*e) # apply 'manual' formula for getting the derivative of X^2 -> 2X
    
                #Return the value we changed to zero:
                perturb[p] = 0 # we do this as in the next round it has to be 0 everywhere again
                
            #Return Weights to original value we have saved earlier:
            self.setWeights(weightsOriginal)
    
            return numgrad 
            
