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

import numpy as np
import random
import copy


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

LAYER_SUBTYPE_INPUT = "LAYER_SUBTYPE_INPUT"
LAYER_SUBTYPE_OUTPUT = "LAYER_SUBTYPE_OUTPUT"
LAYER_SUBTYPE_HIDDEN = "LAYER_SUBTYPE_HIDDEN"



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
    
def k_linear(X, deriv=False):
    if not deriv:
        return X
    else:
        if len( X.shape ) == 1 : return np.ones(X.shape[0])    
        else: return np.ones((X.shape[0], X.shape[1]))

   
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



class knn:
    def __init__(self):
        self.layers = []
        self.num_layers = 0
        self.DEBUG = False
        self.CAN_ADD_LAYERS = True

    def addLayer(self,layer) :  # adds a layer to the neural networks list of layers
        if self.CAN_ADD_LAYERS : 
            self.num_layers = self.num_layers+1
            self.layers.append(layer)
            print( "layer " + type(layer).__name__ +" is added to Network " )
        
        
     
    def connectLayers(self, initWeights = True) :   # connects up each layer to its next, and initialises the Weighted layers with the correct dimension matrices  
        # the first layer only has a next layer
        self.layers[0].initConnections( {"PREV":None, "NEXT":self.layers[1] }, initWeights )
          
        if self.num_layers > 2 : # if there  are any hidden layers
            
            for i in range( 1, (self.num_layers-1) ):  # go through all layers except last
               self.layers[i].initConnections(  {"PREV":self.layers[i-1], "NEXT":self.layers[i+1] }, initWeights )
            

        # connect Output
        self.layers[-1].initConnections( {"PREV":self.layers[-2], "NEXT":None}, initWeights  ) # the last layer only has a prev layer
           
        print("All layers created, Neural Network ready")
        

    # Main forward propagation algo: it usually return Sigma(XB), and feeds it int the next layer's input
    def forward_propagate(self, data): # data is the design matrix
        return( self.layers[0].forward_propagate(data) )# this recursively calls all subsequent layers' forward propagate until the last (the output, which produces the yhat)
    
    
    # calculates the cumulative errors (Delta) at each node, for all layers
    def backpropagate(self, yhat, labels):
        labels = labels.reshape(labels.shape[0],-1)
        # calculate first Error Delta directly from output
        Error_Delta_Output = (yhat - labels).T
        #print ("Error_Delta_Output dims: " + str(Error_Delta_Output.shape)  + " // yhat dims: " + str(yhat.shape) + " // labels dims: " + str(labels.shape) )
        self.layers[-1].backpropagate(Error_Delta_Output) # call this on the last layer, which then will recursively call all the rest all the way back to the first
        
        
    # goes through and updates each layer (for ones that have weights this updates the weights, for others this may do something else or nothing)
    def update(self, eta, minibatch_size, friction ):
        for i in range(1, self.num_layers):       # loop from the 2nd layer to the last ( the 1st layer is the input data, and that cannot be updated)
            self.layers[i].update(eta, minibatch_size, friction)

        
    def learn(self, train_X, train_Y, test_X, test_Y,num_epochs=500, eta=0.05, eval_train=False, eval_test=True, eval_freq = 100,  friction = 0.0):
        print ( "Starting to train Neural Network for for num iterations: " + str(num_epochs)  )
        minibatch_size = train_Y[0].shape[0] # this refers to the number of rows for matrices, and the length for vectors
        
        if len( train_Y[0].shape ) == 1 : outPutType = OUT_REGRESSION # if the thing we are trying to predict has only 1 column, then it is a regression problem
        else : outPutType = OUT_MULTICLASS                        
        
        # cosmetics: if we are classifying then we are evaluating via 'accuracy' where as if we are regressing, then we care about error
        evaluation = "prediction (r^2)"
        if outPutType != OUT_REGRESSION :  evaluation = "accuracy (%)"                      
        
        results = {}
        results["epochs"] = list()
        if eval_train: results["train_accuracy"]  = list()
        if eval_test: results["test_accuracy"]  = list()

        for t in range(0, num_epochs):
            out_str = "it: "  + str(t) # "[{0:4d}] ".format(t)

            # 1) Complete an entire training cycle: Forward then Backward propagation, then update weights, do this for ALL minibatches in sequence
            for b_data, b_labels in zip(train_X, train_Y):
                Yhat = self.forward_propagate(b_data)
                self.backpropagate(Yhat, b_labels)
                self.update(eta, minibatch_size, friction)


            if t % eval_freq == 0: # only evaluation fit every 100th or so iteration, as that is expensive
                results["epochs"].append(t) # add the epoch's number
                
                if eval_train:
                    N_train = len(train_Y)*len(train_Y[0])
                    #print("N_train is:" + str(N_train))
                    totals = 0
                    for b_data, b_labels in zip(train_X, train_Y):
                        yhat = self.forward_propagate(b_data)
                        b_labels = b_labels.reshape(b_labels.shape[0],-1) # force 2D structure
                        # depending on if we are in a classification or regression problem, we evaluate performance differently
                        if outPutType != OUT_REGRESSION :  
                            currentRate = calc_Accuracy(yhat,b_labels )   # calculate accuracy, this is ROUNDED   
                        else :
                            # evaluate via mean average error
                            # residualSQ = np.sum ( (yhat - b_labels)**2 )  # residual Squared (Yhat - Y)^2
                            #currentRate =  np.sqrt(  residualSQ) # total error is the error so far plus the sum of the above
                              
                            # evaluate via r^2
                            currentRate = np.corrcoef(yhat,b_labels, rowvar=0)[1,0]**2
                            N_train = len(train_Y) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
                        
                           
                        totals = totals +currentRate # sum in all minibatches
                        # sum of squares
                       # residuals =
                      #  totals += residuals
                      ### rounded prediction 
                      #  yhat = np.argmax(yhat, axis=1)
                      #  errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])
                    accuracy = round( float(totals)/N_train,5)
                    results["train_accuracy"].append(accuracy)
                    out_str =  out_str + " / Training " +evaluation +": "+ str( accuracy )
    
                if eval_test:
                    N_test = len(test_Y)*len(test_Y[0])
                    #print("N_test is:" + str(N_test))
                    totals = 0
                    for b_data, b_labels in zip(test_X, test_Y):
                        yhat = self.forward_propagate(b_data)
                        b_labels = b_labels.reshape(b_labels.shape[0],-1) # force 2D structure
                        
                        # depending on if we are in a classification or regression problem, we evaluate performance differently
                        if outPutType != OUT_REGRESSION :  
                            currentRate = calc_Accuracy(yhat,b_labels )    # calculate accuracy, this is ROUNDED 
                        else :
                            # evaluate via mean average error
                            # residualSQ = np.sum ( (yhat - b_labels)**2 )  # residual Squared (Yhat - Y)^2
                            #currentRate =  np.sqrt(  residualSQ) # total error is the error so far plus the sum of the above
                              
                            # evaluate via r^2
                            currentRate = np.corrcoef(yhat,b_labels, rowvar=0)[1,0]**2
                            #print("currentRate is:" + str(currentRate))
                            N_test = len(test_Y) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
                          
                                                   
                        totals = totals +currentRate # sum in all minibatches

                    accuracy = round( float(totals)/N_test,5)
                    results["test_accuracy"].append(accuracy)
                    out_str =  out_str + " / Test " +evaluation +": "+  str( accuracy )


            print(out_str, flush=True)
            
        return ( { "results" : results})
            
            
    ####### Debugger functions: used for numerical Gradient checking 
    # replaces the weights in the network from an external source: a 1D array (IE the same format as that is returned by getWeightsAsVector() )
    def setWeights(self, allWeights):
        for i in range(1, self.num_layers):  # go through all layers except first, as that cannot have weights
            allWeights = self.layers[i].addDebugData(allWeights) # if layer has weights, then this function takes as many as needed to fill up layer weight matrix, adds them , and removes them from the list before passing back what is left
               

    def regressionCost_SumSquaredError(self, batch_data, y):   #
        #Compute cost for given X,y, use weights already stored in class.
        yHat = self.forward_propagate(batch_data)
        y = y.reshape(y.shape[0],-1) # force 2D structure
        batch_num_samples = batch_data.shape[0]  # normalise by the batch size
        J =  0.5*  np.sum((y-yHat)**2) / batch_num_samples + self.getRegularizerCostTerms()
        return J
    

    def multiclassCost_softMax(self, batch_data, y):
        yHat = self.forward_propagate(batch_data)
        y = y.reshape(y.shape[0],-1) # force 2D structure
        batch_num_samples = batch_data.shape[0] # normalise by the batch size
        J = -np.multiply(y, np.log(yHat)).sum() / batch_num_samples + self.getRegularizerCostTerms()
        return  J


    # goes through and retreives the regularizer cost terms from all layers (Which have it)
    def getRegularizerCostTerms(self) :
        allTerms = np.zeros(0)
        for i in range(0, self.num_layers): # go through all layers, and get their weights 
            allTerms = self.layers[i].getDebugInfo(allTerms, Type = REG_COST) # if layer has weights, it adds it into the array, if not just simply returns the orig
        
        return(np.sum (allTerms) )
    
    
    def getCurrentWeightGradients(self, data, y):
        minibatch_size = y.shape[0]
        yHat = self.forward_propagate(data)
        self.backpropagate(yHat, y) # we need o backpropagate also, as we need to know the CURRENT error, IE that is resulted from the weights that we have now (as the errors we have atm are the errors due to the previous iteration)
        origDEBUG = self.DEBUG
        self.DEBUG = True # temporarily set this, so that the next function saves the weight gradients
        self.update(0, minibatch_size, 0.0) # this calculates the current weight GRADIENTS into an array, without actually updating them ( as eta is 0)
                                                # must disable Friction, otherwise gradient checks will always fail
        allWeightGrads = np.zeros(0)
        for i in range(0, self.num_layers): # go through all layers, and get their weights gradients 
            allWeightGrads = self.layers[i].getDebugInfo(allWeightGrads, Type = GRAD) # if layer has weight grads, it adds it into the array, if not just simply returns the orig
          
        self.DEBUG = origDEBUG # reset
        return(allWeightGrads)
   

    # gets the current weights across all layers in a 1D vector
    def getWeightsAsVector(self) :   
        allWeights = np.zeros(0)
        for i in range(0, self.num_layers): # go through all layers, and get their weights 
            allWeights = self.layers[i].getDebugInfo(allWeights, Type = WEIGHT ) # if layer has weights, it adds it into the array, if not just simply returns the orig
        
        return(allWeights)
       
    
    def gradientCheck(self, batch_data, y):
        if len( y.shape ) == 1 : outPutType = OUT_REGRESSION  # if the thing we are trying to predict has only 1 column, then it is a regression problem      
        else : outPutType = OUT_MULTICLASS     

        y = y.reshape(y.shape[0],-1) # force 2D structure
        weightsOriginal = self.getWeightsAsVector()  # gets all weights
        # init some empty vectors same num as weights
        
        numgrad = np.zeros(weightsOriginal.shape) # the numerical approximation for the derivatives of the weights
        perturb = np.zeros(weightsOriginal.shape) # perturbations: these are #1-of-k' style, where we have 0 for all else, except for the current
        e = 1e-4  # the perturbation

        #num_samples_total = batch_data.shape[0] # the total number of samples are the same as the minibatch size, as gradient checks are only performed on signle minibatches
        # the costfunction differs based on if it is a regression NN or a multiclass classification
        costFunction = None
        if outPutType == OUT_REGRESSION :
            costFunction =  self.regressionCost_SumSquaredError # must use the '.self' otherwise we cannot get a reference to the function as a variable
        else :
            costFunction =  self.multiclassCost_softMax
          

        for p in range(len(weightsOriginal)): # go through each original weight
            #Set perturbation vector
            perturb[p] = e   # add a slight difference at the position for current weight
            
            # here we slightly change the current weight, and recalculate the errors that result
            self.setWeights(weightsOriginal + perturb) # add the changed weights into the neural net (positive offset)
            loss2 = costFunction(batch_data, y) # get squared error: IE X^2  (IE this is the 'cost')
            self.setWeights(weightsOriginal - perturb) # add the changed weights into the neural net (negative offset)
            loss1 = costFunction(batch_data, y) # get squared error: IE X^2  (IE this is the 'cost')
            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e) # apply 'manual' formula for getting the derivative of X^2 -> 2X

            #Return the value we changed to zero:
            perturb[p] = 0 # we do this as in the next round it has to be 0 everywhere again
            
        #Return Weights to original value we have saved earlier:
        self.setWeights(weightsOriginal)

        return numgrad 
        




#This only exists in R, as we need class fields that reference 'knnBaseLAyer' in the knnBaseLayer class itself and R doesn't like that
class knnDummyLayer :
    def __init__(self, iparams = -1, iparentNetwork = None ):
        self.params=iparams
        #self.isNull=True
        self.parentNetwork= iparentNetwork
        
    def generateOutput(self,Input) : ...
    def forward_propagate(self,Input) : ...   
    def calcError(self,Input) : ...   
    def backpropagate(self,Input) : ...   
    def update(self, eta, minibatch_size, friction) : ...
    def initConnections(self, prevNext = {}, initWeights = True ) : ...
    def getDebugInfo(self,dataSoFar, Type) : return(dataSoFar)  # the base method still needs to return the passed in data if nothing else   
    def addDebugData(self,allWeights)  : return(allWeights)   # the base method still needs to return the passed in data if nothing else
    


# base class for all knn layers: should never be directly instantiated
class knnBaseLayer(knnDummyLayer) :

    def __init__(self, iparams, iparentNetwork):
        super().__init__(iparams, iparentNetwork)
        #params="numeric",
        #parentNetwork="knn",
        self.prevLayer = None  # this should be 'knnBaseLayer', but R cannot reference this type of class in the class definition, so we have to use a 'dummy' class
        self.nextLayer = None
        # params <<- iparams
        # parentNetwork <<- iparentNetwork
        #self.isNull = False # R: only for an instantiated layer, we set this to FALSE to let others know this really exists...
    
      
        # add this layer to its parent Neural Network's list of layers
        self.parentNetwork.addLayer(self)

    
    # produces output from the layer's data, this is used by forward_propagate
    def generateOutput(self,Input) : ... 

    
    # passes along the output of this layer to the next, if any 
    def forward_propagate(self,Input) :
        output = self.generateOutput(Input)
          
        if self.nextLayer is None :  # if there is no next layer, IE this is the last one the Output
            return(output)
        else : # if there are any next layers still, we recursively call them
            return( self.nextLayer.forward_propagate(output) ) # send output from this layer to the next
        
    
    # computes the Error Delta of the current layer, based off from the error passed back from the layer after this during backpropagation
    def calcError(self,Input) : ...
    
                 
    # passes errors backwards from the output onto preceding layers
    def backpropagate(self,Input) : # this receives the Error_Delta from the layer after this   
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            # generate output rom this layer
            error_D = self.calcError(Input) # this is USUALLY really just the Error_D, but the Joiner layer for instance, passes along the weights of the next layer too
            return( self.prevLayer.backpropagate(error_D) )
             # else: if there are no prev layers then stop backpropagating ... IE we reached the INPUT layer
     

    # generic update function, (it usually updates weights for layers that have them)
    #def update(self,eta, minibatch_size, friction) : ...
    
    # Lets each layer know about its neighbours: the previous and next layer in the stack: this is called once the entire network has been set up
    def initConnections(self, prevNext = {}, initWeights = True ) :
        # check previous layer
        if prevNext["PREV"] is not None : self.prevLayer = prevNext["PREV"] 
         
        # check next layer
        if prevNext["NEXT"] is not None : self.nextLayer = prevNext["NEXT"]
        


# specialised layer type used by Convolutional topologies:  takes input, and splits it into a set of predefined regions, and passes these along to the next layer
# params: [totalNumUnitsinLayer, regionStart1, regionEnd1, regionStart2, regionEnd2, regionStart3....]
class knnSplitter(knnBaseLayer) :
    def __init__(self, iparams, iparentNetwork):
        super().__init__(iparams, iparentNetwork) # call init on parent
        self.regions = list() # list of arrays describing the regions: where [0] is the start, and [1] is the last unit in a region
        self.Error_D = None
        self.Input_S = None
      
        # parse params into regions
        self.numRegions= int ( (len(self.params) -1) /2  ) # 1st element sets the number of regions we will need, so the number of regions will be half the length of the params-1 (as we have 2 values / region)
                              # params[1] = a = # units in current layer;  need to keep the 1st element being the total number of units in this layeer (including in all regions), so that other layers can access this information the same way as for any other layers)
        regionStart = int(0)
        for i in range( 1, len(self.params),2 ): # loop from 2nd element, where two consecutive elements define the start/end of region
            regionStart = int(self.params[i])
            regionEnd = int(self.params[i+1])
            self.regions.append( np.array([regionStart, regionEnd]) )
            

    # Splitter splits an incoming Input data into the parameters' predefined set of regions
    def generateOutput(self, Input) :
        self.Input_S = Input # save input
          
        subMatrices = list()
        for i in range(self.numRegions): # go through all regions we are meant to be having
            regionStart = self.regions[i][0]
            regionEnd = self.regions[i][1]
            subMatrices.append(  Input[:,regionStart:regionEnd] ) # cut the Input, into submatrices in their cols (as we are splitting by predictors)

        return(subMatrices) # return the splits
  

    # Splitter performs the opposite function for backpropagation: it Joins an incoming list error Delta submatrices into a single matrix
    def calcError(self, Input) :
      
        # input here is a list()  of all the Error_D s from the next layer (the Convolution1D)
        new_Error_D = None
        for i in range(len(Input)):
            if(new_Error_D is None) : new_Error_D =  Input[i]
            else : new_Error_D = np.row_stack( (new_Error_D, Input[i]) ) # as the Di submatrices s are transposed, so we join them by adding them under each other
                               
        self.Error_D = new_Error_D
        return(self.Error_D)



# specialised layer type used by Convolutional topologies:  takes input, and joins them up, and passes these along to the next layer
# params: [totalNumRegions, lastUnitInRegion1, lastUnitInRegion2,...,, lastUnitInRegionn]
class knnJoiner(knnBaseLayer) :
    def __init__(self, iparams, iparentNetwork):
        super().__init__(iparams, iparentNetwork) # call init on parent
        self.Error_D = None # this refers to the ENTIRE Error_D of the layer after this 
        self.nextLayerWeights = None # list of the weights that were split into regions of the layer after this
        # self.nextLayerWeight_bias = None # this is unused, we don't transfer this, although could be sent forward to Splitter, which could then add this into the D it joined... (but as bias weights are generally not used to calculate the Error Deltas this is fine)
        self.Output_Z = None 
        self.prevLayerOutputs = None # the outputs of a Conv layer

        # parse params into regions
        self.numRegions = len(self.params) -1 # the number of regions, is the length of the array minus the 1st element: 
        # params[1] total number of units in the previous layer (this may NOT be the number of regions, if there are multiple units that represent each region), need to keep this reflecting his info, so that we can access it outside the same way as any other layer
           
        self.regions = list() # list of arrays describing the regions: where [0] is the start, and [1] is the last unit in a region
        regionStart = int(0)
        for i in range( 1, len(self.params) ): # loop from 2nd element, where each element is the LAST predictor in matrix
            regionEnd = int( self.params[i] +1 )
             
            self.regions.append(  np.array([ regionStart, regionEnd]) )
            regionStart = regionEnd #the next region's start is just the one after this ended
         
     
    # Joiner joins an incoming list of submatrices; outputs from a convolutional layer, into a single output
    def generateOutput(self,Input) : # Joiner gets a list of outputs, that it will merge
        self.prevLayerOutputs = Input # save input   
        output = None
        
        for i in range(len(self.prevLayerOutputs)):
            if(output is None) : output = self.prevLayerOutputs[i]
            else : output = np.column_stack( (output, self.prevLayerOutputs[i]) ) # as we split by predictors (cols) we join them by placing them next to each other
               
        self.Output_Z = output # save output too
        return(self.Output_Z)
     
     
    # Joiner performs the opposite function for backpropagation: it splits the incoming Weights, and passes along the entire Error Delta
    def calcError(self,Input) :
        self.Error_D = Input #  save the entire Error_D  of next layer, which is NOT split, but will be used to multiply each weight by
         
        # get the Weight of the next layer, this will be needed by the Conv1D layer that follows this
        weightOfNextLayer = self.nextLayer.Weights_W
           
        self.nextLayerWeights = list()
        
        for i in range(self.numRegions):
            regionStart = self.regions[i][0]
            regionEnd = self.regions[i][1]
            #print("regionStart: " + str(regionStart) +  " / regionEnd: " + str(regionEnd) + " out of shape of weightOfNextLayer: " + str(weightOfNextLayer.shape) )
            
            # need to force python to treat data as 2D:
            weights = weightOfNextLayer[regionStart:regionEnd]
            weights = weights.reshape(weights.shape[0],-1)
            self.nextLayerWeights.append( weights ) # cut the submatrix, as the Error Deltas are transposed at this stage we spit by the rows, to get the effecot of splitting by the predictors
                 
        return( [self.Error_D, self.nextLayerWeights] ) # the error itself has not changed, but we also pass along the next layer's weights split into a list of submatrices for each region
       


# specialised layer type used by Convolutional topologies:  takes input, and joins them up, and passes these along to the next layer
# params: [totalNumRegions, lastUnitInRegion1, lastUnitInRegion2,...,, lastUnitInRegionn]
# [numSamplesinBatch(n), currentLayerUnits(a), nextLayerUnits(b)]
class knnConv1D(knnBaseLayer) :
    def __init__(self, iparams, iparentNetwork, inumUnitsInregions, inumUnitsInPrevLayer_Pi, iregionRegularizers, iregionShrinkageParams):
        super().__init__(iparams, iparentNetwork) # call init on parent
        self.Error_D = None # this refers to a list of 
        self.Input_S =  None # list of matrices of outputs of the previous layer
   
        # parse params into regions
        self.numRegions = self.params[0] # params only has 1 element for Conv1D
        self.regions = list() # list of knn layers, 1 for each region  
        self.numUnitsInregions = inumUnitsInregions # how many units are in each region
        self.numUnitsInPrevLayer_Pi = inumUnitsInPrevLayer_Pi # vector of each of the number of units,  (IE columns/predictors/weights) in previous layer's regions
        self.regionRegularizers = iregionRegularizers # list of the type of regularizers for each region (layer)
        self.regionShrinkageParams = iregionShrinkageParams # each region is allowed to have its own lambda
         
        #dummyKnn = knn() # as all layers must belong to a Knet, we need to create a dummy one here (this is a Hacky fix...)
        self.parentNetwork.CAN_ADD_LAYERS = False # as all layers must belong to a Knet, (as they derive overall network wide properties such as DEBUG), but nested layers inside Conv1Ds should not be added to the main network flow, so we disable that
        for i in range(self.numRegions):
           # print("self.numUnitsInregions[i]: " + str(self.numUnitsInregions[i]))
            # create layer
            self.regions.append( knnLayer(  np.array([ self.numUnitsInregions[i] ] ) , self.parentNetwork, LAYER_SUBTYPE_HIDDEN, iactivation=k_linear, regularizer = self.regionRegularizers[i], shrinkageParam = self.regionShrinkageParams[i]) )
        #                         knnLayer(      b                       , myNet             , LAYER_SUBTYPE_HIDDEN, iactivation=k_linear, regularizer = REGULARIZER_NONE          , shrinkageParam = 0.0)
            # need to set up Connections of each layer that represents a region, but without having to rely on prevLayer. as there won't be', so we directly set the weight matrix' dims
            # will need to make sure that prev/next layer won't exist
            self.regions[i].initWeightMatrix(self.numUnitsInPrevLayer_Pi[i])
                
        self.parentNetwork.CAN_ADD_LAYERS = True # re enable this
        

    # goes through a list of mini Weighted layers, and collects their output, which is then sent forward
    def generateOutput(self,Input) : # Convolution1D gets a list of inputs (Xi) from a Splitter
        self.Input_S = Input # save away the list of subMatrices
         
        convOutput = list()
        for i in range(self.numRegions): # go through each region's layer, and make each output its bit
           convOutput.append(self.regions[i].forward_propagate(Input[i]) )

        return(convOutput)

               
    # update here delegates the updating of the weights to each of its regions' layers
    def update (self, eta, num_samples_minibatch, friction) :
        for i in range(self.numRegions): # go through each region's layer, and force them to 
           self.regions[i].update(eta, num_samples_minibatch, friction)
  
      
    # Conv1D gets a single matrix for Error_D and a list of weights, which come from the next FC(Fully Connected) layer (split into matching parts via the Joiner)
    def calcError(self, Input) :
        Error = Input[0]
        nextLayerWeights = Input[1]

        convOutput = list()
        for i in range(self.numRegions):
            # each region's layer calculates its D, from the previous D, and the relevant submatrix of W
            convOutput.append( self.regions[i].calcError(Error, nextLayerWeights[i] ) ) # this performs Di_part = W_NEXT * t(D_FC_all) * Fp
                              # here we override the weights, by passing one in directly (this is kindof hacky)

        self.Error_D = convOutput # save away ALL the Error_Ds of all the regions
        return(self.Error_D) # return the splits
          
           
    # returns all weights in ALL the region's layers into a 1D array
    def getWeightsAsVector(self) : # a Conv layer just goes through ALL of its mini layer's weights
        allWeights = np.zeros(0)
        for i in range(self.numRegions):
                allWeights = np.concatenate( (  allWeights, self.regions[i].getWeightsAsVector() ) )
        
        return(allWeights)
    
      
    # returns all weight Gradients in ALL the region's layers into a 1D array
    def getWeightsGradsAsVector(self) :  # a Conv layer just goes through ALL of its mini layer's weights
        allWeightGrads = np.zeros(0)
        for i in range(self.numRegions):
            allWeightGrads = np.concatenate( (  allWeightGrads, self.regions[i].DEBUG_WeightGrads.ravel() ) )
      
        return(allWeightGrads)
     
      
    def getRegularizerCostTerm(self) : # a Conv layer just goes through ALL of its mini layer's regularizer cost terms
        allCostTerms = np.zeros(0)
        for i in range(self.numRegions):
            allCostTerms = np.concatenate( (  allCostTerms,self.regions[i].getRegularizerCostTerm().ravel() ) )
       
        return(allCostTerms)
       
          
    # concats the weights of this layer (if it has any), into the ones we got so far
    def getDebugInfo(self, dataSoFar, Type = WEIGHT) :
        if(Type == WEIGHT ) :  #if we are NOT looking for weight gradients, then we want the weights
             return(  np.concatenate( (dataSoFar, self.getWeightsAsVector()) ) )
        elif (Type == GRAD  ) : # if we do want the gradients then we use the ones saved during debug
            return(  np.concatenate( (dataSoFar, self.getWeightsGradsAsVector() ) ) )
        else : # if we want  the regularizer cost terms
            return(  np.concatenate( (dataSoFar, self.getRegularizerCostTerm() ) ) )
        
 
    def addDebugData(self, allWeights) :
        for i in range(self.numRegions): # go through all layers/regions
              allWeights = self.regions[i].addDebugData(allWeights) # if layer has weights, then this function takes as many as needed to fill up layer weight matrix, adds them , and removes them from the list before passing back what is left
        
        return(allWeights)

        

# the main 'Weighted' neural network layer, used by all FC (Fully Connected) layers
# params: a scalar, it is simply just the number of units in his layer
class knnLayer(knnBaseLayer):
    def __init__(self, iparams, iparentNetwork, isubtype, iactivation=None, ibiasEnabled = True, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0):
        super().__init__(iparams, iparentNetwork) # call init on parent
        
        self.Weights_bias = None
        self.Momentum = None
        self.Bias_Momentum = None
        self.regularizer = regularizer
        self.subtype = isubtype # as we have 3 different subtypes, for Input/Output and Hidden, we need to differentiate it via this flag
        self.Lambda = shrinkageParam
        self.biasEnabled = ibiasEnabled
        self.layer_numUnits = self.params[0] # here params has just 1 element  # number of neurons in layer
         
        # The activation function is an externally defined function (with a derivative)
        self.activation = iactivation
        self.Output_Z= None # Z is the matrix that holds output values
        self.Weights_W = None # weights are stored as 'incoming' IE they refer to the weights between this layer and the one before
        self.Input_S = None # S is NULL matrix that holds the inputs to this layer
        self.Error_D = None # D is the matrix that holds the deltas for this layer
        self.Derivative_Fp = None # Fp is the matrix that holds the derivatives of the activation function applied to the input
        self.DEBUG_WeightGrads = None # holds the debug weights for this layer (only used if parent layer has DEBUG == TRUE)
         
        print( "layer "+self.subtype+" is regularized by: " + self.regularizer+ " / its params are ", str(self.params) )
     
          
    # sets up relationships to neighbouring layers, and if we have previous layer (IE it is not an input or a nested minilayer in a Conv), then we can init the weight matrix to the correct dimension
    def initConnections(self, prevNext = {}, initWeights = True ): 
        super().initConnections(prevNext)
        
        # Weights can be initialised only by knowing the number of units in the PREVIOUS layer (but we do NOT need to know the size of the minibatches (IE n), as Weight matrix' dimension does not depend on that
        if (self.prevLayer is not None and initWeights) : # IE the we are testing if, subtype != LAYER_SUBTYPE_OUTPUT
            prevLayer_size = self.prevLayer.params[0] # find out how big the next layer is 
            self.initWeightMatrix(prevLayer_size)

     
    # (this is usually called from initConnections(), but this might also be called directly from outside, by the conv layer for example)
    def initWeightMatrix(self, prevLayer_size) :
       # print("initWeightMatrix for subtype: "+ str(self.subtype) + " / prev layer has size: "+ str(prevLayer_size) + " / self.layer_numUnits: " +str(self.layer_numUnits) )
        # Weights are sized as: rows: number of units in previous layer, cols: number of units in current layer (IE the minibatch size doesnt matter) 
        self.Weights_W = np.random.normal(size=[ int(prevLayer_size),self.layer_numUnits], scale=np.sqrt(2.0 / prevLayer_size )) # modified xavier init of weights for RELU/softplus (this also has the effect of basic L2 regularization)
        self.Momentum = np.zeros((self.Weights_W.shape[0], self.Weights_W.shape[1])) # stores a 'dampened' version of past weights (IE its an older version of the above with the same dimensions)
            
        if self.biasEnabled :  # if we have bias/intercept, we add a row of Weights that we keep separate
            self.Weights_bias = np.random.normal(size=(1,self.Weights_W.shape[1]), scale=np.sqrt(2.0 / prevLayer_size ))  
            self.Bias_Momentum = np.zeros( (1,self.Weights_bias.shape[0]) ) # stores a 'dampened' version of past weights  for intercepts

                         
    # performs the addition of the bias terms effect  onto the output
    def add_bias(self, ZW) :
        if(self.biasEnabled == False) : return(ZW) # if bias wasn't enabled in the first place we just return the orig

        ZWb = ZW + self.Weights_bias # we simply add the bias to every row, the bias is implicitly multipliedby 1
        return(ZWb) 
    
         
    # as knn layer can be 1 of 3 subtypes, we have to produce an output for forward propagation differently
    def generateOutput(self, Input) :
        # if its the first layer, we just return the data that was passed in
        if (self.subtype == LAYER_SUBTYPE_INPUT) :
            self.Output_Z = Input
            return( self.Output_Z  ) 
               
            # if its NOT an input, then all FC layers will have weights (even Output, as weights are now stored as 'incoming' weights between this and prev layer)
        else : 
            self.Input_S  = Input # save this away for later access
            self.Output_Z = self.Input_S.dot(self.Weights_W)  # the output is constructed by first multiplying the input by the weights
            
       
        self.Output_Z = self.add_bias(self.Output_Z) # we then add the intercept (the linear predictor is now complete)
         
        # non Output subtype layers need to figure out the rate of change in the output (this is redundant, if we are just making predictions, IE backpropagation does NOT follow this)
        if (self.subtype != LAYER_SUBTYPE_OUTPUT) :  self.Derivative_Fp = self.activation(self.Output_Z, deriv=True).T  # this is transposed, as D is also transposed during back propagation
         
        # output is completed by passing the linear predictor through an activation (IE we squash it through a sigmoid)
        self.Output_Z = self.activation(self.Output_Z) # if its a hidden layer (or output), we need to activate it
         
        return ( self.Output_Z )
         
    
    # computes the accumulated error up until this layer, this usually happens by 
    # each layer's Delta, is calculated from the Delta of the Layer +1 outer of it, and this layer's Weights scaled by the anti Derivative Gradient
    def calcError(self, Input, customWeight = None) :
        #print("BEFORE Input dims: " + str(Input.shape) )
        Input = Input.reshape(Input.shape[0],-1)
        #print("AFTEr Input dims: " + str(Input.shape) )
         
        if (self.subtype == LAYER_SUBTYPE_OUTPUT) :
           # print("Output Error dims: " + str(Input.shape) )
            self.Error_D = Input # this is stored as Transposed D ()
        else : # 'input' here basically refers to 'nextLayer$Error_D'
           
            #    to get the current layer's ErrorDelta, we need the NEXT layer's weight, this can be usually directly accessed
            # , except if this is a 'mini' layer nested in a convolutional layer, in that case it had to be directly passed in as a 'customWeight'
            if( customWeight is None ) : weightToUse = self.nextLayer.Weights_W
            else : weightToUse = customWeight
           # print("weightToUse dims: " + str(weightToUse.shape) + " / Input dims: " + str(Input.shape) + " / Derivative_Fp dims: " + str(self.Derivative_Fp.shape))
            self.Error_D = weightToUse.dot(Input) * self.Derivative_Fp # the current layer's Delta^T = (NextWeights_W * D_next^T) *Schur* F^T  
            
        return(self.Error_D)
      

    # updates the Bias Weights separately ( if enabled)
    def update_bias_weights(self, num_samples_minibatch, eta, friction) :
        if(self.biasEnabled == True) :
           
            W_bias_grad = np.sum(self.Error_D, axis =1)  # via implicit multiplication of D by 1 ( IE row sums)
            W_bias_grad = W_bias_grad / num_samples_minibatch
               
            # add Mometum (velocity)
            self.Bias_Momentum = friction * self.Bias_Momentum - (eta*W_bias_grad)
               
            # bias gradients are NOT regularized, so we just apply them as is
            self.Weights_bias =  self.Weights_bias + self.Bias_Momentum

        
    # updates the weights (including intercept) for current layer by calculating a 'change in weights': this is basically a scaled version of Error*Input
    # we scale by: learning rate(eta) and number of samples in minibatch ( to ensure consistent updates between different minibatch sizes)
    # this is the stage where we add regularizers too: we scale those by the 'total number of samples in ALL minibatches' (kindof redundant)
    # finally entire update is applied via 'momentums' we build up acceleration from previous updates, so if we keep moving int othe same direction then we can go(IE learn) faster and faster
    def update(self, eta, num_samples_minibatch, friction) :
        if(self.Weights_W is not None) : # if it has any outgoing weights (IE not an input that feeds into a splitter)
            W_grad = (self.Error_D.dot(self.Input_S) ).T # the weight changes are (D_next^t * Input_current) and transposed back so dims match
            W_grad = W_grad / num_samples_minibatch # normalise by N, to keep the same as the cost function
               
            regularizers = self.getRegularizerGrad()
               
            W_grad = W_grad + regularizers
               
            if (self.parentNetwork.DEBUG == True) : # if we are in Debug mode, we want to save away the current Weight gradients
               self.DEBUG_WeightGrads = W_grad
             
            # add Mometum (velocity)
            self.Momentum = friction * self.Momentum - (eta*W_grad) # total update is: (dampened) pastUpdates + current update
               
            self.Weights_W = self.Weights_W +  self.Momentum
               
            # update bias/intercept terms separately (if needed)
            self.update_bias_weights(num_samples_minibatch, eta,friction)  # bias weights are not trained ...for now (I think this has to do with the fact that NNs have universal approx ability, as long as the function doesn't go through 0)
            
       
    def getRegularizerGrad(self) : # no longer normalise by total number of sampels, as we assume that the L2 norm has been found by EMMA to be precisely correct for the given sample size
        if ( self.regularizer == REGULARIZER_NONE ) :
            return(0.0)
        elif (self.regularizer == REGULARIZER_RIDGE ) :
            return ( self.Lambda * self.Weights_W )
        else : # LASSO
            return ( self.Lambda * np.sign(self.Weights_W ) )   # this will cause the numerical check to fail.. it will still be small but much bigger than with L2 or without, this is probably due to the funky L1 derivative at 0
     
       
    # the cost of the regularizer term: only used for  gradient checking
    def getRegularizerCostTerm(self) : # no longer normalise by total number of sampels, as we assume that the L2 norm has been found by EMMA to be precisely correct for the given sample size
        if (self.regularizer == REGULARIZER_NONE) :
            return(0.0) 
        elif (self.regularizer == REGULARIZER_RIDGE) :
            return ( (self.Lambda * 0.5) * np.sum( self.getWeightsAsVector()**2 ) ) 
        else : # LASSO
            return ( (self.Lambda * 0.5) * np.sum( np.abs( self.getWeightsAsVector() ) ) ) 
      
       
    # returns the weights in in the current layer concated into a 1D array (used oly by getRegularizerCostTerm()  )
    def getWeightsAsVector(self) :
        return( self.Weights_W.ravel() )
       
       
    # concats the weights of this layer (if it has any), into the ones we got so far
    def getDebugInfo(self, dataSoFar, Type = WEIGHT) :
        if(self.Weights_W is not None) :
            if(Type == WEIGHT) :  #if we are NOT looking for weight gradients, then we want the weights
                return(  np.concatenate( (dataSoFar, self.getWeightsAsVector()) ) )
            elif (Type == GRAD) : # if we do want the gradients then we use the ones saved during debug
                return(  np.concatenate( (dataSoFar, self.DEBUG_WeightGrads.ravel() ) ) )
            else :
                return(  np.concatenate( (dataSoFar,  [self.getRegularizerCostTerm()] ) ) )
            
        else :
            return(dataSoFar)
        # if we dont have weight, then we wont have weight gradients either so we will just return the data passed along so far (it wont matter which case it was)
        
     
    # removes an equal number of weights from a vector as this layer had, and then replaces the matching values in its Weight matrix
    def addDebugData(self, allWeights) :
        if(self.Weights_W is not None) :
            numWeightsInLayer = self.Weights_W.size

            self.Weights_W = np.reshape(allWeights[0:numWeightsInLayer], (len(self.Weights_W) , len(self.Weights_W[0]) ) )

            totalLength = len(allWeights)
            allWeights = allWeights[numWeightsInLayer:totalLength] # subset the original vector, in a way that leaves off an equal number of elements

            return(allWeights) # return what is left of weights
            