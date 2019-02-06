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

import numpy

import numpy as np
import random
import copy
import gc
import sys 
import scipy
import time
import scipy.stats
import scipy.ndimage as nd
from scipy.ndimage.filters import median_filter
from ....application.logic.knet.im2col import knetim2col
from ....application.utils.geno_qc import zscore

# STATIC vars
OUT_BINARY = "OUT_BINARY"
OUT_MULTICLASS = "OUT_MULTICLASS"
OUT_REGRESSION = "OUT_REGRESSION"
OUT_MAE = "OUT_MAE"

HIDDEN_ACT_SIG = "HIDDEN_ACT_SIG"
HIDDEN_ACT_RELU = "HIDDEN_ACT_RELU"
HIDDEN_ACT_SPLUS= "HIDDEN_ACT_SPLUS"

REGULARIZER_NONE = "REGULARIZER_NONE"
REGULARIZER_RIDGE = "REGULARIZER_RIDGE"
REGULARIZER_LASSO = "REGULARIZER_LASSO"
REGULARIZER_ORTHO = "REGULARIZER_ORTHO"
REGULARIZER_ORTHO2 = "REGULARIZER_ORTHO2"

REGULARIZER_ORTHO_LASSO = "REGULARIZER_ORTHO_LASSO"
REGULARIZER_ORTHO2_LASSO = "REGULARIZER_ORTHO2_LASSO"

GRAD = "GRAD"
WEIGHT = "WEIGHT"
REG_COST = "REG_COST"

LAYER_SUBTYPE_INPUT = "LAYER_SUBTYPE_INPUT"
LAYER_SUBTYPE_OUTPUT = "LAYER_SUBTYPE_OUTPUT"
LAYER_SUBTYPE_HIDDEN = "LAYER_SUBTYPE_HIDDEN"

EPSILON = 1e-8 # this is for FP32, for FP16, this sohuld be 1e-4
ALMOST_1 = 1.0 - EPSILON


OPTIMIZER_SGD = 0
OPTIMIZER_ADAM = 1
OPTIMIZER_AMSGRAD = 2


NETWORK_DATATYPE = 'float32'

KNET_ORIG = False
ORHTO_TRANSPOSE = False
SELU_LAM = 1.0507
SELU_ALPHA = 1.67326

###############################################################################
# Main system utils
###############################################################################
GPU_mode = False
knetim2col.mode_set(np)

def setDataType(datatype) :
    global NETWORK_DATATYPE
    if datatype == "float16" or datatype == "float32" or datatype == "float64" : print("datatype set to: " + str(datatype))
    else: print("accepted datatypes are: float16, float32 and float64, where as it was attempted to be set to: " +  + str(datatype))
    NETWORK_DATATYPE = datatype
    

# if we want to switch to GPU mode: this will override the 'np' var to make all computations on the GPU
def initGPU(forceFloat16 = False) :
    global GPU_mode
    global NETWORK_DATATYPE
    global EPSILON
    global ALMOST_1
    
    #NETWORK_DATATYPE = 'float64'
    
    if forceFloat16 : # this is disabled by default, as it can cause non-convergence
        NETWORK_DATATYPE = 'float16' # for GPU we try and use float16s to reduce RAM footprint
        EPSILON = 1e-4 # this is for FP16
        ALMOST_1 = 1.0 - EPSILON
        
    
    GPU_mode = True
    from ....application.utils import numpy_gpu_bridge
    global np # this global only affects np in this class though, np defined elsewhere will still refer to actual numpy
    np = numpy_gpu_bridge.get_numpy_or_cupy(True)
    
    knetim2col.mode_set(np,GPU_mode)
    
def disableGPU() :
    global GPU_mode
    global NETWORK_DATATYPE
    global EPSILON
    global ALMOST_1
    NETWORK_DATATYPE = 'float32' # for GPU we try and use float16s to reduce RAM footprint
    EPSILON = 1e-8 # this is for FP32, for FP16, this sohuld be 1e-4
    ALMOST_1 = 1.0 - EPSILON 
    GPU_mode = False
    import numpy as numpy_orig
    global np # this global only affects np in this class though, np defined elsewhere will still refer to actual numpy
    np = numpy_orig 
    knetim2col.mode_set(np,GPU_mode)
    print("GPU disabled3")

def set_seed(seed) :
    print("setting knet seed for numpy/cupy (and python) to: " + str(seed))
    global np 
    np.random.seed(seed)
    random.seed(seed)



# When returning the data outside of Knet, we may want to ensure that it is cast back to the CPU
def castOutputToCPU(data) :
    global GPU_mode
    if GPU_mode : 
        #print("GPU was enabled so we cast back to numpy")
        return(np.asnumpy(data))
    else : 
        #print("we were on numpy to begin with, return data as is")
        return data
    
def casToGPU(data) :
    return(np.asarray(data))


def getSizeInMBs(myObject) :
    if myObject is None : return 0.
    return ( np.round( myObject.nbytes  / 1024/ 1024 )  )

def getSizeInGBs(myObject) :
    if myObject is None : return 0.
    return ( np.round( myObject.nbytes * 10 / 1024/ 1024 / 1024 ) / 10  )

def getNetRAMUsage(myNet) :
    for i in range(len(myNet.layers) ) :
        layer = myNet.layers[i]
        totalUsage = 0.0
        if type(layer).__name__ is 'knnLayer' :  
            totalUsage = totalUsage + ( getSizeInMBs(layer.Output_Z) + getSizeInMBs(layer.Weights_W)+  getSizeInMBs(layer.Input_S) +  getSizeInMBs(layer.Error_D) +  getSizeInMBs(layer.Derivative_Fp) +  getSizeInMBs(layer.Momentum) ) 
    
    print("Knet has RAM footprint of (MBs):", totalUsage ) 
    


# Truncated normal distribution, that forces all values to be within given limits (IE within 2SDs) by checking values more than a limit away from the mean are discarded and re-drawn.
def truncatedGaussian(shape, SD, limit = 2.) :
    # create an initial pool of random values, 1D array of the right size
    global NETWORK_DATATYPE
    dims = numpy.prod(shape)
    #print("Dims are: " + str(dims) + " / SD is: " + str(SD), flush=True)
    Weights_W = np.random.normal(size=int(dims), scale=SD ).astype(NETWORK_DATATYPE)
    
    # find the indices of where such values are (if any)
    largeValue_indices =  np.where(np.abs(Weights_W)> limit*SD ) # this line is buggy in cupy version < v4.0.0b3 , so we have to install latest beta: pip install cupy==v4.0.0b3 --no-cache-dir
    numLargeValues = len(largeValue_indices[0])
   # print("largeValue_indices[0] is: " + str(largeValue_indices[0]) )

    while numLargeValues > 0 : # recursively keep going until we have replaced all large values
        #print("found too many large values: " + str(numLargeValues ) , flush=True)
        #print("first few elements of Weights_W: " + str(Weights_W[largeValue_indices]) )

        # draw new random variables again, of the same length that we have had too large ones
        subStituteWeights = np.random.normal(size=numLargeValues, scale=SD).astype(NETWORK_DATATYPE)
        #print("first few subStituteWeights are: " + str(subStituteWeights) )
        
        Weights_W[largeValue_indices] = subStituteWeights # sub in these new weights
        largeValue_indices =  np.where(np.abs(Weights_W)> limit*SD )
        numLargeValues = len(largeValue_indices[0])
        #print("largeValue_indices[0] is: " + str(largeValue_indices[0]) )
      
    
    # reshape Weights into original dimensions
    Weights_W = Weights_W.reshape(shape)
    
    return(Weights_W)
    
# diagnostics function that can be used to quickly check if conv/pool etc will produce valid (integer) dimensions (without having to init network and load mem etc)
# https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/

# to keep orig dimensions, use padding of size, and we need stride 1, and filter size of:
def getPaddingForMaintainVol(filter_height = 3) :
    padding_keep = (filter_height-1)/2
    return(padding_keep)
    
def checkConvOutput (myNet, input_shape) :
    print("Orig Input shape: " + str(input_shape))
    
    for layerIndex in range(1,len(myNet.layers)) :
        convLayer = False    
        if type(myNet.layers[layerIndex]) == knnConvLayer:
            filter_height  = myNet.layers[layerIndex].filter_height
            filter_width  = myNet.layers[layerIndex].filter_width
            
            p_h = myNet.layers[layerIndex].p_h
            p_w = myNet.layers[layerIndex].p_w
            stride_h = myNet.layers[layerIndex].stride_h
            stride_w = myNet.layers[layerIndex].stride_w
            #print(str(  (input_shape[0], filter_height,2* p_h , stride_h)  ))
            #print(str( (input_shape[1], filter_width, 2 * p_w , stride_w)  ))
            input_shape[0] = (input_shape[0] - filter_height +2* p_h) / stride_h + 1
            input_shape[1] = (input_shape[1] - filter_width +2* p_w) / stride_w + 1
            print("Conv (layer"+str(layerIndex)+") reshaped data to: " + str(input_shape))
            convLayer = True 
            
        elif type(myNet.layers[layerIndex]) == knnMaxPool  :
            size_h = myNet.layers[layerIndex].size_h
            size_w = myNet.layers[layerIndex].size_w
            stride_h = myNet.layers[layerIndex].stride_h
            stride_w = myNet.layers[layerIndex].stride_w
            input_shape[0] = (input_shape[0] - size_h) / stride_h + 1
            input_shape[1] = (input_shape[1] - size_w) / stride_w + 1
            print("Maxpool (layer"+str(layerIndex)+") reshaped data to: " + str(input_shape))     
            
        elif type(myNet.layers[layerIndex]) == knUpsample :
            input_shape[0] = input_shape[0] *2
            input_shape[1] = input_shape[1]*2
                       
        elif type(myNet.layers[layerIndex]) == knnFlatten  :
            print("finished conv layers")
            break

        
        remainder_h =  input_shape[0] % 1
        remainder_w =  input_shape[1] % 1
        if(remainder_h != 0 or remainder_w != 0) :

            if convLayer : 
                print("dimensions are not integers! (but for conv layers it isn't fatal)")  
            else : 
                print("dimensions are not integers!")
                break
        
        # after each layer dimensions are always integers
        input_shape[0] -= remainder_h
        input_shape[1] -= remainder_w
        
def getConvExpansionFactor(layer) :
    filter_height = layer.filter_height
    filter_width = layer.filter_width
    stride_h = layer.stride_h 
    stride_w = layer.stride_w
    expansionFactor = (filter_height+1-stride_h) * (filter_width+1-stride_w)
    #print("Conv layer input RAM expansionFactor is: " + str(expansionFactor))
    return(expansionFactor)
    
    
# get how much memory NN will use BEFORE actually running it, (IE this is near the high watermark during FP (Conv layers may slightly exceed it), on BP as we keep disposing the Inputs/Errors as we go back it shouldn't go much higher)
            # so only compute RAM for stuff that is used for forward Prop
            
            
          
       #print(self.layers[i])
       #print(isinstance(self.layers[i],knnLayer))
       #if ( isinstance(self.layers[i],knnLayer) ) :
       #    print(self.layers[i].Weights_W is not None)
       
  
def getTotalNumNeurons(myNet) :
    totalNumNeurons = 0

    for layerIndex in range(0,len(myNet.layers)) :
        layer = myNet.layers[layerIndex]
        if type(layer) == knnLayer: # for non input types, we have
            if (layer.subtype != LAYER_SUBTYPE_INPUT) : totalNumNeurons += layer.layer_numUnits           
        
        elif type(layer) == knnConvLayer: totalNumNeurons += layer.num_filters

    return(totalNumNeurons)
    
       
def getNetworkMemUsage(myNet, input_shape) :
    if myNet.suppressPrint == False : print("getNetworkMemUsage with number with input shape: " + str(input_shape[0]) + ":")
    bytesPerDataype = 8
    global NETWORK_DATATYPE
    if NETWORK_DATATYPE == 'float16' : bytesPerDataype = 2 # if it is 16 bit, it is just 4 bytes
    elif NETWORK_DATATYPE == 'float32' : bytesPerDataype = 4
    if myNet.suppressPrint == False : print("bytesPerDataype ("+str(NETWORK_DATATYPE)+") is: " + str(bytesPerDataype))
    #totalBytes_managed_running = 0.0 # how much memory network will use if trained via 'managed' learning (cleaning up outputs/Deltas as we go)
    totalBytes_running = 0.0 # how much mem network will use if trained naively, keeping al Errors/Outputs
    
    lastLayer_outShape = input_shape
    totalNumParams = 0 
    
    numHiddenLayers = 0
    firstLayerNeurons = -1
    totalNumNeurons = 0
    
    for layerIndex in range(0,len(myNet.layers)) :
        layer = myNet.layers[layerIndex]
        #totalBytes_managed = 0.0 # how much memory network will use if trained via 'managed' learning (cleaning up outputs/Deltas as we go)
        totalBytes = 0.0 # how much mem network will use if trained naively, keeping al Errors/Outputs
          
        extraText = ""
        if layerIndex == 0 : extraText =" Input"
        
        if type(layer) == knnLayer: # for non input types, we have
            currentParams =0
            if (layer.subtype == LAYER_SUBTYPE_INPUT) : # if its the input then the byes will be directly determined by the minibatch shape
                #datatByes = numpy.prod(lastLayer_outShape) * bytesPerDataype
                #totalBytes_managed += datatByes
                #totalBytes += datatByes 
                # input does NOT add anything, as it does not save anything
                #print("input adds MBs: " + str(round(datatByes/1024/1024)) )
                if myNet.suppressPrint == False : print("Input does not add anything")
                
            else :
                 # Weights_W
                currentParams = lastLayer_outShape[1] * layer.layer_numUnits
                totalNumParams += currentParams
                datatByes = currentParams * bytesPerDataype
                totalBytes += datatByes
                numHiddenLayers += 1
                totalNumNeurons += layer.layer_numUnits
                if firstLayerNeurons == -1 : firstLayerNeurons = layer.layer_numUnits
                
                #print("Weights_W added RAM (MBs): "  + str(round(datatByes/1024/1024)))  
                
                # Momentum
                totalBytes += datatByes
                #print("Momentum added RAM (MBs): "  + str(round(datatByes/1024/1024)))  
                
                # Past_Grads
                totalBytes += datatByes
                #print("Past_Grads added RAM (MBs): "  + str(round(datatByes/1024/1024)))  
                
                 # Input
                datatByes = numpy.prod(lastLayer_outShape) * bytesPerDataype  # lastLayer_outShape[0] * layer.layer_numUnits * bytesPerDataype
                totalBytes += datatByes 
                #print("Input added RAM (MBs): "  + str(round(datatByes/1024/1024)))  
                 
                # Error_D: this is disposed on the way back along with the Input, so we will ne
                
                lastLayer_outShape = [lastLayer_outShape[0],layer.layer_numUnits ]
                #print("lastLayer_outShape: "  + str(lastLayer_outShape)) 
                
            totalBytes_running += totalBytes
            if myNet.suppressPrint == False : print("FC (layer"+str(layerIndex)+ extraText + ") added RAM (MBs): " + str(round(totalBytes/1024/1024)) + " (trainable params ~"+str(currentParams) + ")") 
  
        
        elif type(layer) == knnConvLayer:
            # Weights_W  
            field_height = layer.filter_height
            field_width = layer.filter_width
            currentParams = layer.num_filters * lastLayer_outShape[1] * field_height * field_width
            totalNumParams += currentParams
            datatByes = currentParams * bytesPerDataype
            totalBytes += datatByes
            #print("Weights_W added RAM (MBs): "  + str(round(datatByes/1024/1024)) + " layer.num_filters: " + str(layer.num_filters))  
            numHiddenLayers += 1
            totalNumNeurons += layer.num_filters
            if firstLayerNeurons == -1 : firstLayerNeurons = layer.num_filters
  
            # Momentum
            totalBytes += datatByes
            #print("Momentum added RAM (MBs): "  + str(round(datatByes/1024/1024)))  
            
            
            # Past_Grads
            totalBytes += datatByes
            #print("Past_Grads added RAM (MBs): "  + str(round(datatByes/1024/1024))) 
        
            # compute Input size as cols
            N = lastLayer_outShape[0]
            C = lastLayer_outShape[1]
            inputHeight = lastLayer_outShape[2]
            inputWidth = lastLayer_outShape[3]

            p_h = layer.p_h
            p_w = layer.p_w
            stride_h = layer.stride_h
            stride_w = layer.stride_w
            
            numlocations_Height = ((inputHeight-field_height)/stride_h)+1 + 2 * p_h
            numlocations_Width = ((inputWidth-field_width)/stride_w)+1 + 2 * p_w
            #print("N: "  + str(N) + " / C: " + str(C) + " / inputWidth: " + str(inputWidth) + " / inputHeight: " + str(inputHeight) + " / p_h: " + str(p_h) + " / stride_h: " + str(stride_h) + " / stride_w: " + str(stride_w) + " / numlocations_Height: " + str(numlocations_Height) + " / numlocations_Width: " + str(numlocations_Width))

            X_col_shape = [field_height  * field_width * C, numlocations_Width * numlocations_Height * N]

            datatByes = numpy.prod(X_col_shape) * bytesPerDataype 
            totalBytes += datatByes              #datatByes = layer.num_filters * X_col_shape[1]* bytesPerDataype    #datatByes = lastLayer_outShape[0] * layer.num_filters * bytesPerDataype
            #print("Input added RAM (MBs): "  + str(round(datatByes/1024/1024))) 
            #print("X_col_shape: "  + str(X_col_shape)) 
            
            # Error_D: this is disposed on the way back for the managed versions
            #datatByes =  layer.num_filters * X_col_shape[1] * bytesPerDataype # this is d x Q
            #totalBytes += datatByes
            #print("Error_D added RAM (MBs): "  + str(round(datatByes/1024/1024))) 

            # compute output dimensions
            output_height = (inputHeight - field_height + 2 * p_h) / stride_h + 1 # 28 # this could only be cached for the 1st layer, as subsequent deeper layers may not be 28, depending on filter size / padding
            output_width = (inputWidth -field_width + 2 * p_w) / stride_w + 1 # 28
            lastLayer_outShape = [lastLayer_outShape[0],layer.num_filters, output_height,output_width ]
            #print("lastLayer_outShape: "  + str(lastLayer_outShape)) 

            totalBytes_running += totalBytes
            expansionFactor = getConvExpansionFactor(layer)
            if myNet.suppressPrint == False : print("Conv (layer"+str(layerIndex)+") added RAM (MBs): " + str(round(totalBytes/1024/1024)) + " (expansion factor: "+ str(expansionFactor) +")"  + " (trainable params ~"+str(currentParams) + ")")   
  
   
        elif type(layer) == knnMaxPool  :
            
            # Get Input size
            N = lastLayer_outShape[0]
            C = lastLayer_outShape[1]
            size_h = layer.size_h
            size_w = layer.size_w
            stride_h = layer.stride_h
            stride_w = layer.stride_w
            p_h = layer.p_h
            p_w = layer.p_w

            inputHeight = lastLayer_outShape[2]
            inputWidth = lastLayer_outShape[3]
            
            numlocations_Height = ((inputHeight-size_h)/stride_h)+1 + 2 * p_h
            numlocations_Width = ((inputWidth-size_w)/stride_w)+1 + 2 * p_w

            X_col_shape = [size_h  * size_w * C, numlocations_Width * numlocations_Height * N]
            
            datatByes = numpy.prod(X_col_shape) * bytesPerDataype 
            totalBytes += datatByes
            totalBytes_running += totalBytes
            
            # output shape
            output_height = (lastLayer_outShape[2] - size_h) / stride_h + 1 #, this is 14, for an image of 28 with stride 2 and padding 0 ( IE we just downsample it) get output dimensions
            output_width = (lastLayer_outShape[3] - size_w) / stride_w + 1

            lastLayer_outShape = [lastLayer_outShape[0],lastLayer_outShape[1], output_height,output_width ]            
            

            if myNet.suppressPrint == False : print("Maxpool (layer"+str(layerIndex)+") added RAM (MBs): " + str(round(totalBytes/1024/1024)))   

            
        elif type(layer) == knUpsample :
            # I am not sure about this as none of the below is stored as class vars
            output_height = lastLayer_outShape[2] * 2
            output_width  = lastLayer_outShape[3] * 2
            
             # Output
            #datatByes = lastLayer_outShape[0] * lastLayer_outShape[1] * output_height * output_width * bytesPerDataype
            #totalBytes += datatByes   
            
            lastLayer_outShape = [lastLayer_outShape[0],lastLayer_outShape[1], output_height,output_width ]

            #totalBytes_running += totalBytes
            
            if myNet.suppressPrint == False : print("Upsample (layer"+str(layerIndex)+") added RAM (MBs): " + str(round(totalBytes/1024/1024)))   

                       
        elif type(layer) == knnFlatten  : 
            # I think this does not add anything  
            lastLayer_outShape = [lastLayer_outShape[0],lastLayer_outShape[1]* lastLayer_outShape[2] * lastLayer_outShape[3] ]
            
            if myNet.suppressPrint == False : print("Flatten (layer"+str(layerIndex)+") added RAM (MBs): " + str(round(totalBytes/1024/1024)))   


        elif type(layer) == knnBatchNorm  or type(layer) == knnSpatialBatchNorm : 
             # It saves Input, 
            datatByes = numpy.prod(lastLayer_outShape) * bytesPerDataype
            totalBytes += datatByes  
            
            # and Input_nrom too
            totalBytes += datatByes   
            
            #lastLayer_outShape = [lastLayer_outShape[0],lastLayer_outShape[1]* lastLayer_outShape[2] * lastLayer_outShape[3] ]
            totalBytes_running += totalBytes
            if myNet.suppressPrint == False : print("(Spatial) Batch norm (layer"+str(layerIndex)+") added RAM (MBs): " + str(round(totalBytes/1024/1024)))   

                       
        elif type(layer) == knRELU or type(layer) == knLeakyRELU or type(layer) == knSigmoid : 
        
             # It saves Input, 
            datatByes = numpy.prod(lastLayer_outShape) * bytesPerDataype
            totalBytes += datatByes  
            
            # Error, this is not actually saved as as class variable
            # totalBytes += datatByes  

            totalBytes_running += totalBytes
            #print("lastLayer_outShape: " + str(lastLayer_outShape))
            if myNet.suppressPrint == False : print("Activation (layer"+str(layerIndex)+") added RAM (MBs): " + " / " + str(round(totalBytes/1024/1024)))   

    numHiddenLayers -= 1 # we don't count the output layer
    if myNet.suppressPrint == False : print("In total network uses RAM (MBs): "  + " / " + str(round(totalBytes_running/1024/1024)))
    if myNet.suppressPrint == False : print("Neural Net total number of trainable params ~" + str(totalNumParams) + ")") 
    if myNet.suppressPrint == False : print("Number of hidden layers: " + str(numHiddenLayers) )  
    
    return ( {'numHiddenLayers' : numHiddenLayers, 'totalNumNeurons' : totalNumNeurons, 'totalBytes_running' : totalBytes_running, 'firstLayerNeurons' : firstLayerNeurons })


    def getSizeInBytes(myObject) :
        if myObject is None : return 0.
        return (  myObject.nbytes  )
    
    
    # same as above, but this needs the network to be already inited (IE wont work if we get out of Mem error already )
    def getActualRAMUsage(myNet) :
       
        # find the actual usage stats
        RAMTotal = 0.
        for i in range( 0, len(myNet.layers) ) :
            layer = myNet.layers[i]
            
            # Layer Conv1    
            MBsTotal= 0.
            if hasattr(layer,'Weights_W') : 
                MBsTotal +=getSizeInBytes(layer.Weights_W)
                #if( layer.Weights_W is not None) :
                #    print("layer.Weights_W shape is: " + str(layer.Weights_W.shape))
                #    print("manual RAM computation is then: " + str(round( numpy.prod(layer.Weights_W.shape) * 4   /1024/1024)))
            if hasattr(layer,'Error_D') : MBsTotal +=getSizeInBytes(layer.Error_D)
            if hasattr(layer,'Input_S') : MBsTotal +=getSizeInBytes(layer.Input_S)
            if hasattr(layer,'Momentum') : MBsTotal +=getSizeInBytes(layer.Momentum)
            if hasattr(layer,'Past_Grads') : MBsTotal +=getSizeInBytes(layer.Past_Grads)
            if hasattr(layer,'W_grad') : MBsTotal +=getSizeInBytes(layer.W_grad)
            
            if hasattr(layer,'Input_S') : MBsTotal +=getSizeInBytes(layer.Input_S)
            
            if hasattr(layer,'Input_norm') : MBsTotal +=getSizeInBytes(layer.Input_norm)
            
            # this virtually add nothing
            if hasattr(layer,'Weights_bias') : MBsTotal +=getSizeInBytes(layer.Weights_bias)
            if hasattr(layer,'Bias_Momentum') : MBsTotal +=getSizeInBytes(layer.Bias_Momentum)
            if hasattr(layer,'W_bias_grad') : MBsTotal +=getSizeInBytes(layer.W_bias_grad)
            
            
            MBsTotal # 300.0
            
            RAMTotal+=MBsTotal
            
            print( type(layer).__name__ + "("+str(i)+") added " + str( round( MBsTotal  / 1024/ 1024  ) )  + "MBs of RAM")
        
        print ("total RAM used: " + str(round( RAMTotal  / 1024/ 1024  ) ) , flush=True)


  

    
###############################################################################
# Activation functions
###############################################################################   

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
        return np.ones(X.shape)


def k_leakyRELU(X, deriv=False, alpha=0.001):  # 0.001, TF uses 0.3
    if not deriv:
        return np.maximum(X, alpha * X)
    else:
        Error_D = np.ones(X.shape)  
        Error_D[X <= 0] = alpha
        return Error_D
    
def k_RELU(X, deriv=False): 
    if not deriv:
        return np.maximum(X,0)
    else:
        Error_D = np.ones(X.shape)  
        Error_D[X <= 0] = 0
        return Error_D


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

## calculates multiclass classification accuracy  (use for softmax output)
#def calc_Accuracy(yhat, y): 
#    num_matches = 0.0
#    for i in range( len(yhat) ):  # go row by row
#        if  np.argmax(yhat[i,:]) ==  np.argmax(y[i,:]) : 
#            num_matches = num_matches+1
#         
#    #perc = num_matches/len(yhat)
#    return(num_matches)
    
#def calc_Accuracy(yhat, y):
#    return np.mean(yhat == y) 
#




def calc_Accuracy(yhat, y):
    y_preds= np.argmax(yhat, axis=1)
    y_trues= np.argmax(y, axis=1)

    num_matches = np.sum(y_preds == y_trues)
    return(num_matches)
    

def l2norm(x):
    return ( np.sqrt(np.sum(x**2)) )
#  return ( total / len(yhat) )




###############################################################################
###############################################################################

class knn:
    def __init__(self, suppressPrint = False, optimizer = 0):
        self.suppressPrint = suppressPrint
        self.optimizer = optimizer
        self.layers = []
        self.num_layers = 0
        self.DEBUG = False
        self.CAN_ADD_LAYERS = True
        self.supportLayers = [] # layers such as the output or (spatial) Batch norm, that would interfere with inference
        self.supportLayer_indices = []
        self.backedupLayers = None 
        #self.dataType = 'float32' # controls the default datatype used by the network (should be 32 for CPU, and 32 or 16 for GPU)
        
        
        if self.suppressPrint == False : print("KNET NEW CREATED, with optimizer " + str(self.optimizer))

    def addLayer(self,layer) :  # adds a layer to the neural networks list of layers
        if self.CAN_ADD_LAYERS : 
            self.num_layers = self.num_layers+1
            self.layers.append(layer)
            if self.suppressPrint == False : print( "layer " + type(layer).__name__ +" is added to Network " )
            
    def removeLayer(self,layer) :  # adds a layer to the neural networks list of layers
        # 1) Clean up from behind: IE remove references to this layer from behind 
        #if layer.prevLayer is not None : # if a layer was connected to this layer from behind
        #    if layer.nextLayer is not None : # if this layer had a next layer, then we 'stitch' together the network by connecting the prev layer to the one following the layer being removed
        #        layer.prevLayer.nextLayer = layer.nextLayer
        #        print("stitching layers together, from layer before to the one after")
         #   else : 
         #       layer.prevLayer.nextLayer = None # otherwise set it to None
          
        # 2) Clean up from the front: IE remove references to this layer from ahead
        #if layer.nextLayer is not None : # if this was not the last layer, IE there are layers ahead
        #    if layer.prevLayer is not None : # if this layer had a layer before, then we 'stitch' together the network by connecting the next layer to the one before the layer being removed
        #        layer.nextLayer.prevLayer = layer.prevLayer
        #        print("stitching layers together, from layer after this to the one behind")
        #    else : 
        #        layer.nextLayer.prevLayer = None  
            
        self.num_layers = self.num_layers-1
        self.layers.remove(layer)
        
        
        self.connectLayers(False) 
        
        #print( "layer " + type(layer).__name__ +" was removed from Network " )
        
        
    def addLayerBackAt(self,layer, layerPos) :  # adds a previously removed layer, back to its original position

        self.num_layers = self.num_layers+1
        self.layers.insert(layerPos, layer)
        
       # # Need to connect up the layers before/after this to make sure FP/BP works
       # if layer.prevLayer is not None : # if layer to be added had a layer BEFORE
       #     layer.prevLayer.nextLayer = layer # connect it back to this layer
       #     print("connected layer BEFORE to current")
        
       # if layer.nextLayer is not None : # if layer to be added had a layer AFTER
       #     layer.nextLayer.prevLayer = layer # connect it back to this layer
       #     print("connected layer AFTER to current")
            
        self.connectLayers(False)
            
        #print( "layer " + type(layer).__name__ + " is added back to Network at position" + str(layerPos) )
        
    def getLayersOutput(self,layer) :
        # get the next activation type
        # if there is another non activation or non support type layer, then this layer was NOT activated
        activatedLayer = layer
        foundOutput = False
        while activatedLayer.nextLayer is not None and foundOutput == False :
            if activatedLayer.nextLayer.supportLayerType == False and activatedLayer.nextLayer.activationType == False : foundOutput = True # if its not an activation nor a support type, then it must be a 'regular' layer, IE we reached the next regular layer, so we have to use the previous one
            else :
                activatedLayer = activatedLayer.nextLayer
                if activatedLayer.activationType : foundOutput = True # if its an activation type layer, then we have found what we are looking for
                # if its a support type layer, 
                
        return(activatedLayer)

    def getNumTrainableParams(self) :
                
       totalNumParams = 0            
       #print(self.layers[i])
       #print(isinstance(self.layers[i],knnLayer))
       #if ( isinstance(self.layers[i],knnLayer) ) :
       #    print(self.layers[i].Weights_W is not None)
       
       for i in range(0, self.num_layers):
           if ( isinstance(self.layers[i],knnLayer) and self.layers[i].Weights_W is not None) :
               totalNumParams += np.prod(self.layers[i].Weights_W.shape)
               print("layer", str(i), " has number of params: ", str(np.prod(self.layers[i].Weights_W.shape)) )
           
       print("Neural Net num trainable params: ", str(totalNumParams)) 
            
        
    def connectLayers(self, initWeights = True) :   # connects up each layer to its next, and initialises the Weighted layers with the correct dimension matrices  
        # the first layer only has a next layer
 
        self.layers[0].initConnections( {"PREV":None, "NEXT":self.layers[1] }, initWeights )
          
        if self.num_layers > 2 : # if there  are any hidden layers
            
            for i in range( 1, (self.num_layers-1) ):  # go through all layers except last
               self.layers[i].initConnections(  {"PREV":self.layers[i-1], "NEXT":self.layers[i+1] }, initWeights )

            

        # connect Output
        self.layers[-1].initConnections( {"PREV":self.layers[-2], "NEXT":None}, initWeights  ) # the last layer only has a prev layer
           
        #print("All layers created, Neural Network ready")

    # Main forward propagation algo: it usually return Sigma(XB), and feeds it int the next layer's input
    def forward_propagate(self, data, train = True, saveInput = True, forceCast_toCPU = False, finalLayer = None): # data is the design matrix
        global NETWORK_DATATYPE
        data = np.asarray( data, dtype=NETWORK_DATATYPE) ;
        yhat = self.layers[0].forward_propagate(data,train,saveInput, finalLayer) # this recursively calls all subsequent layers' forward propagate until the last (the output, which produces the yhat)
        
        global GPU_mode
        if GPU_mode and forceCast_toCPU : yhat = castOutputToCPU(yhat) # if it was requested to cast data back onto CPU/numpy then we perform that here ( this could be used if )
        return( yhat )
    
    
    # calculates the cumulative errors (Delta) at each node, for all layers
    # Inference controls if we backpropagate back to the 1st layer (IE get gradients wrt the input space), which is not needed for training and spared calculation to speed things up
    # but is needed for both saliency maps and deep dreaming operations
    # guidedBacjprop: this is needed for saliency maps
    def backpropagate(self, yhat, labels, inference = False, guidedBackprop = False, targetLayer = -1 , batchNum = -1): # Inference is a flag that switches on Guided BP and makes sure that we BP to the first layer to get D1 (otherwise that is not needed for Delta Ws )
        global NETWORK_DATATYPE
        labels = labels.reshape(labels.shape[0],-1)
        # calculate first Error Delta directly from output
        Error_Delta_Output = np.asarray( (labels - yhat), dtype=NETWORK_DATATYPE ) # (yhat - labels)#.T  # -> this would cause all gradients to be inverted
        
        #print ("Error_Delta_Output dims: " + str(Error_Delta_Output.shape)  + " // yhat dims: " + str(yhat.shape) + " // labels dims: " + str(labels.shape) )
        self.layers[targetLayer].backpropagate(Error_Delta_Output, inference, guidedBackprop ) # call this on the last layer, which then will recursively call all the rest all the way back to the first
        

    # goes through and updates each layer (for ones that have weights this updates the weights, for others this may do something else or nothing)
    def update(self, eta, friction, gamma, epoch_num ):
        for i in range(1, self.num_layers):       # loop from the 2nd layer to the last ( the 1st layer is the input data, and that cannot be updated)
            self.layers[i].update(eta, friction, gamma, epoch_num)





    # For inference type operation we need to remove BN / softmax type of layers as that would negatively impact 
    def removeSupportLayers(self, removeBNlayers = True) :
        for i in range(len(self.layers)) :
            if isinstance(self.layers[i],knSoftmax) or isinstance(self.layers[i],knnSpatialBatchNorm) and removeBNlayers or isinstance(self.layers[i],knnBatchNorm) and removeBNlayers :

                self.supportLayers.append(self.layers[i])
                self.supportLayer_indices.append(i)

        for i in range(len(self.supportLayers)) :
            #print( "removing layer " + type(layersRemoved[i]).__name__ +" from saliency computation " )
            self.removeLayer(self.supportLayers[i])
            
    def addBackSupportLayers(self):
        for i in range(len(self.supportLayers)) :
             self.addLayerBackAt(self.supportLayers[i], self.supportLayer_indices[i])
             
        self.supportLayers = []
        self.supportLayer_indices = []


    def dream_2(self, target_class, targetClass_strength = 100, startImage = None, iterations = 20,lr = 1.5 , normalize = False, blur = 0.2, L2mode = False, jitter=-1, blurFreq = 10, l2decay = .01,mFilterSize = 3, mFilterEvery = 8, small_val_percentile = 0, small_norm_percentile = 20, savePartial = True, removeBNlayers = True, targetLayerNum = 0):
        global EPSILON
        
        if self.layers[0].last_input_shape is None : 
            print("network must have been forward propagated once before dreaming")
            return(None)
        
        self.removeSupportLayers(removeBNlayers)

        convNet = True
        if len(self.layers[0].last_input_shape) < 3 : convNet = False # if it is NOT a convnet, IE input has shape [n,p]
            
        if convNet: INPUT_DIM = [self.layers[0].last_input_shape[1],self.layers[0].last_input_shape[2], self.layers[0].last_input_shape[3]]
        else : INPUT_DIM = [self.layers[0].last_input_shape[1]]

        if startImage is None : X_fooling = np.zeros( (1,*INPUT_DIM) )  # start from blackness
        else: X_fooling = startImage.copy() # start from a supplied image

        # this shape should be num_units x 1 ( as it is the Error_D^t of that layer)
        if targetLayerNum == 0 : targetLayerImportance = X_fooling # if we want the Input layer, that does not have a Weight matrix, so we can just reuse the X_fooling
        else : targetLayerImportance = np.zeros( (1, self.layers[targetLayerNum].Weights_W.shape[1]) ) 
        print("targetLayerImportance shape is: " + str(targetLayerImportance.shape))
        
        Yhat = self.forward_propagate( X_fooling, False )
        labels = np.zeros( Yhat.shape )
        labels[0, target_class] = targetClass_strength

        partialResults = []
        for i in range(iterations) :
            
               
            if self.suppressPrint == False : print("dreaming iteration: "  + str(i))
            Yhat = self.forward_propagate( X_fooling, False )
 
            self.backpropagate(Yhat,  labels, inference = True, guidedBackprop = False ) 

            
            dX =  self.layers[0].Error_D  # get the error at the inpit
            dX= dX.reshape( dX.shape[0],*INPUT_DIM)

            # basic update, we keep creating the synthetic input which results in the highest activation
            X_fooling += lr * dX
            print("self.layers[targetLayerNum].Error_D shape is: " + str(self.layers[targetLayerNum].Error_D.shape))
            targetLayerImportance += lr * self.layers[targetLayerNum].Error_D # but we actually save the Errors of the layer that we are interested in. If this is the Input layer then this is identical to the above,   IE at input layer we get an importance score for each SNP, at deeper layer we get an importance score for each neuron
            # if the above is the same as X_fooling, then we are adding the derivative 2x... which is BAD
            
            if savePartial : partialResults.append( castOutputToCPU( targetLayerImportance.copy() ) )
            
            
        self.addBackSupportLayers()
        return( castOutputToCPU(targetLayerImportance) , partialResults)

    # evaluate Knet via associating each SNP via the Garson weights
    def dream_Garson(self, garsonActivate = False) :
    
        NNinference = self.getWeightImportanceScores(self.layers[0], activate = garsonActivate).ravel()
        return(NNinference)

    # produces 'expected' importance scores from the Weight matrices of a network: this is ~ to the Importance scores from the 'deep dreaming' (which are the observed) but seem to be less accurate
    def getWeightImportanceScores(self, startLayer, activate = False) :
        
        importanceScores = None
        startLayerFound = False
        
        for layerIndex in range(0,len(self.layers)) :
            layer = self.layers[layerIndex]
            if layer == startLayer : startLayerFound = True
    
            if startLayerFound is False : continue
        
            if type(layer) == knnConvLayer or type(layer) == knnLayer:
                if hasattr(layer,'Weights_W') and layer.Weights_W is not None : 
                    weights = layer.Weights_W
                    # if we meant to activate weights, and there is a next layer which is an activation type of layer
                    if activate and layer.nextLayer is not None and type(layer.nextLayer) != knnConvLayer  and type(layer.nextLayer) != knnLayer and type(layer.nextLayer) != knnMaxPool:
                        weights = layer.nextLayer.generateOutput(weights, train = False, saveInput = False) # apply activation, without saving it (IE dont modify the network)
                    
                    weights = np.abs(weights) # only take abs AFTER the nonlinearity
                    if importanceScores is None :   importanceScores = weights
                    else : importanceScores = importanceScores @ weights
             
    
        return(importanceScores.ravel())

    # dreams 
    def dream_targetLayer(self, finalLayer, target_class, targetClass_strength = 100, startImage = None, iterations = 20,lr = 1.5 , normalize = False, blur = 0.2, L2mode = False, jitter=-1, blurFreq = 10, l2decay = .01,mFilterSize = 3, mFilterEvery = 8, small_val_percentile = 0, small_norm_percentile = 20, savePartial = True, removeBNlayers = True):
        global EPSILON

        if self.layers[0].last_input_shape is None : 
            print("network must have been forward propagated once before dreaming")
            return(None)

        self.removeSupportLayers(removeBNlayers)

        convNet = True
        if len(self.layers[0].last_input_shape) < 3 : convNet = False # if it is NOT a convnet, IE input has shape [n,p]
            
        if convNet: INPUT_DIM = [self.layers[0].last_input_shape[1],self.layers[0].last_input_shape[2], self.layers[0].last_input_shape[3]]
        else : INPUT_DIM = [self.layers[0].last_input_shape[1]]

        if startImage is None : X_fooling = np.zeros( (1,*INPUT_DIM) )  # start from blackness
        else: X_fooling = startImage.copy() # start from a supplied image

        Yhat = self.forward_propagate( X_fooling, False, finalLayer = finalLayer.nextLayer ) # only forward propagate until the final layer not to output
        allButTargetIndices = [x for x in range(Yhat.shape[1]) if x != target_class] # build a selector that selects all indices except the target index, this will then be reused as a mask later

        labels = np.zeros( Yhat.shape ) # copy this with zeros
        labels[:, target_class] = targetClass_strength # this sets the target neuron's output to be the given strength

        partialResults = []
        for i in range(iterations) :
     
            if self.suppressPrint == False : print("dreaming iteration: "  + str(i))
            Yhat = self.forward_propagate( X_fooling, False, finalLayer = finalLayer.nextLayer ) # compute the output up until the final layer, we reference the 'nextLayer' as that will stop, and return the output of the actual finalLayer
            Yhat[:,allButTargetIndices] = 0 # zero out all but the target neuron's output (this has the same effect as if all but the other neurons were zeroed out in the weight matrix)
           
    
            # labels = labels.T # need to transpose it for BP
            self.backpropagate(Yhat,  labels, inference = True, guidedBackprop = False, targetLayer = self.layers.index(finalLayer) ) # backpropagate from the final layer

            
            dX =  self.layers[0].Error_D  # get the error at the Input
            dX= dX.reshape( dX.shape[0],*INPUT_DIM)

            # basic update
            X_fooling += lr * dX
            

            if savePartial : partialResults.append( castOutputToCPU( X_fooling.copy() ) )
            
            
        self.addBackSupportLayers()
        return( castOutputToCPU(X_fooling) , partialResults)
        
        

    # Deep Dreaming via gradient ascent: https://github.com/shksa/My-cs231n-assignment/blob/master/ImageGradients.ipynb
    def dream(self, target_class, targetClass_strength = 100, startImage = None, iterations = 20,lr = 1.5 , normalize = False, blur = 0.2, L2mode = False, jitter=-1, blurFreq = 10, l2decay = .01,mFilterSize = 3, mFilterEvery = 8, small_val_percentile = 0, small_norm_percentile = 20, savePartial = True, removeBNlayers = True):
        global EPSILON
        
        if self.layers[0].last_input_shape is None : 
            print("network must have been forward propagated once before dreaming")
            return(None)
        
        self.removeSupportLayers(removeBNlayers)

        convNet = True
        if len(self.layers[0].last_input_shape) < 3 : convNet = False # if it is NOT a convnet, IE input has shape [n,p]
            
        if convNet: INPUT_DIM = [self.layers[0].last_input_shape[1],self.layers[0].last_input_shape[2], self.layers[0].last_input_shape[3]]
        else : INPUT_DIM = [self.layers[0].last_input_shape[1]]

        if startImage is None : X_fooling = np.zeros( (1,*INPUT_DIM) )  # start from blackness
        else: X_fooling = startImage.copy() # start from a supplied image

        Yhat = self.forward_propagate( X_fooling, False )
        labels = np.zeros( Yhat.shape )
        labels[0, target_class] = targetClass_strength

        partialResults = []
        for i in range(iterations) :
            
            #if jitter > 0 : 
            #    ox, oy = np.random.randint(-jitter, jitter+1, 2)
            #    X_fooling = np.roll(np.roll(X_fooling, ox, -1), oy, -2) # apply jitter shift
            
                        
                    
            if self.suppressPrint == False : print("dreaming iteration: "  + str(i))
            Yhat = self.forward_propagate( X_fooling, False )
 
            self.backpropagate(Yhat,  labels, inference = True, guidedBackprop = False ) 

            
            dX =  self.layers[0].Error_D  # get the error at the target layer ( IE this will be 1 x units, IE at input layer we get an importance score for each SNP, at deeper layer we get an importance score for each neuron # np.abs( myNet.layers[0].Error_D  )
            dX= dX.reshape( dX.shape[0],*INPUT_DIM)
            #if L2mode : X_fooling = X_fooling + (lr * dX)**2
           # else : X_fooling = X_fooling + lr * dX
        # normalizing by L2 norm from here: https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py
        
          

            # utility function to normalize a tensor by its L2 norm
            #dX = dX / (np.sqrt(np.mean(np.square(dX))) + EPSILON)       
                    
            # apply normalized ascent step to the input image
            #X_fooling += lr/np.abs(dX).mean() * dX
        
            # basic update
            X_fooling += lr * dX
            
            
            if(i != iterations-1 or convNet == False): # No regularization on last iteration for good quality output
                # apply L2 decay
                if l2decay > 0:
                    X_fooling *= (1 - l2decay)
            
                #if jitter > 0 : X_fooling = np.roll(np.roll(X_fooling, -ox, -1), -oy, -2) # unshift image
                
                
                #src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
                        
            #if clip:
               # bias = np.mean(X_fooling)
               # X_fooling = np.clip(X_fooling, -bias, 255-bias)
    
    
                if blur > 0 and i % blurFreq == 0:
                    X_fooling[0][0] = nd.filters.gaussian_filter(X_fooling[0][0], blur, order=0)
                    X_fooling[0][1] = nd.filters.gaussian_filter(X_fooling[0][1], blur, order=0)
                    X_fooling[0][2] = nd.filters.gaussian_filter(X_fooling[0][2], blur, order=0)
    
    
                if mFilterSize is not 0 and i % mFilterEvery == 0 :
                    X_fooling = median_filter(X_fooling, size=(1, 1, mFilterSize, mFilterSize))
                
                if small_val_percentile > 0:
                    small_entries = (np.abs(X_fooling) < np.percentile(np.abs(X_fooling), small_val_percentile))
                    X_fooling = X_fooling - X_fooling*small_entries   # e.g. set smallest 50% of xx to zero
                   
                if small_norm_percentile > 0:
                    if convNet: pxnorms = np.linalg.linalg.norm(X_fooling, axis=1) # from pylab import *
                    else : pxnorms = np.linalg.linalg.norm(X_fooling, axis=0)
                    smallpx = pxnorms < np.percentile(pxnorms, small_norm_percentile)
                    if convNet: smallpx3 = np.tile(smallpx[:,np.newaxis,:,:], (1,3,1,1))
                    else : smallpx3 =  np.tile(smallpx, (1,1))
                    X_fooling = X_fooling - X_fooling*smallpx3
             
    #               X_fooling = testImage.copy()
    
                        
                if i > 0 and normalize :
                    if convNet:
                        #X_fooling = zscore(X_fooling)[0]
                        X2 = np.zeros(X_fooling.shape)
                        for j in range(INPUT_DIM[0]) :
                            #print(j)
                            
                            # normalise it to prevent nans
                            channel = X_fooling[0][j].copy() 
                            channel = channel.ravel()
                            channel = zscore(channel)[0]
                            X2[0][j] = channel.reshape( (INPUT_DIM[1], INPUT_DIM[2]) )
                            
                        X_fooling = X2
                        
                    else :
                        X_fooling = zscore(X_fooling,axis=1, EPSILON = EPSILON)[0]
                        
           
            if savePartial : partialResults.append( castOutputToCPU( X_fooling.copy() ) )
            
            
        self.addBackSupportLayers()
        return( castOutputToCPU(X_fooling) , partialResults)

        
    def computeSaliency(self, X, layer = -1, targetNeuron = -1 , removeBNlayers = True): # X is a 4D array, N,C,H,W (num in minibatch, C is filters/neurons/ Height/Width)
        # I) if we want final classification
        global NETWORK_DATATYPE
        if ( layer == -1 ) : 
            print ("computeSaliency")
        
        # II) if want inference on a specific neuron/filter
        else : # only work on conv/FC layers
            if ( isinstance(self.layers[layer],knnConvLayer) is False and  isinstance(self.layers[layer],knnLayer) is False  ) : raise ValueError('target layer must be either a FC or Conv layer type')
            print ("computeSaliency for layer" + str(layer) + " for neuron/filter: "  + str(targetNeuron) )  
            
            origWeights = self.layers[layer].Weights_W # save original weights as we ll be overriding them 
            
            self.layers[layer].Weights_W =  np.zeros(self.layers[layer].Weights_W.shape)
     
            # if this is a conv then work on rows, as filters are on rows:  W x Xcol = Outconv: [d,Q]
            if ( isinstance(self.layers[layer],knnConvLayer) ) : self.layers[layer].Weights_W[targetNeuron] = origWeights[targetNeuron]
            else : self.layers[layer].Weights_W[:,targetNeuron] = origWeights[:,targetNeuron] # for FC layers, neurons are on the columns
                
            
            
        # III) Forward prop, non-train, with Inference set to true so we use Guided backprop
        # don't include last layer   if its a softmax
        #resetLastLayer = None
        #if isinstance(self.layers[-1],knSoftmax) :
        #    print("last layer is a softmax, but saliency is computed on the linear predictor: so we remove that from the output")
        #    resetLastLayer = self.layers[-1]
        #    self.removeLayer(self.layers[-1]) # temporarily remove softmax activation layer ( this will remove it both from layers list, and the 'nextLayer' )
            
            
        # remove any layers from the network, that would negatively affect inference: batchnorm (adds noise to the gradients) and softmax (as we have to compute gradients against the linear predictor not the class probabilities)
        self.removeSupportLayers(removeBNlayers)
            
            
        # add Upsampling layers before each Conv layer
        #ConvLayers = list()
        #upSampleLayers = list()
        #origCanAddLayers = self.CAN_ADD_LAYERS
        #if upsample :
#
        #    self.CAN_ADD_LAYERS = False # otherwise we would add layers 2x, once at the target location, and once at the end...
        #    for i in range(len(self.layers)) : # first save the Conv layers themselves
        #        if isinstance(self.layers[i],knnConvLayer)  :
#                    print("conv layer found at index: " + str(i))
        #            ConvLayers.append(self.layers[i])
        #        
        #for i in range(len(ConvLayers)) : # go through each conv layer
        #    targetIndex = self.layers.index(ConvLayers[i]) # get their index, and the target index is -1 before that
        #    upsampleLyer = knUpsample(  self, [-1])
        #    self.addLayerBackAt(upsampleLyer, targetIndex)
        #    upSampleLayers.append(upsampleLyer)
        #    print("added upsample layer at index: " + str(targetIndex))
        #    
        #self.CAN_ADD_LAYERS = origCanAddLayers

            
        
        Yhat = self.forward_propagate( X, False ) # use standard forward prop
        
        # generate the labels from the prediction (IE we assume that the network is trained well enough so that it can correctly classify the given data)
        dscores = np.zeros(Yhat.shape) # create derivative of scores same shape but filled with 0s
        dscores[np.arange( len(X) ), np.argmax(Yhat, axis=1)] = 1 # add a 1 at the highest scoring category: 57 (IE a chimp)  #top 1 prediction saliency map
                                                              # this is a bit weird, as we add '1' for the true label, which is a probability, but we are using the linear predictor which could be like 756432... IE yhat- y may be 756432 -1... if we get weird results, one thing that I could try is to use the maxvalue instead of 1s??
        # IV) standard Backpropagation: this comutes derivatives/Errors for each layer, setting inference to true makes sure we go back to the input layer, which ordinarily we don't need
        self.backpropagate(Yhat,  dscores, inference = True, guidedBackprop = True  )  
        


        # once finished we remove upsample layers, do this BEFORE adding back the softmax etc, as those refer to indices  before adding the upsample layers
        #for i in range(len(upSampleLayers)) :
        #    print("removing Upsample layer")
        #    self.removeLayer(upSampleLayers[i])   

        # add back each layer at the correct position
        self.addBackSupportLayers()
     
             
        #if resetLastLayer is not None :
        #     self.layers[-2].nextLayer = resetLastLayer
        #     self.addLayer(resetLastLayer)
        
        
        # V) reset weights if we used per neuron/filter inference ( make sure to do this after we have the original layers back in order so that index refers to the correct pos...)
        if ( layer != -1 ) : 
             self.layers[layer].Weights_W = origWeights
        
        return(self.layers[0].Error_D) # return the input layers derivative, which is the saliency map 


    # Smoothgrad saliency map: https://github.com/PAIR-code/saliency/blob/master/saliency/base.py
    def computeSaliency_SmoothGrad(self, X, layer = -1, targetNeuron = -1 , stdev_spread = 0.15, nsamples = 50, mode = 0, removeBNlayers = True ):  # stdev_spread: is the noise level should be between 0.1-0.5
        print ("Smoothgrad Saliency maps for layer" + str(layer) + " for neuron/filter: "  + str(layer) )

        saliencies = list()
        for i in range(len(X)) : # go through all images separately

            x_value = X[i]
            stdev = stdev_spread * (np.max(x_value) - np.min(x_value))
            
            total_gradients = None
            for j in range(nsamples): # go through the image n times
                noise = np.random.normal(0, stdev, x_value.shape) # generate noise for the image's dimensions
                x_plus_noise = x_value + noise # add the noise to the image
                #plt.imshow(deprocess_image(x_plus_noise, data['mean_image'])) # this shows the image made a bit blurry
                   
                x_plus_noise = x_plus_noise[np.newaxis] #add back an axis for the Saliency map computation ( it is always 1,1,Height,Width)
                # this is the usual Guided BP
                grad = self.computeSaliency(x_plus_noise, layer, targetNeuron, removeBNlayers)
                  
                if mode == 0:      grad = np.abs(grad) # take the absolute, i think this makes most sense for genomic data        
                elif mode == 1 : grad = (grad * grad) # square the gradients when adding up ( I suppose this makes everytihng positive), this has the effect of sharpening the focus to the really important pixels ( IE less fuzzyness)
                else :  grad = grad
      
                if total_gradients is None : total_gradients = grad
                else: total_gradients += grad 
               
            
     
            saliencies.append( (total_gradients / nsamples)[0] ) # remove the redundant first axis from the saliency map
        saliencies = np.asarray(saliencies)
        
        return(saliencies)
        
 
    # force release RAM
    def releaseRAM(self) :
      
        for i in range( 0, len(self.layers) ) :
            layer = self.layers[i]
            
            if hasattr(layer,'Error_D') : layer.Error_D = None
            if hasattr(layer,'Input_S') : layer.Input_S = None        
            if hasattr(layer,'Input_S') : layer.Input_S = None  
            
            if hasattr(layer,'Input_norm') : layer.Input_norm = None
            
            
#    def learnCycleBatch_regular(self,b_data, b_labels, eta, friction, gamma, t, batchNum) :        
#        global NETWORK_DATATYPE
#
#        b_data = np.asarray( b_data, dtype=NETWORK_DATATYPE ) ; b_labels = np.asarray( b_labels, dtype=NETWORK_DATATYPE ) # cast them, agnostically to either numpy or cupy
#        #print ("Cast to CUPY "  , flush=True  )
#        Yhat = self.forward_propagate( b_data )
#        #print ("Forward prop done"  , flush=True  )
#        b_data = None
#        self.backpropagate(Yhat,  b_labels )
#        #print ("Back prop done"  , flush=True  )
#        self.update(eta, friction, gamma, t ) 
#        #print ("Update prop done"  , flush=True  )     
#        
#    def learnCycleBatch_Stem(self,b_data, b_labels, eta, friction, gamma, t, batchNum) :        
#        global NETWORK_DATATYPE 
#        
#    def learnCycleBatch_Main(self,b_data, b_labels, eta, friction, gamma, t, batchNum) :        
#        global NETWORK_DATATYPE  
#        
#    # performs a full cycle of learning iteration for a batch: Forward/Backward prop/update
#    def learnCycleBatch(self,b_data, b_labels, eta, friction, gamma, t, batchNum) :
#        # depending on network type we execute a different path...
#        if self.mainNetwork is None and self.stemNetwork is None : self.learnCycleBatch_regular(b_data, b_labels, eta, friction, gamma, t, batchNum)
#        elif self.mainNetwork is not None : self.learnCycleBatch_Stem(b_data, b_labels, eta, friction, gamma, t, batchNum)
#        else : self.learnCycleBatch_Main(b_data, b_labels, eta, friction, gamma, t, batchNum)
#        
        
    def learn(self, train_X, train_Y, test_X, test_Y,num_epochs=500, eta=0.05, eval_train=False, eval_test=True, eval_freq = 100,  friction = 0.0,  gamma = 0.999, decayEnabled = True, decayEpochStart = 10):
        #gamma = 1.0 # disable ADAM
        self.connectLayers(False) # note: this just lets each layer know about who is the next layer, but it now does NOT instantiate Weight matrices, as we ll actually need to know the size of the input, so that is lazily instantiated dureing the first training step
        #minibatch_size = train_Y[0].shape[0] # this refers to the number of rows for matrices, and the length for vectors
        global NETWORK_DATATYPE
        
        if  train_Y[0].shape[1]  > 1 and isinstance(self.layers[-1],knSoftmax) : outPutType = OUT_MULTICLASS  # if its multicolumn AND we have softmax (IE  the cols sum to 1 for a prob), then its a multiclass classifcation problem
        elif  train_Y[0].shape[1]  > 1 : outPutType = OUT_MAE # if it has more than 1 column, but the final layer isn't softmax, then we can only evaluate this by Mean Average Error
        else : outPutType = OUT_REGRESSION # if the thing we are trying to predict has only 1 column, then it is a regression problem            
        
        # cosmetics: if we are classifying then we are evaluating via 'accuracy' where as if we are regressing, then we care about error
        
        if outPutType == OUT_REGRESSION :  evaluation = "prediction (r^2)" 
        elif outPutType == OUT_MULTICLASS : evaluation ="accuracy (%)"  
        else : evaluation = "MAE"                 
        
        results = {}
        results["epochs"] = list()
       # if eval_train: results["train_accuracy"]  = list()
       # if eval_test: results["test_accuracy"]  = list()
        results["train_accuracy"]  = list()
        results["test_accuracy"]  = list()


        decay = 0.0
        if decayEnabled : decay = 0.001 #eta / num_epochs
        if self.suppressPrint == False : print ("ORTHO8 Starting to train Neural Network for for num iterations: " + str(num_epochs) + " with LR decay enabled at: " + str(decay) , flush=True  )
        t = 0
        
        # to facilitate early stop we will record highest validation accuracy and its epoch
        results['highestAcc'] = -1.
        results['highestAcc_epoch'] = -1
        while t < num_epochs: #for t in range(0, num_epochs):
            #if ( t == 1) : self.getNumTrainableParams()
            #print ("epoch: " + str(t) , flush=True  )
        
            out_str = " | it: "  + str(t) # "[{0:4d}] ".format(t)
            start_time = time.time()

            # 1) Complete an entire training cycle: Forward then Backward propagation, then update weights, do this for ALL minibatches in sequence
            currentBatchNum = 0
            totalBatches = len(train_X)
            #for b_data, b_labels in zip(train_X, train_Y):
            for batchNum in range(len(train_X)):
                b_data = train_X[batchNum] ; b_labels = train_Y[batchNum]
                #print ("batch: " + str(currentBatchNum) , flush=True  )
                #self.learnCycleBatch(b_data, b_labels, eta, friction, gamma, t, j)
                
                b_data = np.asarray( b_data, dtype=NETWORK_DATATYPE ) ; b_labels = np.asarray( b_labels, dtype=NETWORK_DATATYPE ) # cast them, agnostically to either numpy or cupy
                #print ("Cast to CUPY "  , flush=True  )
                Yhat = self.forward_propagate( b_data )
                #print ("Forward prop done"  , flush=True  )
                b_data = None
                self.backpropagate(Yhat,  b_labels, batchNum = batchNum )
                #print ("Back prop done"  , flush=True  )
                self.update(eta, friction, gamma, t ) 
                #print ("Update prop done"  , flush=True  )
                
                barPos =  currentBatchNum / totalBatches # the bar position in %
                barPos = round(20 * barPos) # scale it to 1-20
                if self.suppressPrint == False : 
                    sys.stdout.write('\r')
                    sys.stdout.write("[%-20s] %d%%" % ('='*barPos, 5*barPos))
                    sys.stdout.flush()
                
                currentBatchNum += 1.0
            
            ### DEBUG:
           # print("Cost: " , round( self.layers[-3].getRegularizerCostTerm(), 2), flush=True )
            
            if t % eval_freq == 0 or t == num_epochs -1: # only evaluation fit every 100th or so iteration, as that is expensive
                results["epochs"].append(t) # add the epoch's number
                
                if eval_train or t == num_epochs -1:
                    accuracy = self.evalAccuracy(outPutType, train_X, train_Y)  
                    
                    out_str =  out_str + " / Training " +evaluation +": "+ str( accuracy )                  
                    results["train_accuracy"].append(accuracy)  

                if eval_test or t == num_epochs -1:
                    accuracy = self.evalAccuracy(outPutType, test_X, test_Y, validEval = True) 
                    
                    if accuracy > results['highestAcc'] :
                        results['highestAcc'] = accuracy
                        results['highestAcc_epoch'] = t
                        
                    results["test_accuracy"].append(accuracy)
                    out_str =  out_str + " / Test " +evaluation +": "+  str( accuracy )
  
            #gc.collect()
            elapsed_time = time.time() - start_time 
            if self.suppressPrint == False : print(out_str + " / " + str( round(elapsed_time) ) + " secs (LR: " + str(eta) + ")" , flush=True)
            
            # update learning rate
            if t > decayEpochStart :
                eta = eta * 1/(1 + decay * t) # as t starts as 0, this will only kick in for the 3rd iteration
            
            t += 1
            #getNetRAMUsage(self)
            #self.checkNetworkDatatype()
            
        return ( { "results" : results})
         
    
    def evalAccuracy(self, outPutType, test_X, test_Y, validEval = False) :
        N_test = len(test_Y)*len(test_Y[0])
        self.releaseRAM() # want to release RAM before this, otherwise this would add +20% RAM usage when evaluating
        #print("N_test is:" + str(N_test))
        totals = 0
        for b_data, b_labels in zip(test_X, test_Y):
        #for j in range(len(test_X)):
        #    b_data = test_X[j] ; b_labels = test_Y[j]
            b_labels = np.asarray( b_labels, dtype=NETWORK_DATATYPE ) # cast them, agnostically to either numpy or cupy
            yhat = self.forward_propagate(b_data, False, False)
            b_labels = b_labels.reshape(b_labels.shape[0],-1) # force 2D structure
            b_data = None
            #self.releaseRAM() # otherwise we would get an extra +10% memory usage
            
            # depending on if we are in a classification or regression problem, we evaluate performance differently
            if outPutType == OUT_REGRESSION :  
                currentRate = np.corrcoef( yhat  , b_labels  , rowvar=0)[1,0]**2  # from cupy v5.0.0b4, corrcoef is supported, so dont need to cast this, as that is expensive
                N_test = len(test_Y) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
              
            elif outPutType == OUT_MULTICLASS : currentRate = calc_Accuracy(yhat,b_labels )    # calculate accuracy, this is ROUNDED 
            
            else : # mean absolute error
                currentRate = -np.mean(np.abs(yhat - b_labels)) # negative as the rest of the metrics are accuracy, IE the greater the error, the lower the accuracy
                N_test = len(test_Y) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
     
         
               
            totals = totals +currentRate # sum in all minibatches

        accuracy = round( float(totals)/N_test,5)
        return(accuracy)
    
    
    def checkNetworkDatatype(self) :

        global NETWORK_DATATYPE
        
        foundWrongDatatype = False
        for i in range( 0, len(self.layers) ) :
            layer = self.layers[i]
            
            if hasattr(layer,'Weights_W') and layer.Weights_W is not None: 
                print("Layer " +str(i)+ "("+ str(type(layer).__name__) +") Weights_W TYPE is : " + str(layer.Weights_W.dtype), flush=True)
                if NETWORK_DATATYPE != layer.Weights_W.dtype :
                    foundWrongDatatype = True
                    print("WRONG DATATYPE", flush=True)
                
            if hasattr(layer,'Error_D')  and layer.Error_D is not None: 
                print("Layer " +str(i)+ "("+ str(type(layer).__name__) +") Error_D TYPE is : " + str(layer.Error_D.dtype), flush=True)
                if NETWORK_DATATYPE != layer.Error_D.dtype :
                    foundWrongDatatype = True
                    print("WRONG DATATYPE", flush=True)
                
            if hasattr(layer,'Input_S')  and layer.Input_S is not None:
                print("Layer " +str(i)+ "("+ str(type(layer).__name__) +") Input_S TYPE is : " + str(layer.Input_S.dtype), flush=True)
                if NETWORK_DATATYPE != layer.Input_S.dtype :
                    foundWrongDatatype = True
                    print("WRONG DATATYPE", flush=True)
                

                

        print ("Network supposed to be datatype: " + str(NETWORK_DATATYPE)  + " and foundWrongDatatype: " + str(foundWrongDatatype), flush=True)


    def backupLayer(self,layer):
        if self.backedupLayers == None : self.backedupLayers = {}
        self.backedupLayers[layer] = list()
        self.backedupLayers[layer].append( layer.Weights_W.copy() )
        self.backedupLayers[layer].append( layer.Momentum.copy() )
        self.backedupLayers[layer].append( layer.Past_Grads.copy() )
 
        if layer.biasEnabled :
            self.backedupLayers[layer].append( layer.Weights_bias.copy() )
            self.backedupLayers[layer].append( layer.Bias_Momentum.copy() ) 
            self.backedupLayers[layer].append( layer.Past_Grads_bias.copy() )
            
    def restoreLayer(self,layer):
        if layer in self.backedupLayers:
            layer.Weights_W =  self.backedupLayers[layer][0]
            layer.Momentum =  self.backedupLayers[layer][1]
            layer.Past_Grads =  self.backedupLayers[layer][2]

            if layer.biasEnabled :
                layer.Weights_bias =  self.backedupLayers[layer][3] 
                layer.Bias_Momentum =  self.backedupLayers[layer][4]
                layer.Past_Grads_bias =  self.backedupLayers[layer][5] 
                
            self.backedupLayers.pop(layer, None)  
        else : print ("Layer was NOT backed up!")
        

        
    def deleteNeuron(self,layer, targetNeuron):
        layer_nextLayer = layer.nextLayer
        if type(layer.nextLayer) != knnLayer: layer_nextLayer = layer_nextLayer.nextLayer # need to find the next layer that has weights, this is usually comes after the activation
     
        # delete it from the neuron from the target layer's weights
        layer.Weights_W = np.delete(layer.Weights_W, targetNeuron, 1)  # delete neuron 2 (for the main weight matrices it is in the cols)
        layer.Momentum = np.delete(layer.Momentum, targetNeuron, 1)  # delete neuron 2 (for the main weight matrices it is in the cols)
        layer.Past_Grads = np.delete(layer.Past_Grads, targetNeuron, 1)  # delete neuron 2 (for the main weight matrices it is in the cols)

        if layer.biasEnabled :
            layer.Past_Grads_bias = np.delete(layer.Past_Grads_bias, targetNeuron, 1)  # delete neuron from biases
            layer.Weights_bias = np.delete(layer.Weights_bias, targetNeuron, 1)  # delete neuron from biases
            layer.Bias_Momentum = np.delete(layer.Bias_Momentum, targetNeuron, 1)  # delete neuron from biases
                               
        # neuron 2 also needs to be removed from the next layer's Weights, as if we delete 1 neuron from b, from W1_aXb , then next layer's W is W2_bXc , so we need to remove a row from next layer's matrix too
        layer_nextLayer.Weights_W = np.delete(layer_nextLayer.Weights_W, targetNeuron, 0) 
        layer_nextLayer.Momentum = np.delete(layer_nextLayer.Momentum, targetNeuron, 0) 
        layer_nextLayer.Past_Grads = np.delete(layer_nextLayer.Past_Grads, targetNeuron, 0) 
        
        
    # restores a previously deleted neuron
    def restoreLastDeletedNeuron(self) :
        if self.lastdeletedNeuron_layer is not None: 
            layer = self.lastdeletedNeuron_layer
            layer.Weights_W = self.lastdeletedNeuron_layer_Weights_W_orig
            layer.Weights_bias = self.lastdeletedNeuron_layer_Weights_bias_orig
            layer.Momentum = self.lastdeletedNeuron_layer_Momentum_orig
            layer.Bias_Momentum = self.lastdeletedNeuron_layer_Bias_Momentum_orig
            layer.Past_Grads_bias = self.lastdeletedNeuron_layer_Past_Grads_bias_orig
            layer.Past_Grads = self.lastdeletedNeuron_layer_Past_Grads_orig
            
            layer_nextLayer = self.lastdeletedNeuron_layer_nextLayer
            layer_nextLayer.Weights_W = self.lastdeletedNeuron_layer_nextLayer_Weights_W_orig
            layer_nextLayer.Weights_bias = self.lastdeletedNeuron_layer_nextLayer_Weights_bias_orig
            
         
    ####### Debugger functions: used for numerical Gradient checking 
    # replaces the weights in the network from an external source: a 1D array (IE the same format as that is returned by getWeightsAsVector() )
    def setWeights(self, allWeights):
        for i in range(1, self.num_layers):  # go through all layers except first, as that cannot have weights
            allWeights = self.layers[i].addDebugData(allWeights) # if layer has weights, then this function takes as many as needed to fill up layer weight matrix, adds them , and removes them from the list before passing back what is left
               

    def regressionCost_SumSquaredError(self, batch_data, y):   #
        #Compute cost for given X,y, use weights already stored in class.
        yHat = self.forward_propagate(batch_data,False, False)
        y = y.reshape(y.shape[0],-1) # force 2D structure
        batch_num_samples = batch_data.shape[0]  # normalise by the batch size
        J =  0.5*  np.sum((y-yHat)**2) / batch_num_samples + self.getRegularizerCostTerms()
        return J
    

    def multiclassCost_softMax(self, batch_data, y):
        yHat = self.forward_propagate(batch_data, False, False)
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
        #minibatch_size = y.shape[0]
        yHat = self.forward_propagate(data, False, False)
        self.backpropagate(yHat, y) # we need o backpropagate also, as we need to know the CURRENT error, IE that is resulted from the weights that we have now (as the errors we have atm are the errors due to the previous iteration)
        origDEBUG = self.DEBUG
        self.DEBUG = True # temporarily set this, so that the next function saves the weight gradients
        self.update(0, 0.0, 0.99, 1) # this calculates the current weight GRADIENTS into an array, without actually updating them ( as eta is 0)
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
        
    # 1 input, multiple output (IE multiple objective functions that are blended together)
class knn_forked(knn):
    def __init__(self, suppressPrint = False, optimizer = 0): # this assumes a single forking-point that may have many subNetwork (could use list of lists to be able to have multiple forkingpoints)
        super().__init__(suppressPrint, optimizer) # call init on parent   
        self.subNetwork = None
        self.lastSubNetworkYhat = None # save the outputs of each subnetwork
        self.subNetwork_errorContributions = None
        self.blendParam = None
        self.subNetwork_trainYs = None
        self.subNetwork_validYs = None

     
    def connectLayers(self, initWeights = True) :   # connects up each layer to its next, and initialises the Weighted layers with the correct dimension matrices  
        # also update this network
        super().connectLayers(initWeights = initWeights) 
        
        if self.subNetwork is not None :   
            self.subNetwork.connectLayers(initWeights = initWeights)


    def addSideNetwork(self, sideNetwork,  blendParam,forkingLayer,trainY, validY) :
        self.forkingLayer = forkingLayer # the layers whose output the branched networks should feed off of
        self.blendParam = blendParam # blendParam controls how much of the network should favour the branches over the main stem ( first refers to the main stem)
        self.subNetwork = sideNetwork
        
        # each subnetwork has its own objective function, so we need true labels for each 
        self.subNetwork_trainYs =trainY
        self.subNetwork_validYs =validY
        

    # Forked FP: run data through the main Stem network (this), 
    def forward_propagate(self, data, train = True, saveInput = True, forceCast_toCPU = False, finalLayer = None): # data is the design matrix

        if self.subNetwork is None : 
            raise ValueError('subNetwork is None')
            return None

        Yhat = super().forward_propagate(data, train = train, saveInput = True, forceCast_toCPU = forceCast_toCPU, finalLayer = finalLayer) # we want to save the input, as the side branches will work off of those

        # the output of the Branching layer is saved on the NEXT layer's Input
        self.lastSubNetworkYhat = self.subNetwork.forward_propagate(self.forkingLayer.nextLayer.Input_S, train = train, saveInput= saveInput, forceCast_toCPU = forceCast_toCPU, finalLayer = finalLayer) # must not cast to CPU here, as the subnetwork output is never exposed

        return( Yhat )

    # Forked BP: BP on each subnetwork up until the forking points, then BP on the main network up until each branching point, and at there, blend the Errors together, and then continue BP back towards the Input of the main network (this)
    def backpropagate(self, yhat, labels, inference = False, guidedBackprop = False , targetLayer = -1, batchNum = -1): # Inference is a flag that switches on Guided BP and makes sure that we BP to the first layer to get D1 (otherwise that is not needed for Delta Ws )
        if self.subNetwork is None : 
            raise ValueError('subNetwork is None')
            return None
        
        global NETWORK_DATATYPE
        # I) backpropagate subnetwork to where it forked off the main network
        labels_sub = self.subNetwork_trainYs[batchNum].reshape(self.subNetwork_trainYs[batchNum].shape[0],-1) # grab the relevant batch' Y true
        
        labels_sub = np.asarray( labels_sub, dtype=NETWORK_DATATYPE ) # cast them, agnostically to either numpy or cupy
        # calculate first Error Delta directly from output
        Error_Delta_Output = np.asarray( (labels_sub - self.lastSubNetworkYhat), dtype=NETWORK_DATATYPE ) #  we saved the Yhats from last backprop
        self.subNetwork.layers[-1].backpropagate(Error_Delta_Output, inference = True, guidedBackprop = guidedBackprop ) # we set the Inference to be True, as that forces to generate an error, as we need to blend those together
  
 
        # II) backpropagate on the main stem back to the forking ( we have broken the backprop chain there) 
        # need to 'break' the backward connections for each layer where there is a branch, a in backpropagation we need to blend the errors together, then manually propagate back further with those errors
        origPrevLayerForForked = self.forkingLayer.prevLayer.prevLayer
        self.forkingLayer.prevLayer.prevLayer = None # we break the BP chain 1 layer before, as the Error from the Forking layer is stored not on itself, but a layer below it
### First just try NOT breaking/restoring the connection ( IE less efficient doing the BP 2x)
### Then try not restoring
        

        labels = labels.reshape(labels.shape[0],-1)
        # calculate first Error Delta directly from output
        Error_Delta_Output = np.asarray( (labels - yhat), dtype=NETWORK_DATATYPE ) # (yhat - labels)#.T  # -> this would cause all gradients to be inverted
        #print ("Error_Delta_Output dims: " + str(Error_Delta_Output.shape)  + " // yhat dims: " + str(yhat.shape) + " // labels dims: " + str(labels.shape) )
        self.layers[targetLayer].backpropagate(Error_Delta_Output, inference = True, guidedBackprop = guidedBackprop ) # we set the Inference to be True, as that forces to generate an error, which we have to pass back to the subnetwork
        # only the main network could have a 'truncated' backprop chain, as the main network is the only one that is connected to the Input which we might be interested in
        
        # III) Aggregate the errors at each branchpoint according to the blend params     
        Error_total = self.forkingLayer.prevLayer.Error_D * (1 - self.blendParam )# the basic error is the main stem networks error at the forking layer, blended at the weight of 1-All_Alphas
        Error_total += (self.subNetwork.layers[0].Error_D  * self.blendParam)

        # IV) resume backpropagation with the aggregated error that represents error from both main and subnetwork
        self.forkingLayer.prevLayer.prevLayer = origPrevLayerForForked # need to restore connection otherwise it wont backrpop
        self.forkingLayer.prevLayer.backpropagate(Error_total, inference = inference, guidedBackprop = guidedBackprop ) 
        # (resume from the layer directly beneath the forking layer, as that is where  we left off, IE that is where we 'overwrite' the original error)

    def update(self, eta, friction, gamma, epoch_num ):
        if self.subNetwork is None : 
            raise ValueError('subNetwork is None')
            return None
        
        self.subNetwork.update(eta, friction, gamma, epoch_num)

        # also update this network
        super().update(eta, friction, gamma, epoch_num)

        
    # as we are NOT using the test_Y, we kindof hack it by using the 
    def evalAccuracy(self, outPutType, test_X, test_Y, validEval = False) :
        # need to override the output type, as if we use the one passed into the function that would be for the AE network which is regression...
        if len( self.subNetwork_trainYs[0].shape ) != 1 and isinstance(self.subNetwork.layers[-1],knSoftmax) : outPutType = OUT_MULTICLASS # if the thing we are trying to predict has only 1 column, then it is a regression problem
        else : outPutType = OUT_REGRESSION  
        
        #print("eval accuracy for Forked net with evalAccuracy:" , outPutType)
        N_test = len(test_Y)*len(test_Y[0]) # should be the same for both AE and the assoc networks
        self.releaseRAM() # want to release RAM before this, otherwise this would add +20% RAM usage when evaluating
        #print("N_test is:" + str(N_test))
        totals = 0
        counter = 0
        for b_data, b_labels in zip(test_X, test_Y):
            if validEval : b_labels =  self.subNetwork_validYs[counter] # a REALLY hacky way to ensure we are evaluating the subnetwork correctly
            else :  b_labels =  self.subNetwork_trainYs[counter]
            # take care to evaluate against the Y_true of the side network ( as the b_labels in this context would be the validX, as the main network is an AE)
            b_labels = np.asarray( b_labels, dtype=NETWORK_DATATYPE ) # cast them, agnostically to either numpy or cupy
            b_labels = b_labels.reshape(b_labels.shape[0],-1) # force 2D structure
            
            self.forward_propagate(b_data, False, False)
            yhat = self.lastSubNetworkYhat # the actual output of the network is the output of the sidenetwork which is generated in the above step (we assume that there is only one side network)
            b_data = None
            
            
            # depending on if we are in a classification or regression problem, we evaluate performance differently
            if outPutType == OUT_REGRESSION :  
                currentRate = np.corrcoef( yhat  , b_labels  , rowvar=0)[1,0]**2  # from cupy v5.0.0b4, corrcoef is supported, so dont need to cast this, as that is expensive
                N_test = len(self.subNetwork_validYs) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
              
            elif outPutType == OUT_MULTICLASS : currentRate = calc_Accuracy(yhat,b_labels )    # calculate accuracy, this is ROUNDED 
            
            else : # mean absolute error
                currentRate = -np.mean(np.abs(yhat - b_labels)) # negative as the rest of the metrics are accuracy, IE the greater the error, the lower the accuracy
                N_test = len(self.subNetwork_validYs) # as we are testing correlation, the N should refer to the number of batches, and NOT the total number of observations
            
            
            counter += 1
                                       
            totals = totals +currentRate # sum in all minibatches

        accuracy = round( float(totals)/N_test,5)
        return(accuracy)
        
        
        
    # multiple inputs, single output
class knn_branched(knn):
    def __init__(self, suppressPrint = False, optimizer = 0, subNetworks = [], subNetworkInputs = [], subNetwork_outputDims = []):
        super().__init__(suppressPrint, optimizer) # call init on parent   
        self.subNetworks = subNetworks
        self.subNetworkInputs = subNetworkInputs
        self.subNetwork_outputDims = subNetwork_outputDims # in order to efficiently copy together the outputs of each network, we need to know in advance the number of columns
        self.subNetworks_errorContributions = list()
     
    def connectLayers(self, initWeights = True) :   # connects up each layer to its next, and initialises the Weighted layers with the correct dimension matrices  
        for i in range(len(self.subNetworks)) :    
            self.subNetworks[i].connectLayers(initWeights = initWeights)

        # also update this network
        super().connectLayers(initWeights = initWeights) 


    # Branched FP: perform FP on all branches leading up to the main network (this), glue the outputs together as a new Input, and then proceed with the FP as normal
    def forward_propagate(self, data, train = True, saveInput = True, forceCast_toCPU = False, finalLayer = False): # data is the design matrix

        start = 0
        end = self.subNetwork_outputDims[0]
        gluedData = np.zeros( (data.shape[0], np.sum(self.subNetwork_outputDims) ) , dtype =data.dtype)  # assume a FC output, IE not convnets with channels..
        for i in range(len(self.subNetworks)) :
            gluedData[:,start:end] = self.subNetworks[i].forward_propagate(self.subNetworkInputs[i], train = train, saveInput= saveInput, forceCast_toCPU = False, finalLayer = False) # must not cast to CPU here, as the subnetworks output is never exposed
            start = end
            end = start + self.subNetwork_outputDims[i]

        return( super().forward_propagate(gluedData, train = train, saveInput = saveInput, forceCast_toCPU = forceCast_toCPU, finalLayer = finalLayer) )
    
    # Branched BP: BP on the main network up until branching point, then split the Error there into the relevant sections which are then fed to each subnetwork that individually continue BPing to their respective inputs as usual
    def backpropagate(self, yhat, labels, inference = False, guidedBackprop = False , targetLayer = -1, batchNum = -1): # Inference is a flag that switches on Guided BP and makes sure that we BP to the first layer to get D1 (otherwise that is not needed for Delta Ws )
        global NETWORK_DATATYPE
        labels = labels.reshape(labels.shape[0],-1)
        # calculate first Error Delta directly from output
        Error_Delta_Output = np.asarray( (labels - yhat), dtype=NETWORK_DATATYPE ) # (yhat - labels)#.T  # -> this would cause all gradients to be inverted
        
        #print ("Error_Delta_Output dims: " + str(Error_Delta_Output.shape)  + " // yhat dims: " + str(yhat.shape) + " // labels dims: " + str(labels.shape) )
        # only the main network could have a truncated backprop chain
        self.layers[targetLayer].backpropagate(Error_Delta_Output, inference = True, guidedBackprop = guidedBackprop ) # we set the Inference to be True, as that forces to generate an error, which we have to pass back to the subnetworks
        LastError = self.layers[0].Error_D
        
        # go through each subnet, and feed the relevant part of the error to them
        start = 0
        end = self.subNetwork_outputDims[0]
        self.subNetworks_errorContributions = list()
        for i in range(len(self.subNetworks)) :
            currentError = LastError[start:end,:]
            self.subNetworks_errorContributions.append(np.sum(currentError))
            self.subNetworks[i].layers[-1].backpropagate(currentError, inference = inference, guidedBackprop =guidedBackprop )
            start = end
            end = start + self.subNetwork_outputDims[i]


    # computes the % of errors of each subnetwork, IE this method can be used to evaluate how much does each subnetowrk contributes to the final prediction accuracy (IE inversely proportional to the error %... higher the error the less it contributes)
    def returnErrorContributions(self ):
        errorTotal = np.sum(self.subNetworks_errorContributions)
       
        percErrors = list()
        for i in range(len(self.subNetworks_errorContributions)) :
           percErrors.append(self.subNetworks_errorContributions[i]/errorTotal)
            
        return(percErrors)
        
        
    def update(self, eta, friction, gamma, epoch_num ):
        for i in range(len(self.subNetworks)) :    
            self.subNetworks[i].update(eta, friction, gamma, epoch_num)

        # also update this network
        super().update(eta, friction, gamma, epoch_num)
        
       
        
# base class for all knn layers: should never be directly instantiated
class knnBaseLayer :
    def __init__(self, iparentNetwork = None, iparams = -1 ):
        self.params = iparams
        self.parentNetwork= iparentNetwork
        self.prevLayer = None  # this should be 'knnBaseLayer', but R cannot reference this type of class in the class definition, so we have to use a 'dummy' class
        self.nextLayer = None
        self.activationType = False # if this layer is an activation type of layer
        self.supportLayerType = False # if this is not a proper layer, so it is like a batchnorm or maxpool etc
        # add this layer to its parent Neural Network's list of layers
        self.parentNetwork.addLayer(self)

    
    # produces output from the layer's data, this is used by forward_propagate
    def generateOutput(self,Input, train = True, saveInput = True) : ... 

    
    # passes along the output of this layer to the next, if any 
    def forward_propagate(self,Input, train = True, saveInput = True, finalLayer = None) :
        output = self.generateOutput(Input, train, saveInput)
          
        if self.nextLayer is None:  # if there is saveInputno next layer, IE this is the last one the Output
            return(output)
        else : # if there are any next layers still, we recursively call them
            #if finalLayer is not None : print("looking for an early termination, finalLayer == self.nextLayer is: " + str(finalLayer == self.nextLayer) + " where finalLayer is: " + str(finalLayer) + " /  self.nextLayer is: " + str( self.nextLayer))
            
            if finalLayer is not None and finalLayer == self.nextLayer : return(output) # if we specified an early termination point
            
            return( self.nextLayer.forward_propagate(output, train, saveInput, finalLayer = finalLayer) ) # send output from this layer to the next
        
    
    # computes the Gradient, and optionally the Error Delta for the next layer. The Error of the current layer, based off from the error passed back from the layer after this during backpropagation
    def calcGrads_and_Errors(self,Input, computeErrors = True, inference = False) : ...
    
                 
    # passes errors backwards from the output onto preceding layers
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this   
        if self.prevLayer is not None or inference: # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            # generate output from this layer
            #print("backprop called on layer " + str(self.parentNetwork.layers.index(self)) )
        
        
            error_D = self.calcGrads_and_Errors(Input,  inference or self.prevLayer.prevLayer is not None, inference ) # if the previous layer is input, we dont need to actually compute Errors for previous layer, we only compute gradients for the normal case, we only want errors for the Input layers for inference/saliency runs
            if self.prevLayer is not None: return( self.prevLayer.backpropagate(error_D, inference, guidedBackprop ) ) # this can happen on Inference runs
             # else: if there are no prev layers then stop backpropagating ... IE we reached the INPUT layer
    
    
    def update(self, eta, friction, gamma, epoch_num) : ...    
    # generic update function, (it usually updates weights for layers that have them)


    # Lets each layer know about its neighbours: the previous and next layer in the stack: this is called once the entire network has been set up
    def initConnections(self, prevNext = {}, initWeights = True ) :
        # check previous layer
        if prevNext["PREV"] is not None : self.prevLayer = prevNext["PREV"] 
         
        # check next layer
        if prevNext["NEXT"] is not None : self.nextLayer = prevNext["NEXT"]
        
    def getDebugInfo(self,dataSoFar, Type) : return(dataSoFar)  # the base method still needs to return the passed in data if nothing else   
    def addDebugData(self,allWeights)  : return(allWeights)   # the base method still needs to return the passed in data if nothing else
        

# the main 'Weighted' neural network layer, used by all FC (Fully Connected) layers
# params: a scalar, it is simply just the number of units in his layer
class knnLayer(knnBaseLayer):
    def __init__(self, iparentNetwork, iparams, isubtype, ibiasEnabled = True, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0, p_dropout = -1, selu_dropout = False):
        super().__init__(iparentNetwork, iparams) # call init on parent
        
        self.Weights_bias = None
        self.Momentum = None
        self.Bias_Momentum = None
        self.regularizer = regularizer
        self.subtype = isubtype # as we have 3 different subtypes, for Input/Output and Hidden, we need to differentiate it via this flag
        self.Lambda = shrinkageParam
        self.biasEnabled = ibiasEnabled
        self.layer_numUnits = self.params[0]  # here params has just 1 element  # number of neurons in layer
        self.last_input_shape = None #  keep the original shape of the input before stretchin, used for back propagation
          
 
        # The activation function is an externally defined function (with a derivative)
        #self.activation = iactivation
        #self.Output_Z= None # Z is the matrix that holds output values
        self.Weights_W = None # weights are stored as 'incoming' IE they refer to the weights between this layer and the one before
        self.Input_S = None # S is NULL matrix that holds the inputs to this layer
        self.Error_D = None # D is the matrix that holds the deltas for this layer
       # self.Derivative_Fp = None # Fp is the matrix that holds the derivatives of the activation function applied to the input
        self.W_grad = None # hold the gradients for last backprop step
        self.W_bias_grad = None # same for bias
        #self.DEBUG_WeightGrads = None # holds the debug weights for this layer (only used if parent layer has DEBUG == TRUE)
         
        self.dropout_mask  = None # the mask where to apply the dropout
        self.selu_dropout = False
        
        if p_dropout == -1 : self.p_dropout  = p_dropout # p_dropout is the % of neurons to be SWITCHED OFF ( -1 for not using this feature)
        else : 
            if selu_dropout :
                global SELU_LAM 
                global SELU_ALPHA 
                self.selu_dropout = True
                self.lambd = SELU_LAM
                self.alpha = SELU_ALPHA
                self.aprime = -self.lambd * self.alpha
                
                self.q = 1.0 - p_dropout  # q is the KEPT probability ( which is what we use to scale as well) == p_dropout
                self.p = p_dropout # p is the to DROP probability
        
                self.a = (self.q + self.aprime**2 * self.q * self.p)**(-0.5)
                self.b = -self.a * (self.p * self.aprime)    
                
            self.p_dropout  =  1.0 - p_dropout #store the % of neurons to be KEPT
        
        
        if self.parentNetwork.suppressPrint == False : print( "layer "+self.subtype + " (dropout: "+ str(self.p_dropout) +") is regularized by: " + self.regularizer+ " / its params are ", str(self.params) )
     
          
#    # sets up relationships to neighbouring layers, and if we have previous layer (IE it is not an input or a nested minilayer in a Conv), then we can init the weight matrix to the correct dimension
#    def initConnections(self, prevNext = {}, initWeights = True ): 
#        super().initConnections(prevNext)
#        
#        # Weights can be initialised only by knowing the number of units in the PREVIOUS layer (but we do NOT need to know the size of the minibatches (IE n), as Weight matrix' dimension does not depend on that
#        if (self.prevLayer is not None and initWeights) : # IE the we are testing if, subtype != LAYER_SUBTYPE_OUTPUT
#            prevLayer_size = self.prevLayer.params[0] # find out how big the next layer is 
#            self.initWeightMatrix(prevLayer_size)

     
    # (this is usually called from initConnections(), but this might also be called directly from outside, by the conv layer for example)
    def initWeightMatrix(self, prevLayer_size) :
        # print("initWeightMatrix for subtype: "+ str(self.subtype) + " / prev layer has size: "+ str(prevLayer_size) + " / self.layer_numUnits: " +str(self.layer_numUnits) )
        # Weights are sized as: rows: number of units in previous layer, cols: number of units in current layer (IE the minibatch size doesnt matter) 
        #print("initting weight matrix for FC layer, prevLayer_size: " +str(prevLayer_size), flush=True)
        
        # Knet New
        self.Weights_W = truncatedGaussian([ int(prevLayer_size),self.layer_numUnits], np.sqrt(2.0 / prevLayer_size ))  # HE init with truncated normal ( SD = sqrt(2/Fan_in) )
        ## KNet orig
        #self.Weights_W = np.random.normal(size=[ int(prevLayer_size),self.layer_numUnits], scale=np.sqrt(2.0 / prevLayer_size ))
        
        
        self.Momentum = np.zeros((self.Weights_W.shape[0], self.Weights_W.shape[1]), dtype=self.Weights_W.dtype) # stores a 'dampened' version of past weights (IE its an older version of the above with the same dimensions)
        self.Past_Grads = np.zeros((self.Weights_W.shape[0], self.Weights_W.shape[1]), dtype=self.Weights_W.dtype) # stores a 'dampened' version of past weights, Squared, used by RMSProp
           
        if self.biasEnabled :  # if we have bias/intercept, we add a row of Weights that we keep separate
            
            # Knet New
            self.Weights_bias = np.zeros( (1,self.Weights_W.shape[1]), dtype=self.Weights_W.dtype )  # HE biases as inited as zero
            ## KNet orig
            #self.Weights_bias = np.random.normal(size=(1,self.Weights_W.shape[1]), scale=np.sqrt(2.0 / prevLayer_size ))  
            
            
            self.Bias_Momentum = np.zeros( (1,self.Weights_bias.shape[1]), dtype=self.Weights_W.dtype ) # stores a 'dampened' version of past weights  for intercepts
            self.Past_Grads_bias = np.zeros( (1,self.Weights_bias.shape[1]), dtype=self.Weights_W.dtype ) # stores a 'dampened' version of past weights  for intercepts, Squared, used by RMSProp

  
    # performs the addition of the bias terms effect  onto the output
    def add_bias(self, ZW) :
        if(self.biasEnabled == False) : return(ZW) # if bias wasn't enabled in the first place we just return the orig

        ZW += self.Weights_bias # we simply add the bias to every row, the bias is implicitly multipliedby 1
        return(ZW) 
    
         
    # as knn layer can be 1 of 3 subtypes, we have to produce an output for forward propagation differently
    def generateOutput(self, Input, train = True, saveInput = True) :
        self.last_input_shape = Input.shape # save this for back prop
        # if its the first layer, we just return the data that was passed in
        if (self.subtype == LAYER_SUBTYPE_INPUT) :

            # if saveInput : self.Input_S  = Input
            return( Input  ) # we dont save the Input away for the Input layer as that is never used for anything
               
            # if its NOT an input, then all FC layers will have weights (even Output, as weights are now stored as 'incoming' weights between this and prev layer)
        else : 
            self.Input_S  = Input
            if self.Weights_W is None : self.initWeightMatrix(Input.shape[1]) # lazy instantiation
            Output_Z = self.Input_S.dot(self.Weights_W)  # the output is constructed by first multiplying the input by the weights
            if saveInput is False: self.Input_S  = None # save this away for BP, but dont do this for Evaluation runs
       
        Output_Z = self.add_bias(Output_Z) # we then add the intercept (the linear predictor is now complete)
         
        # non Output subtype layers need to figure out the rate of change in the output (this is redundant, if we are just making predictions, IE backpropagation does NOT follow this)
        #if (self.subtype != LAYER_SUBTYPE_OUTPUT) :  self.Derivative_Fp = self.activation(self.Output_Z, deriv=True).T  # this is transposed, as D is also transposed during back propagation
        Output_Z = self.dropout_forward(Output_Z, train)
        
        # output is completed by passing the linear predictor through an activation (IE we squash it through a sigmoid)
        #self.Output_Z = self.activation(self.Output_Z) # if its a hidden layer (or output), we need to activate it
        #self.parentNetwork.checkNetworkDatatype()
        return ( Output_Z )
    
    def dropout_forward(self, Output_Z, train = True):
        if self.p_dropout == -1 or train == False : return Output_Z
        else :
            global GPU_mode
            global NETWORK_DATATYPE 

            # Alpha dropout used for SELUS: set 'dropped' neurons to an alpha parameter rather than 0 (and also scale)
            if self.selu_dropout :
                # generate mask, in shape of that for the indices that we want to keep ( IE True for the ones we do keep)
                self.dropout_mask = np.random.rand(*Output_Z.shape) < self.q     
                # I think that we are NOT scaling as in regular dropout by "/keep", but we actually do that in a special way via the 'a and b' thing 
                Output_Z[~self.dropout_mask] = self.aprime # set dropped neurons to be alpha prime ("~" this inverts the indices, IE gets the one that we do NOT want
                Output_Z = self.a*Output_Z + self.b # scale the activations        
            else : 
                # From Cupy v5.0.0b4, binomial IS supported, so we try that, otherwise we get masive slowdowns
                self.dropout_mask = np.random.binomial(1, self.p_dropout, size=Output_Z.shape).astype(NETWORK_DATATYPE)
                self.dropout_mask /= self.p_dropout   # scale the activations of the surviving neurons by the 'keep %' to compansate for the lost neurons
                Output_Z *= self.dropout_mask
                
            return Output_Z
        
        

    
    def dropout_backward(self, inference = False):
        if self.p_dropout != -1 and inference == False: # if we are running deep dreaming, we don't want to apply backprop dropout (as we didn't apply it in the forward prop either)
            if self.selu_dropout : self.Error_D[~self.dropout_mask] *= self.aprime  # SELU alpha dropout: set to alpha prime where neurons were switched off
            else : self.Error_D *= self.dropout_mask # usual dropout: zero out the error where neurons were switched off
            self.dropout_mask = None
            
  
    # Compute Gradients and (optionally) the accumulated error up until this layer, this usually happens by 
    # each layer's Delta, is calculated from the Delta of the Layer +1 outer of it, and this layer's Weights scaled by the anti Derivative Gradient
    def calcGrads_and_Errors(self, Error_current, computeErrors = True, inference = False) :
        #print("BEFORE Error_current dims: " + str(Error_current.shape) )
        Error_current = Error_current.reshape(Error_current.shape[0],-1) # force 2D structure
        #print("AFTEr Error_current dims: " + str(Error_current.shape) )
         
       # if (self.subtype == LAYER_SUBTYPE_OUTPUT) :
       #    # print("Output Error dims: " + str(Error_current.shape) )
       #     self.Error_D = Error_current # this is stored as Transposed D ()
       # else : # 'Error_current' here basically refers to 'current Layer$Error_D'
       #    # print("weightToUse dims: " + str(weightToUse.shape) + " / Error_current dims: " + str(Input.shape) + " / Derivative_Fp dims: " + str(self.Derivative_Fp.shape))
       #     #self.Error_D = self.nextLayer.Weights_W.dot(Error_current) * self.Derivative_Fp # the current layer's Delta^T = (NextWeights_W * D_next^T) *Schur* F^T  
       #     # this is stupid, I am accessing BOTH the weights AND the Error_D for the next layer,
       #     # Instead I should compute both of those on the next layer, and only pass the 'inactivated' version
       #      self.Error_D = Error_current * self.Derivative_Fp

        self.Error_D = Error_current
        self.dropout_backward(inference)
        #print("FC layer self.Error_D is: " + str(self.Error_D))
        #if (self.subtype == LAYER_SUBTYPE_INPUT) :print ("Input layer calcGrads_and_Errors")
        if (self.subtype == LAYER_SUBTYPE_INPUT) : return (None) # if this is the input layer, then this won't have Weights, so break out (we would normally only reach here for an input layer if we are computing saliency maps)
        
        global KNET_ORIG
        
        # compute gradients
        num_samples_minibatch = self.Input_S.shape[0] # safest way to get the number of items, as that is always produced and it is always N x something
        self.W_grad = self.Input_S.T.dot(self.Error_D) # the weight changes are Input^T x Error_D
        self.W_grad /= num_samples_minibatch # normalise by N, to keep the same as the cost function    
        
        if KNET_ORIG : self.W_grad *= -1 # Knet Orig version
        
        self.W_grad += self.getRegularizerGrad() # I am applying the regularizers AFTEr normalizing for minibatch size... (as I am assuming that the L2 norms come externally, and determined to be just the right size)

        if inference == False : self.Input_S = None # no longer need this so release memory
        
        if(self.biasEnabled == True) :
            self.W_bias_grad = np.sum(self.Error_D, axis =0)  # via implicit multiplication of D by 1 ( IE col sums)
            self.W_bias_grad /= num_samples_minibatch
            
            if KNET_ORIG :self.W_bias_grad *= -1 # Knet Orig version
        
        # If error computation was requested, we return that
        if computeErrors : 
            Error_current = self.Error_D.dot(self.Weights_W.T)
            if inference == False : self.Error_D = None # if we are NOT in inference mode, IE we dont want to save the Errors, we can release this
            
            
            return(Error_current)  
        else : 
            #print("Last hidden layer before input, no need to compute Errors")
            if inference == False : self.Error_D = None # if we are NOT in inference mode, IE we dont want to save the Errors, we can release this
            return (None) # if not (like if previous layer is the Input), then Errors are not needed
        # (this is because Dprev = W_current*D_current), IE, if this is Hidden Layer 1, then we have already computed the gradients for this above, and it is pointless to compute another large error matrix that will never be used

       #    self.Error_D = weightToUse.dot(Input) * self.Derivative_Fp # the current layer's Delta^T = (NextWeights_W * D_next^T) *Schur* F^T  
             

    # updates the Bias Weights separately ( if enabled)
    def update_bias_weights(self, eta, friction, gamma, epoch_num) :
        if(self.biasEnabled == True) :
       
            global OPTIMIZER_SGD
            global OPTIMIZER_AMSGRAD            
            global KNET_ORIG
            global EPSILON
            
            if self.parentNetwork.optimizer == OPTIMIZER_SGD :
                # add Mometum (velocity)
                if KNET_ORIG : self.Bias_Momentum = friction * self.Bias_Momentum - (eta*self.W_bias_grad) # KNet Orig version 
                else :         self.Bias_Momentum = friction * self.Bias_Momentum + (eta*self.W_bias_grad) # KNet NEW version 
                
                self.Weights_bias += self.Bias_Momentum  # bias gradients are NOT regularized, so we just apply them as is
                
            else : # ADAM or AMSGrad
                # ADAM: RMSprop with MOMENTUM
                if KNET_ORIG : self.Bias_Momentum = friction * self.Bias_Momentum - (1.0 - friction) * self.W_bias_grad
                else :         self.Bias_Momentum = friction * self.Bias_Momentum + (1.0 - friction) * self.W_bias_grad
                
                # AMSGRAD 
                if self.parentNetwork.optimizer == OPTIMIZER_AMSGRAD :
                    if KNET_ORIG :
                        Past_Grads_bias_new = gamma * self.Past_Grads_bias - (1.0 - gamma) * self.W_bias_grad**2 
                        self.Past_Grads_bias = np.minimum(self.Past_Grads_bias, Past_Grads_bias_new)
                    else :
                        Past_Grads_bias_new = gamma * self.Past_Grads_bias + (1.0 - gamma) * self.W_bias_grad**2 
                        self.Past_Grads_bias = np.maximum(self.Past_Grads_bias, Past_Grads_bias_new)
 
                    Past_Grads_bias_new = None
                else :
                    if KNET_ORIG : self.Past_Grads_bias = gamma * self.Past_Grads_bias - (1.0 - gamma) * self.W_bias_grad**2  
                    else :         self.Past_Grads_bias = gamma * self.Past_Grads_bias + (1.0 - gamma) * self.W_bias_grad**2  
                
                
                # upward offset the rates during the first few epochs, this will converge to Momentum/1
                Momentum_biasOffset = self.Bias_Momentum / (1.0 - friction**(epoch_num+1)) # +1 as we dont want a 1-1=0 division by zero situ
                RMSProp_biasOffset = self.Past_Grads_bias / (1.0 - gamma**(epoch_num+1))   # we take it to the power of the current epoch, IE this will tend to 0, so in later iterations 1-0 = 1
               
                # finally update the weight: this is Momentum, normalised by RMSProp
                self.Weights_bias += ( eta * Momentum_biasOffset / (np.sqrt(RMSProp_biasOffset) + EPSILON) ) # bias gradients are NOT regularized, so we just apply them as is
                

        
    # updates the weights (including intercept) for current layer by calculating a 'change in weights': this is basically a scaled version of Error*Error_current
    # we scale by: learning rate(eta) and number of samples in minibatch ( to ensure consistent updates between different minibatch sizes)
    # this is the stage where we add regularizers too: we scale those by the 'total number of samples in ALL minibatches' (kindof redundant)
    # finally entire update is applied via 'momentums' we build up acceleration from previous updates, so if we keep moving int othe same direction then we can go(IE learn) faster and faster
    def update(self, eta, friction, gamma, epoch_num) :
        if(self.Weights_W is not None) : # if it has any outgoing weights (IE not an input that feeds into a splitter)
       
            global OPTIMIZER_SGD
            global OPTIMIZER_AMSGRAD 
            global EPSILON
            
            if self.parentNetwork.optimizer == OPTIMIZER_SGD :
              # add Mometum (velocity)
                if KNET_ORIG : self.Momentum = friction * self.Momentum - (eta*self.W_grad) # Knet Orig version # total update is: (dampened) pastUpdates + current update 
                else :         self.Momentum = friction * self.Momentum + (eta*self.W_grad) # Knet NEW version # total update is: (dampened) pastUpdates + current update 
                
                self.Weights_W += self.Momentum 
            
            else : # ADAM or AMSGrad
                # ADAM: RMSprop with MOMENTUM
                if KNET_ORIG : self.Momentum = friction * self.Momentum - (1.0 - friction) * self.W_grad   # CHECK
                else :         self.Momentum = friction * self.Momentum + (1.0 - friction) * self.W_grad   # CHECK
            
                
                # AMSGRAD
                if self.parentNetwork.optimizer == OPTIMIZER_AMSGRAD :
                    if KNET_ORIG :
                        Past_Grads_new = gamma * self.Past_Grads - (1.0 - gamma) * self.W_grad**2  # CHECK
                        self.Past_Grads = np.minimum(self.Past_Grads, Past_Grads_new) # not sure about this... 'minimum' are we going down??
                    else :
                        Past_Grads_new = gamma * self.Past_Grads + (1.0 - gamma) * self.W_grad**2  # CHECK
                        self.Past_Grads = np.maximum(self.Past_Grads, Past_Grads_new)
                    Past_Grads_new = None
                else :   
                    if KNET_ORIG : self.Past_Grads = gamma * self.Past_Grads - (1.0 - gamma) * self.W_grad**2  # CHECK
                    else :         self.Past_Grads = gamma * self.Past_Grads + (1.0 - gamma) * self.W_grad**2  # CHECK
                                                         
                # upward offset the rates during the first few epochs, this will converge to Momentum/1
                Momentum_biasOffset = self.Momentum / (1.0 - friction**(epoch_num+1)) # +1 as we dont want a 1-1=0 division by zero situ
                RMSProp_biasOffset = self.Past_Grads / (1.0 - gamma**(epoch_num+1))   # we take it to the power of the current epoch, IE this will tend to 0, so in later iterations 1-0 = 1
    
                            
                # finally update the weight: this is Momentum, normalised by RMSProp
                self.Weights_W += ( eta * Momentum_biasOffset / (np.sqrt(RMSProp_biasOffset) + EPSILON) ) # careful we need to now SUBTRACT the deltas
                   
                
            # update bias/intercept terms separately (if needed)
            self.update_bias_weights(eta,friction, gamma, epoch_num)  # bias weights are not trained ...for now (I think this has to do with the fact that NNs have universal approx ability, as long as the function doesn't go through 0)
            
       
    def getRegularizerGrad(self) : # no longer normalise by total number of sampels, as we assume that the L2 norm has been found by EMMA to be precisely correct for the given sample size
        #print("FUCKOFF getRegularizerGrad", flush=True )
        if ( self.regularizer == REGULARIZER_NONE ) :
            return(0.0)
        elif (self.regularizer == REGULARIZER_RIDGE ) :
            return ( -self.Lambda * self.Weights_W ) # this works for BOTH Lambda being a scalar, and Lambda being an array
        elif (self.regularizer == REGULARIZER_LASSO ): # LASSO
            return ( -self.Lambda * np.sign(self.Weights_W ) )   # this will cause the numerical check to fail.. it will still be small but much bigger than with L2 or without, this is probably due to the funky L1 derivative at 0
        elif (self.regularizer == REGULARIZER_ORTHO ): # ORTHAGONAL regularizer: Sum |WW^T -I|  
            W = self.Weights_W
            #global ORHTO_TRANSPOSE
            #if ORHTO_TRANSPOSE : W = W.T
            #WWT = np.dot( W,  W.T) # I COULD cache this in the FP 
            #I = np.eye(WWT.shape[0])
            #D_C = ( ( (self.Lambda * 0.5) * np.dot(np.sign(WWT -I),W) ) + ( (self.Lambda * 0.5) * np.dot(np.sign((WWT + -I.T)), W) ) )
           # if ORHTO_TRANSPOSE : D_C = D_C.T # add an extra transpose

            WTW = np.dot( W.T,  W) # I COULD cache this in the FP 
            I = np.eye(WTW.shape[0])
            D_C = ( ( (self.Lambda * 0.5) * np.dot(W,np.sign(WTW -I)) ) + ( (self.Lambda * 0.5) * np.dot(W, np.sign((WTW + -I))) ) ) #D_C.shape  # Input x Neuron, IE we got back to what it should look like
            return (  -D_C  ) 
        elif self.regularizer == REGULARIZER_ORTHO2  : # ORTHO v2 # Assumes matrix of Input x Neurons, IE neurons on columns, so this won't work for a CNN layer (will need to )
            W = self.Weights_W
            norms = np.linalg.norm(W, axis=0) # dont reshape this as we will use this to divide each column, to norma.reshape(1,-1)
            W = W / norms # want to normalize each neuron ( this should not alter the original Weight matrix)
            WTW = np.dot( W.T,  W) # I COULD cache this in the FP 
            ##D_C = W @ ( WTW - np.diag(WTW) )   # WARNING: np.diag(MATRIX) will create a 1D matrix and subtracting that will fuck up everything as instead of subtracting just from the diagonal it subtracts it from all elements :S
            np.fill_diagonal(WTW,0) # this is equivalent to WTW - diagonalMatrixOfWTW
            D_C = np.dot(W , WTW ) 
            return ( (self.Lambda * 0.5) * -D_C  ) # minus sign, as otherwise this would MAXIMISE the similarity between neurons ??? why are signs flipped? 
        
        elif self.regularizer == REGULARIZER_ORTHO_LASSO :
            W = self.Weights_W
            WTW = np.dot( W.T,  W) # I COULD cache this in the FP 
            I = np.eye(WTW.shape[0])
            D_C = ( ( (self.Lambda[0] * 0.5) * np.dot(W,np.sign(WTW -I)) ) + ( (self.Lambda[0] * 0.5) * np.dot(W, np.sign((WTW + -I))) ) ) #D_C.shape  # Input x Neuron, IE we got back to what it should look like
            
            lasso_loss = self.Lambda[1] * self.Weights_W
            total_loss = D_C + lasso_loss
            return (  -total_loss  ) 
        
        elif self.regularizer == REGULARIZER_ORTHO2_LASSO :
            W = self.Weights_W
            norms = np.linalg.norm(W, axis=0) # dont reshape this as we will use this to divide each column, to norma.reshape(1,-1)
            W = W / norms # want to normalize each neuron ( this should not alter the original Weight matrix)
            WTW = np.dot( W.T,  W) # I COULD cache this in the FP 
            np.fill_diagonal(WTW,0) # this is equivalent to WTW - diagonalMatrixOfWTW
            D_C = np.dot(W , WTW ) 
            D_C = self.Lambda[0] * 0.5 * D_C   

            lasso_loss = self.Lambda[1] * self.Weights_W
            total_loss = D_C + lasso_loss
            return (  -total_loss  ) 
        
        
        else : return(0.)
       
    # the cost of the regularizer term: only used for  gradient checking
    def getRegularizerCostTerm(self) : # no longer normalise by total number of sampels, as we assume that the L2 norm has been found by EMMA to be precisely correct for the given sample size
        #print("FUCKOFF", flush=True )
        if (self.regularizer == REGULARIZER_NONE) :
            return(0.0) 
        elif (self.regularizer == REGULARIZER_RIDGE) :
            
            if type(self.Lambda) is np.ndarray : # UNTESTED CODE: if Lambda is a vector, we need to split weights and sum for each Lambda separately
                #print("getRegularizerCostTerm is CALLED")
                regularizerCostTerm = 0
                for i in range( len(self.Lambda) ) : # loop instead of sum
                    regularizerCostTerm += ( (self.Lambda[i,0] * 0.5) *  np.sum( self.Weights_W[i,:]**2 ) ) # Predictors are stored in the rows for  the Weight mat   
                return(regularizerCostTerm)
                
            else : return ( (self.Lambda * 0.5) * np.sum( self.getWeightsAsVector()**2 ) )
            
            
        elif (self.regularizer == REGULARIZER_LASSO ): # LASSO
            if type(self.Lambda) is np.ndarray :
                regularizerCostTerm = 0
                for i in range( len(self.Lambda) ) : # loop instead of sum
                    regularizerCostTerm += ( (self.Lambda[i,0] * 0.5) *  np.sum(  np.abs( self.Weights_W[i,:] ) )   )
                return(regularizerCostTerm)
            
            else : return ( (self.Lambda * 0.5) * np.sum( np.abs( self.getWeightsAsVector() ) ) ) 
        elif (self.regularizer == REGULARIZER_ORTHO ): # ORTHAGONAL regularizer: Sum |WW^T -I|  
            # I think Weights needs to be transposed, as literature is referring to CNNs not MLPs and in CNNs neurons/filters are stored in eahc row, as WW^t for a W of BigxSmall * SmallxBig = BigxBig ... where as we want a dot product, also InputxNeurons x NeuronsxInput = InputxInput, but we want how the neurons relate to each other IE NeuronsxNeurons (and SmallxSmall)
            W = self.Weights_W
            #global ORHTO_TRANSPOSE
           #if ORHTO_TRANSPOSE : W = W.T
            #WWT = np.dot( W,  W.T)
            WTW = np.dot( W.T,  W)
            C = (  self.Lambda * 0.5) * np.sum(np.abs(WTW - np.eye(WTW.shape[0])) )
            #print("cos1t: ", round(C,2), flush=True )
            return  (C)
       
        
        else : # ORTHO v2 # this is basically the sum of squares of the off diagonal elements of the W^TW matrix, IE the more similar the neurons are the higher the penalty (Again only works for MLP not for CNNs)
            W = self.Weights_W
            #global ORHTO_TRANSPOSE
            #if ORHTO_TRANSPOSE : W = W.T
            # sum (cos2(all pairs of neurons)) == sum ( (dot(Ni,Nj) / (normNi * normNj) **2 ) )
            dotproducts = np.dot(W.T,W) # the lower triangle (excluding diagonal) is the dot products between all neurons
            norms = np.linalg.norm(W, axis=0).reshape(1,-1)
            cosinesimilarities = dotproducts / norms / norms.T
            C = np.sum( np.tril(cosinesimilarities, k =-1)**2 ) # extract the lower triangle, square them, and sum
            #print("cost2: ", round(C,2), flush=True )
            return( (self.Lambda * 0.5) * C)

        
    # returns the weights in in the current layer concated into a 1D array (used oly by getRegularizerCostTerm()  )
    def getWeightsAsVector(self) :
        return( self.Weights_W.ravel() ) # this unravels by ROWS, ie it will have all of row1, thenall of row2 ... then rown
 
       
       
    # concats the weights of this layer (if it has any), into the ones we got so far
    def getDebugInfo(self, dataSoFar, Type = WEIGHT) :
        if(self.Weights_W is not None) :
            if(Type == WEIGHT) :  #if we are NOT looking for weight gradients, then we want the weights
                return(  np.concatenate( (dataSoFar, self.getWeightsAsVector()) ) )
            elif (Type == GRAD) : # if we do want the gradients then we use the ones saved during debug
                return(  np.concatenate( (dataSoFar, self.W_grad.ravel() ) ) )
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
            


# Convolutional layer
class knnConvLayer(knnLayer):
    def __init__(self, iparentNetwork, iparams, isubtype, ibiasEnabled = True, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0, p_dropout = -1, n_filters = 10, h_filter=3, w_filter=3, padding=1, stride=1, oneD = False):
        super().__init__(iparentNetwork,iparams ,isubtype, ibiasEnabled, regularizer, shrinkageParam, p_dropout) # call init on parent
        self.filter_height = h_filter
        self.filter_width = w_filter
        self.padding = padding
        self.stride = stride
        self.num_filters = n_filters # how many filters are used in this layer, this is the main control for how much this layer can learn
        self.last_input_shape = None #  keep the original shape of the input before stretchin, used for back propagation
        self.oneD = oneD
        self.p_w = padding
        self.p_h = padding
        self.stride_h = stride 
        self.stride_w = stride
        if self.oneD : 
            self.p_h = 0
            self.stride_h = 1
        
        # debug vars
        self.firstRun = True
        #self.lastMiniBatchSize = -1 # last minibatch size: used to cache the indices for the im2col operations      
        #self.cachedIndices = None
        if self.parentNetwork.suppressPrint == False : print( "Conv layer "+self.subtype + "(dropout: "+ str(self.p_dropout) +") / w_filter/h_filter: " + str(w_filter) + " / " + str(h_filter) + " / padding: " + str(padding) + " / stride: " + str(stride) +" is regularized by: " + self.regularizer+ " / its params are ", str(self.params)  )
     

    def initWeightMatrix(self, Input_shape) :
        
        
        prevLayer_size = Input_shape[1] * Input_shape[2] * Input_shape[3]# [n,d,height,width], so the number of predictors is height*width, * d (the channels)
        #print("initting weight matrix for Conv layer, prevLayer_size: " +str(prevLayer_size), flush=True)
        self.Weights_W = truncatedGaussian([ int(self.num_filters),Input_shape[1],self.filter_height, self.filter_width], np.sqrt(2.0 / prevLayer_size ))  # HE init with truncated normal ( SD = sqrt(2/Fan_in) )
        # store this into a 2D column format: as it is never used in any other way
        self.Weights_W = self.Weights_W.reshape(self.num_filters, -1)  # turn this into a 10x9 matrix: IE 10 filters, each of them is 9 for the 3x3 patches.... This is the bit when the parameter sharing comes in: We only learn 10 things out of the 784 inputs
        #print("prevLayer_size is: " + str(prevLayer_size) + " / self.Weights_W.shape: " + str(self.Weights_W.shape) + " Input_shape: " + str(Input_shape))
    
        self.Momentum = np.zeros((self.Weights_W.shape[0], self.Weights_W.shape[1]), dtype=self.Weights_W.dtype) # stores a 'dampened' version of past weights (IE its an older version of the above with the same dimensions)
        self.Past_Grads = np.zeros((self.Weights_W.shape[0], self.Weights_W.shape[1]), dtype=self.Weights_W.dtype) # stores a 'dampened' version of past weights, Squared, used by RMSProp
           
        if self.biasEnabled :  # if we have bias/intercept, we add a row of Weights that we keep separate
            self.Weights_bias = np.zeros( (self.Weights_W.shape[0],1) , dtype=self.Weights_W.dtype)  # HE biases as inited as zero
            self.Bias_Momentum = np.zeros( (self.Weights_bias.shape[0],1), dtype=self.Weights_W.dtype ) # stores a 'dampened' version of past weights  for intercepts
            self.Past_Grads_bias = np.zeros( (self.Weights_bias.shape[0],1), dtype=self.Weights_W.dtype ) # stores a 'dampened' version of past weights  for intercepts, Squared, used by RMSProp


    # override for Conv layers as output is not XB, but (BX)
    def generateOutput(self, Input, train = True, saveInput = True) :
        #num_samples_minibatch = len(Input) # find out the length of the minibatches: this is used to determine if we can used a cached version of im2col for faster performance
        num_samples_minibatch, input_channels, input_height, input_width = Input.shape# 1, 1 28, 28, IE batch of length 1, of 28x28 size ( 1 channel) , this could be cached , for when the MB size is the same
        self.last_input_shape = Input.shape # save this for back prop
        if self.Weights_W is None : self.initWeightMatrix(Input.shape ) #Input.shape[1] * Input.shape[2] * Input.shape[3]) # lazy instantiation: need to 
    
        # I) reshaping the input: generating a single matrix,  where each column is a patch, of a given size, that holds all the possible patches in a single array for efficient GEMMs
        self.Input_S = knetim2col.im2col_indices(Input, self.filter_height, self.filter_width, p_h=self.p_h, p_w=self.p_w, stride_h=self.stride_h, stride_w=self.stride_w) # patchsize, numPatches(784) (* number of images), IE stretch the input into the patches, 784 3x3 patches for each 28x28 image
        #self.Input_S  = Input_asCols # save Input in column-stretched format for later access
        #if self.firstRun : self.firstRun = False ; print("Input Space expansion factor is:" , str( np.round( (np.prod(self.Input_S.shape) / np.prod(Input.shape) ) * 100 , 2) / 100 ) , "( input went from " , str( np.prod(Input.shape) ), " to ", str(np.prod(self.Input_S.shape) ) , ")" )
        
        # II) the main bit: WX
        Output_Z = self.Weights_W.dot(self.Input_S)  # This is different for Conv layers, instead of the usual Input*Weights, we have to do Weights * Input: as usually this is Input * Weights : 9x784 *10x9 = NOTGOOD, we have to do 10x9 * 9x784
        Output_Z = self.add_bias(Output_Z) # we then add the intercept (the linear predictor is now complete) 
       
        # III) reshape back to original dimensions
        output_height = (input_height - self.filter_height + 2 * self.p_h) / self.stride_h + 1 # 28 # this could only be cached for the 1st layer, as subsequent deeper layers may not be 28, depending on filter size / padding
        output_width = (input_width - self.filter_width + 2 * self.p_w) / self.stride_w + 1 # 28
        Output_Z = Output_Z.reshape(self.num_filters, int(output_height), int(output_width), num_samples_minibatch) # turns this into 10,28,28,1
        Output_Z = Output_Z.transpose(3, 0, 1, 2) # 1,10,28,28: 1 image, 10 filters, 28x28 sized

        #print("Conv output shape is:" , Output_Z.shape)
        # non Output subtype layers need to figure out the rate of change in the output (this is redundant, if we are just making predictions, IE backpropagation does NOT follow this)
        #if (self.subtype != LAYER_SUBTYPE_OUTPUT) :  self.Derivative_Fp = self.activation(self.Output_Z, deriv=True)  # this is transposed, as D is also transposed during back propagation
        # also more efficient to compute these on the smaller image rather than the stretched out version
        #print("Conv self.Derivative_Fp shape is:" , self.Derivative_Fp.shape)
        # output is completed by passing the linear predictor through an activation (IE we squash it through a sigmoid)
        #self.Output_Z = self.activation(self.Output_Z) # if its a hidden layer (or output), we need to activate it
        
        Output_Z = self.dropout_forward(Output_Z, train)
        
        if saveInput is False: self.Input_S = None # release memory, if we do't have a backprop (IE when only assesing accuracy)
        #self.parentNetwork.checkNetworkDatatype()
        return ( Output_Z )
        
 

    def calcGrads_and_Errors(self, Error_current, computeErrors = True, inference = False) :  # Error_current, us the Error_D^T of THIS layer, just not yet anti activated  
        #print("Conv layer calcGrads_and_Errors")
        # Error_current_shape = Error_current.shape
        # we expect the Unactivated Errors to be received in non-stretched format
       # Error_current = Error_current.T # transpose it back..., Convnet error calculation works on non-tranposed shapes
        
        #print("Conv backprop received Input Error shape: ", str(Error_current.shape), " / last_input_shape: ", str(self.last_input_shape))
            
        #print("Error_current shape is: " , Error_current.shape)
        #print("self.Derivative_Fp shape is: " , self.Derivative_Fp.shape)
        #Error_current = Error_current * self.Derivative_Fp # as the derivatives were acquired on the non-stretched version, so we need to apply it before stretching it again (need to save away the unstretched Errors for the bias computation )
        self.Error_D = Error_current
        self.dropout_backward(inference)
        self.Error_D = Error_current.transpose(1, 2, 3, 0).reshape(self.num_filters, -1) # 10x784: the Error reshaped
        
        
        #print("Conv layer self.Error_D is: " + str(self.Error_D))
        # compute gradients
        num_samples_minibatch = self.last_input_shape[0] 
        #print("num_samples_minibatch in Conv layer is: " , str(num_samples_minibatch))
        

        
        self.W_grad = self.Error_D.dot(self.Input_S.T) # 10x784 * 784x9 = 10x9, # Error_D * Input, this normally belongs to the Update function for me
        self.W_grad = self.W_grad / num_samples_minibatch # normalise by N, to keep the same as the cost function    
        self.W_grad = self.W_grad + self.getRegularizerGrad() # I am applying the regularizers AFTEr normalizing for minibatch size... (as I am assuming that the L2 norms come externally, and determined to be just the right size)
        #self.W_grad = self.W_grad.reshape(W1.shape) # 10,1,3,3 (numFilters, numChannels, patchsize_H, patchSize_W)
        # as I store Weights in column-stretched format I don't actually reshape this back  either  
        
        self.Input_S = None # no longer need this so release memory

    
        if(self.biasEnabled == True) :
            self.W_bias_grad = np.sum(Error_current, axis=(0, 2, 3)) # Error already has the derivatives applied to it, # sum all the errors for all items, and for their 28x28 arrays
            self.W_bias_grad = self.W_bias_grad.reshape(self.num_filters, -1) # turn it into a column vector, I think this is OK as we store Biases in the column format only
            self.W_bias_grad /= num_samples_minibatch 
            
            
        # If error computation was requested, we return that
        if computeErrors :
            # reuse the Error_current variable to save memory to now store the next layers error
            Error_current = self.Weights_W.T.dot(self.Error_D) # This is equivalent to KNet's: calcGrads_and_Errors():self.Error_D = weightToUse.dot(Error_current)   -  9x10 * 10,784 = 9x 784, this is the Error_D for the NEXT layer
            if inference == False : self.Error_D = None # if we are NOT in inference mode, IE we dont want to save the Errors, we can release this
            #print("CONV knetim2col, Error_current dtype is: " +str(Error_current.dtype), flush=True)
            Error_current = knetim2col.col2im_indices(Error_current, self.last_input_shape, self.filter_height, self.filter_width, p_h=self.p_h, p_w=self.p_w, stride_h=self.stride_h, stride_w=self.stride_w)         #so basically we are multiplying the Stretched Weights, with the stretched Delta of the current layer, to stretch it back into the original dimensions
            #print("Conv backprop received Input Error shape: ", str(Error_current_shape), " / Error_new shape: ", str(Error_current.shape) , " / last_input_shape: ", str(self.last_input_shape))
            
            
        
            return(Error_current) # other layers expect Errors to be transposed
        else : 
            #print("Last hidden (Conv) layer before input, no need to compute Errors")
            if inference == False : self.Error_D = None # if we are NOT in inference mode, IE we dont want to save the Errors, we can release this
            return (None) # if not (like if previous layer is the Input), then Errors are not needed
        # (this is because Dprev = W_current*D_current), IE, if this is Hidden Layer 1, then we have already computed the gradients for this above, and it is pointless to compute another large error matrix that will never be used



class knnMaxPool(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1, size = 2, padding=0, stride=2, oneD = False ):
        super().__init__(iparentNetwork, iparams) # call init on parent     

        self.padding = padding
        self.stride = stride
        self.size = size # how many filters are used in this layer, this is the main control for how much this layer can learn
        self.last_input_shape = None #  keep the original shape of the input before stretchin, used for back propagation
        self.InputMaxVal_indices = None # stores the indices of the max values in the Input ( used for back prop)
        self.oneD = oneD
        self.p_w = padding
        self.p_h = padding
        self.size_h = size
        self.size_w = size
        self.stride_h=stride 
        self.stride_w=stride
        self.supportLayerType = True
        
        if self.oneD : 
            self.p_h = 0
            self.size_h = 1
            self.stride_h = 1 
        #self.lastMiniBatchSize = -1 # last minibatch size: used to cache the indices for the im2col operations      
        #self.cachedIndices = None
        if self.parentNetwork.suppressPrint == False : print( "Maxpool layer / size: " +  str(size) + " / padding: " + str(padding) + " / stride: " + str(stride) + " / its params are ", str(self.params) )
     
  
    # override for Conv layers as output is not XB, but (BX)
    def generateOutput(self, Input, train = True, saveInput = True) :
        # Let say our input X is 5x10x28x28
        # Our pooling parameter are: self.size = 2x2, stride = 2, padding = 0
        # i.e. result of 10 filters of 3x3 applied to 5 imgs of 28x28 with stride = 1 and padding = 1
        num_samples_minibatch, num_filters, input_height, input_width = Input.shape # numer of samples, filters Image Height/Width
        self.last_input_shape = Input.shape #[::-1]  # save this for back prop

        # reshape it into the same variable to save memory
        self.Input_S = Input.reshape(num_samples_minibatch * num_filters, 1, input_height, input_width) #  reshape it to 50x1x28x28 to make im2col arranges it fully in column ( IE 5 * 10 = 50)
        self.Input_S = knetim2col.im2col_indices(self.Input_S, self.size_h, self.size_w, p_h=self.p_h, p_w=self.p_w, stride_h=self.stride_h, stride_w=self.stride_w) #  4x9800
        #self.Input_S  = Input_asCols # save Input in column-stretched format for later access
        
        # out, pool_cache = pool_fun(Input_asCols)
        self.InputMaxVal_indices = np.argmax(self.Input_S, axis=0) # this is a 1D array: find at each possible patch location, i.e. at each column, we're taking the max index
        
        
        #print("self.InputMaxVal_indices TYPE is : " + str(self.InputMaxVal_indices.dtype))
        #print("self.InputMaxVal_indices is : " + str(self.InputMaxVal_indices))
        #self.InputMaxVal_indices = self.InputMaxVal_indices.astype(int)
        #print("AFTER self.InputMaxVal_indices TYPE is : " + str(self.InputMaxVal_indices.dtype))
        #print("AFTER self.InputMaxVal_indices is : " + str(self.InputMaxVal_indices))
        
        #testArray = np.asarray([0,1,2])
        #myRange = range(self.InputMaxVal_indices.size)
        #print("myRange is: "  + str(myRange))
        #myRange = myRange.astype(int)
        #Output_Z = self.Input_S[self.InputMaxVal_indices, 0]
        #print("this still worked")
        Output_Z = self.Input_S[self.InputMaxVal_indices, np.arange(self.InputMaxVal_indices.size)] # 1x9800 : get all the max value at each column
        
        
        # Reshape to the output size: 14x14x5x10
        output_height = (input_height - self.size_h) / self.stride_h + 1 #, this is 14, for an image of 28 with stride 2 and padding 0 ( IE we just downsample it) get output dimensions
        output_width = (input_width - self.size_w) / self.stride_w + 1
        Output_Z = Output_Z.reshape(int(output_height), int(output_width), num_samples_minibatch, num_filters)
        Output_Z = Output_Z.transpose(2, 3, 0, 1) # reorganis the axes: Transpose to get 5x10x14x14 output
    
        if saveInput is False: self.Input_S = None # release memory, if we do't have a backprop (IE when only assesing accuracy)
        #self.parentNetwork.checkNetworkDatatype()
        return ( Output_Z )
        
 

    def calcGrads_and_Errors(self, Error_current, computeErrors = True, inference = False) :  # Error_current, us the Error_D^T of THIS layer, just not yet anti activated  

        # compute gradients
        num_samples_minibatch, num_filters, input_height,input_width  = self.last_input_shape # [::-1] 
        #Error_current_shape = Error_current.shape
        #print("Maxpool Backprop")
        # 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
        # Transpose step is necessary to get the correct arrangement
        #Error_current = Error_current.T
        self.Error_D = Error_current.transpose(2, 3, 0, 1).ravel() # need to flatten it as the max_idx is also a 1D array
        #print("Maxpool layer self.Error_D is: " + str(self.Error_D))
        
        if computeErrors :
            # Fill the maximum index of each column with the gradient
            # Essentially putting each of the 9800 grads
            # to one of the 4 row in 9800 locations, one at each column
            Error_current = np.zeros(self.Input_S.shape, dtype=self.Error_D.dtype) # this is much faster than zeros_like
            Error_current[self.InputMaxVal_indices, np.arange(self.Error_D.size)] = self.Error_D
            self.Input_S = None # no longer need this so release memory
            self.InputMaxVal_indices = None
            if inference == False : self.Error_D = None # if we are NOT in inference mode, IE we dont want to save the Errors, we can release this
            
            # We now have the stretched matrix of 4x9800, then undo it with col2im operation
            #print("Maxpool knetim2col, Error_current dtype is: " +str(Error_current.dtype), flush=True)
            Error_current = knetim2col.col2im_indices(Error_current, (num_samples_minibatch * num_filters, 1, input_height, input_width), self.size_h, self.size_w, p_h=self.p_h, p_w=self.p_w, stride_h=self.stride_h, stride_w=self.stride_w) # 50x1x28x28
            Error_current = Error_current.reshape(self.last_input_shape) # Reshape back to match the input dimension: 5x10x28x28
            #print("Maxpool backprop received Input Error shape: ", str(Error_current_shape), " / Error_new shape: ", str(Error_current.shape) , " / last_input_shape: ", str(self.last_input_shape))

            return Error_current
        else : 
            self.Input_S = None # no longer need this so release memory
            self.InputMaxVal_indices = None
            if inference == False : self.Error_D = None # if we are NOT in inference mode, IE we dont want to save the Errors, we can release this
            return (None) # if not (like if previous layer is the Input), then Errors are not needed
        # (this is because Dprev = W_current*D_current), IE, if this is Hidden Layer 1, then we have already computed the gradients for this above, and it is pointless to compute another large error matrix that will never be used
        
        
        
        
        
        
        
# https://github.com/chainer/chainer/blob/v5.0.0/chainer/functions/loss/decov.py#L58
class knnDeCov(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1, half_squared_sum = True):
        super().__init__(iparentNetwork, iparams) # call init on parent   
        
        self.h_centered = None
        self.covariance = None
        self.half_squared_sum = half_squared_sum
        
        if self.parentNetwork.suppressPrint == False : print( "Decov layer, half_squared_sum: ", half_squared_sum)
     
  
    # override for Conv layers as output is not XB, but (BX)
    def generateOutput(self, Input, train = True, saveInput = True) :
        self.Input_S = Input

        self.h_centered = self.Input_S - self.Input_S.mean(axis=0, keepdims=True)
        self.covariance = self.h_centered.T.dot(self.h_centered) # this is the 
        np.fill_diagonal(self.covariance, 0.0) # this is the C - diag(C) operation
        self.covariance /= len(self.Input_S) # normalize by the length of the minibatch

        if self.half_squared_sum :
            # apply the sum square operation
            cost = np.vdot(self.covariance, self.covariance)
            cost *= self.Input_S.dtype.type(0.5)
        else : 
            cost = self.covariance
            
            print("cost shape is: " , cost.shape, " / and  self.covariance shape: ",  self.covariance.shape)
        
        if saveInput is False: self.Input_S = None # release memory, if we do't have a backprop (IE when only assesing accuracy)

        return(cost)


    def calcGrads_and_Errors(self, Error_current, computeErrors = True, inference = False) :  # Error_current, us the Error_D^T of THIS layer, just not yet anti activated  
        self.Error_D = Error_current
        
        if computeErrors :  
            gcost_div_n = Error_current / Error_current.dtype.type(len(self.Input_S))  # normalize by the length of the minibatch
            
            if self.half_squared_sum :
                # apply derivative of the sum squared operation
                Error_current = 2.0 * self.h_centered.dot(self.covariance)
                Error_current *= gcost_div_n
            else :
                np.fill_diagonal(gcost_div_n, 0.0)
                Error_current = self.h_centered.dot(gcost_div_n + gcost_div_n.T)
            
            # release memory
            self.h_centered = None
            self.covariance = None
            
            return Error_current
        else : 
            self.Input_S = None # no longer need this so release memory
            if inference == False : self.Error_D = None # if we are NOT in inference mode, IE we dont want to save the Errors, we can release this
            return (None) # if not (like if previous layer is the Input), then Errors are not needed

   
        
class knnBatchNorm(knnLayer):
    def __init__(self, iparentNetwork, iparams, isubtype, ibiasEnabled = True, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0, running_mean_momentum = 0.1):  # , running_mean_momentum = 0.9):
        super().__init__(iparentNetwork,iparams ,isubtype, ibiasEnabled, regularizer, shrinkageParam) # call init on parent  
        self.running_mean_momentum = running_mean_momentum #  keep the original shape of the input before stretchin, used for back propagation
        self.running_mean = 0
        self.running_var = 0
        self.Input_norm  = None
        self.mu = 0
        self.var = 0
        self.supportLayerType = True
        
        
        #self.slope_grad = None
        #self.bias_grad = None
        # Weights_W = slope of regression
       # Weights_bias = intercept of regression  

        # figure out how to init  (what is H ???)
        # figure out how to update with a common interface

        if self.parentNetwork.suppressPrint == False : print( "BatchNorm / its params are ", str(self.params) )
        

    
        
    def initWeightMatrix(self, prevLayer_size) :
        global NETWORK_DATATYPE
        self.Weights_W=np.ones((1, prevLayer_size), dtype=NETWORK_DATATYPE) # the Slope of the linear regression
        self.Weights_bias=np.zeros((1, prevLayer_size), dtype=self.Weights_W.dtype) # the Intercept of the linear regression  

        self.Momentum = np.zeros((self.Weights_W.shape[0], self.Weights_W.shape[1]), dtype=self.Weights_W.dtype) # stores a 'dampened' version of past weights (IE its an older version of the above with the same dimensions)
        self.Past_Grads = np.zeros((self.Weights_W.shape[0], self.Weights_W.shape[1]), dtype=self.Weights_W.dtype) # stores a 'dampened' version of past weights, Squared, used by RMSProp

        self.Bias_Momentum = np.zeros( (1,self.Weights_bias.shape[1]), dtype=self.Weights_W.dtype ) # stores a 'dampened' version of past weights  for intercepts
        self.Past_Grads_bias = np.zeros( (1,self.Weights_bias.shape[1]), dtype=self.Weights_W.dtype ) # stores a 'dampened' version of past weights  for intercepts, Squared, used by RMSProp
                 
        
        
    # override for Conv layers as output is not XB, but (BX)
    def generateOutput(self, Input, train = True, saveInput = True) :

        if self.Weights_W is None : self.initWeightMatrix(Input.shape[1]) # lazy instantiation
        
        self.Input_S  = Input
  
        if train: # if training, we compute the norms from current minibatch
            self.mu = np.mean(self.Input_S, axis=0) # compute per sample var/means
            self.var = np.var(self.Input_S, axis=0)
    
            self.Input_norm = (self.Input_S - self.mu) / np.sqrt(self.var + EPSILON)
            out = self.Weights_W * self.Input_norm + self.Weights_bias
    
            self.running_mean = self.exp_running_avg(self.running_mean, self.mu, self.running_mean_momentum)
            self.running_var = self.exp_running_avg(self.running_var, self.var, self.running_mean_momentum)
        else: # otherwise we just use the last one, without overwriting the cached variable
            self.Input_norm = (self.Input_S - self.running_mean) / np.sqrt(self.running_var + EPSILON) # we still have to save this otherwise BP wont work, as Error_D and Input_norm will have mismatched shapes
            out = self.Weights_W * self.Input_norm + self.Weights_bias
  
        if saveInput is False: 
            self.Input_S = None # release memory, if we do't have a backprop (IE when only assesing accuracy)
            self.Input_norm = None
        #self.parentNetwork.checkNetworkDatatype()
        return ( out )
        

    
    def calcGrads_and_Errors(self, Error_current, computeErrors = True, inference = False) :  # Error_current, us the Error_D^T of THIS layer, just not yet anti activated  
    
        num_samples_minibatch, D = self.Input_S.shape
    
        self.W_grad = np.sum(Error_current * self.Input_norm, axis=0)
        self.W_bias_grad = np.sum(Error_current, axis=0)
    
        global KNET_ORIG
        if KNET_ORIG : 
            self.W_bias_grad *= -1
            self.W_grad *= -1
    
        self.Input_S -= self.mu # reuse Input_S as Input_mu to save memory
        std_inv = 1. / np.sqrt(self.var + EPSILON)
        #print("Batchnorm calcGrads_and_Errors , BEFORE Error_current dtype is: " +str(Error_current.dtype), flush=True)
        
        #Error_current = Error_current * self.Weights_W # reuse Error_current for what was 'Error_D_norm' to save memory
        Error_current *= self.Weights_W # reuse Error_current for what was 'Error_D_norm' to save memory
        
        #print("Batchnorm calcGrads_and_Errors , AFTER Error_current dtype is: " +str(Error_current.dtype), flush=True)
        
        var_grad = np.sum(Error_current * self.Input_S, axis=0) * -.5 * std_inv**3
        mu_grad = np.sum(Error_current * -std_inv, axis=0) + var_grad * np.mean(-2. * self.Input_S, axis=0)
        #print("Batchnorm  std_inv dtype is: " +str(std_inv.dtype) + " //  var_grad dtype is: " +str(var_grad.dtype) + " // mu_grad dtype is: " +str(mu_grad.dtype), flush=True)
        
        #Error_current = (Error_current * std_inv) + (var_grad * 2 * self.Input_S / num_samples_minibatch) + (mu_grad / num_samples_minibatch)
        # the above does not ensure type safety, we have to use inp-place operations for that:
        Error_current *= std_inv
        Error_current += (var_grad * 2 * self.Input_S / num_samples_minibatch) 
        Error_current += (mu_grad / num_samples_minibatch)
        
        
        #print("Batchnorm calcGrads_and_Errors , FINAL Error_current dtype is: " +str(Error_current.dtype), flush=True)
        
        #print("BNorm layer self.Error_D is: " + str(Error_current))
        self.Input_S = None # release memory
        self.Input_norm = None
    
        return Error_current
    
    def exp_running_avg(self,running, new, momentum=.9):
            return momentum * running + (1. - momentum) * new

  
class knnSpatialBatchNorm(knnBatchNorm):
    def __init__(self, iparentNetwork, iparams, isubtype, ibiasEnabled = True, regularizer = REGULARIZER_NONE, shrinkageParam = 0.0, running_mean_momentum = 0.9):
        super().__init__(iparentNetwork,iparams ,isubtype, ibiasEnabled, regularizer, shrinkageParam, running_mean_momentum) # call init on parent  
        if self.parentNetwork.suppressPrint == False : print( "Spatial BatchNorm / its params are ", str(self.params) )
        

    # the Input.shape is N,C,H,W, and as superclass inits  Weights by:
    # self.initWeightMatrix(Input.shape[1])  # and here [1] is precisely C (the number of filters/channels)
    # def initWeightMatrix(self, prevLayer_size) :  # so this will init the correctly shaped matrices, as in spatial batch norm we should init them as gamma, beta = np.ones(C), np.zeros(C)
    
    
    # override for Conv layers as output is not XB, but (BX)
    def generateOutput(self, Input, train = True, saveInput = True) :
        N, C, H, W = Input.shape
        Input = Input.transpose(0, 2, 3, 1).reshape(-1, C) # flatten input
        out = super().generateOutput(Input,train) # compute output in flattened format
        out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2) # reshape it back into (N, H, W, C)
        
        #self.parentNetwork.checkNetworkDatatype()
        return ( out )
            
    def calcGrads_and_Errors(self, Error_current, computeErrors = True, inference = False) :  # Error_current, us the Error_D^T of THIS layer, just not yet anti activated  
        N, C, H, W = Error_current.shape
        
        Error_current = Error_current.transpose(0, 2, 3, 1).reshape(-1, C) # flatten errors
        #print("Spatial Batchnorm, BEFORE Error_current dtype is: " +str(Error_current.dtype), flush=True)
        
        Error_current = super().calcGrads_and_Errors(Error_current, computeErrors, inference) # compute errors in flattened format
        #print("Spatial Batchnorm, AFTER Error_current dtype is: " +str(Error_current.dtype), flush=True)
        
        Error_current = Error_current.reshape(N, H, W, C).transpose(0, 3, 1, 2) # reshape it back into (N, H, W, C)
        return (Error_current)


class knnFlatten(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1 ):
        super().__init__(iparentNetwork, iparams) # call init on parent     
        self.last_input_shape = None #  keep the original shape of the input before stretchin, used for back propagation
        self.supportLayerType = True
        #self.lastMiniBatchSize = -1 # last minibatch size: used to cache the indices for the im2col operations      
        #self.cachedIndices = None
        if self.parentNetwork.suppressPrint == False : print( "Flatten layer / its params are ", str(self.params) )
      
    # override for Conv layers as output is not XB, but (BX)
    def generateOutput(self, Input, train = True, saveInput = True) :

        self.last_input_shape = Input.shape # [::-1]  # this flips the order, IE 'transposes' the shape 
        #print("Flatten generateOutput shape: " , str(Input.ravel().reshape(Input.shape[0], -1).shape))
        return ( Input.ravel().reshape(Input.shape[0], -1) ) # the first axis always stores the number of training items, so this is OK
        
    # passes errors backwards from the output onto preceding layers
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this  
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            # reshape the input back to what the previous layer expects it to be
            #print("Flatten backprop received Input Error shape: " , str(Input.shape), " / last_input_shape: ", str(self.last_input_shape))
            
            return( self.prevLayer.backpropagate( Input.ravel().reshape(self.last_input_shape), inference, guidedBackprop  ) ) 
    


class knSELU(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1 ):
        super().__init__(iparentNetwork,iparams ) # call init on parent     
        self.Input_S = None #  keep the original linear predictor for Backprop
        if self.parentNetwork.suppressPrint == False : print( "SELU layer")
        global SELU_LAM
        global SELU_ALPHA
        self.alpha = SELU_LAM#float(alpha)
        self.lam = SELU_ALPHA #float(lam)
        self.activationType = True
            
    def generateOutput(self, Input, train = True, saveInput = True):
        if saveInput : self.Input_S = Input 
        
        #x = Input.copy()
        #alpha = 1.6732632423543772848170429916717
        #lamb = 1.0507009873554804934193349852946
        #return lamb * np.where(x > 0., x, alpha * np.exp(x) - alpha)

        y = Input.copy()
        neg_indices = Input <= 0
        y[neg_indices] = self.alpha * (np.exp(y[neg_indices]) - 1)
        y *= self.lam
        return(y )

        
 
    
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this  
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            Error_D = Input.copy()
            neg_indices = self.Input_S <= 0
            
            Error_D[neg_indices] *= self.alpha * np.exp(self.Input_S[neg_indices])
            Error_D *= self.lam

            self.Input_S = None # release memory
            return( self.prevLayer.backpropagate( Error_D, inference, guidedBackprop  ) ) 



class knRELU(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1 ):
        super().__init__(iparentNetwork,iparams ) # call init on parent     
        self.Input_S = None #  keep the original linear predictor for Backprop
        self.debugVar = None
        self.activationType = True
        #self.lastMiniBatchSize = -1 # last minibatch size:
        # used to cache the indices for the im2col operations      
        #self.cachedIndices = None
        if self.parentNetwork.suppressPrint == False : print( "RELU layer")
      
    def generateOutput(self, Input, train = True, saveInput = True) :

        if saveInput : self.Input_S = Input #  
        #self.parentNetwork.checkNetworkDatatype()
        return (   np.maximum(Input,0) )
        
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this  
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            
          ## Knet New version 
            Error_D = Input.copy()  # why do we copy here??? why do we want to keep the original?? I suppose as that belongs to a different Layer... IE its another layer's self.Error_D that we dont want to modify
            #print("Error_D shape is: " , str(Error_D.shape) )
            #print("Input_S shape is: " , str(self.Input_S.shape) )
            #self.debugVar = Error_D
            Error_D[self.Input_S <= 0] = 0 # this is transposed, as D is also transposed during back propagation
            #print("Relu layer self.Error_D is: " + str(Error_D))
            
            
            # KNet Org version: we simply pass the error back, impliciy multiplication by 1s
            #Error_D = Input.copy() # np.ones(self.Input_S.shape)  

            
            
            # Inference: Guided Backpropagation
            #Error_D = np.where(self.Input_S > 0, Input, 0) 
              #dx_orig = dx.copy() # save orig
            if(guidedBackprop == True) : # if we are doing guided Backprop
                #print("guided backrpop")
                Error_D[np.logical_or(Error_D <= 0.0,  self.Input_S <= 0.0)] = 0.0 # then we set the gradient to 0, wherever it was non positive, for either the forward prop (x the input), or the derivative coming backwards from deeper layers
            #print("Relu ( After GP) layer self.Error_D is: " + str(Error_D))
            self.Input_S = None # release memory
            return( self.prevLayer.backpropagate( Error_D, inference, guidedBackprop  ) ) 



class knLeakyRELU(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = 0.001 ):
        super().__init__(iparentNetwork, iparams) # call init on parent     
        self.Input_S = None #  keep the original linear predictor for Backprop
        self.alpha = iparams
        self.activationType = True
        #self.lastMiniBatchSize = -1 # last minibatch size: used to cache the indices for the im2col operations      
        #self.cachedIndices = None
        if self.parentNetwork.suppressPrint == False : print( "Leaky RELU layer")
      
    def generateOutput(self, Input, train = True, saveInput = True) :
        if saveInput : self.Input_S = Input   #  
        return ( np.maximum(self.alpha * Input, Input) )
        
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this  
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this

            global KNET_ORIG
            
            if KNET_ORIG == False:
            ## Knet New version
                Error_D = Input.copy()
                Error_D[self.Input_S < 0] *= self.alpha    # this is transposed, as D is also transposed during back propagation
                
            else :
            # KNet Org version: we simply pass the error back, impliciy multiplication by 1s
                Error_D = Input.copy() # np.ones(self.Input_S.shape)  

                
            
            # Inference: Guided Backpropagation
            #Error_D = np.where(self.Input_S > 0, Input, 0) 
              #dx_orig = dx.copy() # save orig
            if(guidedBackprop == True) : # if we are doing guided Backprop
                Error_D[np.logical_or(Error_D <= 0.0,  self.Input_S <= 0.0)] = 0.0 # then we set the gradient to 0, wherever it was non positive, for either the forward prop (x the input), or the derivative coming backwards from deeper layers

            
            self.Input_S = None # release memory
            return( self.prevLayer.backpropagate( Error_D, inference, guidedBackprop  ) ) 


class knSquare(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1 ):
        super().__init__(iparentNetwork,iparams ) # call init on parent 
        self.activationType = True
        self.Input_S = None #  keep the activated predictor around
        if self.parentNetwork.suppressPrint == False : print( "Square layer")
      
    def generateOutput(self, Input, train = True, saveInput = True) :
        Input =  Input**2 
        if saveInput : self.Input_S =Input
        return (  Input )
        
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this  
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            global KNET_ORIG
            if KNET_ORIG == False :  Error_D = 2 * self.Input_S  * Input 
            else : Error_D = Input.copy()
            
            self.Input_S = None # release memory
            return( self.prevLayer.backpropagate( Error_D, inference, guidedBackprop ) ) 

class knSigmoid(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1 ):
        super().__init__(iparentNetwork,iparams ) # call init on parent  
        self.activationType = True
        self.Input_S = None #  keep the activated predictor around (sigmoid uses that to produce the derivatives)
    
        #self.lastMiniBatchSize = -1 # last minibatch size: used to cache the indices for the im2col operations      
        #self.cachedIndices = None
        if self.parentNetwork.suppressPrint == False : print( "Sigmoid layer")
      
    def generateOutput(self, Input, train = True, saveInput = True) :
        
        #Input_S =  1 / (1 + np.exp(-Input))
        Input_S =  np.exp(-np.logaddexp(0, -Input)) # numerically stable sigmoid

        
        
        if saveInput : self.Input_S =Input_S
        
        return (  Input_S )
        
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this  
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            #self.Input_S = self.Input_S.T
            
            global KNET_ORIG
            if KNET_ORIG == False :  Error_D = self.Input_S *(1 - self.Input_S) * Input  # this is transposed, as D is also transposed during back propagation
            else : Error_D = Input.copy()
            
            self.Input_S = None # release memory
            return( self.prevLayer.backpropagate( Error_D, inference, guidedBackprop ) ) 


class knSoftmax(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1 ):
        super().__init__(iparentNetwork,iparams ) # call init on parent    
        self.activationType = True
        #self.lastMiniBatchSize = -1 # last minibatch size: used to cache the indices for the im2col operations      
        #self.cachedIndices = None
        if self.parentNetwork.suppressPrint == False : print( "Softmax layer")
      
    def generateOutput(self, Input, train = True, saveInput = True) :
       # Output_Z = np.sum(np.exp(Input), axis=1)
        #Output_Z = Output_Z.reshape(Output_Z.shape[0], 1)
        #return np.exp(Input) / Output_Z
        
        #Output_Z =  np.exp (Input - np.max(Input) )
        #return Output_Z / np.sum(Output_Z )
        
    
          probs = np.exp(Input - np.max(Input, axis=1, keepdims=True))
          probs /= np.sum(probs, axis=1, keepdims=True)
            
          return probs
    # https://stackoverflow.com/questions/41947775/avoiding-overflow-error-for-exp-in-numpy
#        SAFETY = 2.
#        mrn = np.finfo(Input.dtype).max # largest representable number
#        thr = np.log(mrn / Input.size) - SAFETY
#        amx = Input.max()
#        if(amx > thr):
#            Output_Z = np.exp(Input - (amx-thr))
#            return Output_Z / (np.exp(thr-amx) + Output_Z.sum())
#        else:
#            Output_Z = np.exp(Input)
#            return Output_Z / (1.0 + Output_Z.sum())
#            
#        
   
        
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this  
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            return( self.prevLayer.backpropagate( Input, inference, guidedBackprop  ) ) # this is the Loss, which is computed outside
    #raise NotImplementedError() # as this is the output layer, this does not backgpopagate the normal way, the loss function is always computed externally



class knSoftplus(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1 ):
        super().__init__(iparentNetwork,iparams ) # call init on parent   
        self.activationType = True
        #self.lastMiniBatchSize = -1 # last minibatch size: used to cache the indices for the im2col operations      
        #self.cachedIndices = None
        if self.parentNetwork.suppressPrint == False : print( "Softplus layer")
      
    def generateOutput(self, Input, train = True, saveInput = True) :#  
        return ( np.log( 1 + np.exp(Input)  ) )
        
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this  
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            global KNET_ORIG
            if KNET_ORIG == False : 
                sigmoid_Input  = ( 1 / (1 + np.exp(-Input)) ) #.            
                Error_D = sigmoid_Input * Input
            else : Error_D = Input.copy()

            return( self.prevLayer.backpropagate(Error_D , inference, guidedBackprop ) ) # the derivative of softplus is sigmoid(x)... so this is Error_D * Input



class knUpsample(knnBaseLayer) :
    def __init__(self, iparentNetwork = None, iparams = -1 ):
        super().__init__(iparentNetwork,iparams ) # call init on parent    
        self.supportLayerType = False
        if self.parentNetwork.suppressPrint == False : print( "knUpsample layer")
      
    def generateOutput(self, Input, train = True, saveInput = True) :#  
        return ( scipy.ndimage.interpolation.zoom(Input, [1,1, 2, 2]) ) # upsize  )
        
    def backpropagate(self,Input, inference = False, guidedBackprop = False ) : # this receives the Error_Delta from the layer after this  
        if self.prevLayer is not None : # if there are any previous layers still, we recursively call them, this means that the Input layer wont be calling this
            return( self.prevLayer.backpropagate(  scipy.ndimage.interpolation.zoom(Input, [1,1, 0.5, 0.5]) , inference, guidedBackprop ) ) # 



#    if not deriv:
#        return np.log( 1 + np.exp(X)  )
#    else:
  #      return k_sigmoid(X)
#