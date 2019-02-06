# -*- coding: utf-8 -*-

# CPU / GPU agnostic code:
# usage:
# instead of doing 'import numpy as np, import np as agnostic as:' 
# global np
# np = get_numpy_or_cupy(True)

# when receiving (numpy) data from the CPU: cast it as the np module we have
# Input_from_CPU = numpy.random.randn(3,3)
# Input_as_GPU_or_CPU = np.asarray(Input_from_CPU) # when receiving data, cast it into the appropriate type

# Output_from_GPU = Input_as_GPU_or_CPU + Input_as_GPU_or_CPU # perform operations on the GPU (or the CPU it is agnostic)

# return( castOutputToCPU(Output_from_GPU) )  # When returning data, cast it back to numpy


# then before running Knet,   to enable GPU as: (otherwise it will default to CPU)
# knet_main.initGPU()



# reference: 
    # https://docs-cupy.chainer.org/en/stable/reference/ndarray.html

import os
import warnings

import numpy


GPU_enabled = False
    
def get_numpy_or_cupy(preferGPU = False):
    global GPU_enabled

    
    if (preferGPU) :
        try:
            print("requested GPU, trying to load CUDA via CUPY")
            import cupy
            import cupy.cuda
            import cupy.cuda.cublas
        
            cuda = cupy.cuda
            cublas = cuda.cublas
        
            ndarray = cupy.ndarray
            Device = cuda.Device
            Event = cuda.Event
            Stream = cuda.Stream
        
            GPU_enabled = True
            print("NVIDIA CUDA enabled via CUPY")
            
        except Exception as e:
            _resolution_error = e
            print("Error: CUDA is NOT installed!")
            print(_resolution_error)
    else : print("asked for CPU, so returning numpy")
    
    if GPU_enabled:
        return cupy
    return numpy
    
    
def castOutputToCPU(data) :
    global GPU_enabled

    if GPU_enabled : 
        print("GPU was enabled so we cast back to numpy")
        return(np.asnumpy(data))
    else : 
        print("we were on numpy to begin with, return data as is")
        return data
    
