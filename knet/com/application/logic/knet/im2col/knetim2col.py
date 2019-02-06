# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:46:12 2017

@author: Martin Kelemen
"""
import numpy as np

cython_mode = False
GPU_mode = False
cython_version = False
DEBUG_FLAG = False
try:
# remote calling from STL10 test ( IE knet3.py is not the root)
    #from ......com.application.logic.knet.im2col.im2col_cython import col2im_cython, im2col_cython
    
#remote calling knet3.py directly
    from com.application.logic.knet.im2col.im2col_cython import col2im_cython, im2col_cython
    
    # does not work
    #from im2col_cython import col2im_cython, im2col_cython 
    cython_mode = True

except ImportError:
    print('Cythonized version of im2col not present, try running (C++ compiler needed):')
    print('python setup.py build_ext --inplace')
    print('defaulting to numpy, so it will be 100x slower')
    print('(also requires Cython: conda install Cython)')

#from com.application.logic.knet.im2col.im2col_cython.im2col_cython import col2im_cython, im2col_cython 
#cython_mode = True

def mode_set(numpy_or_cupy, gpu_enabled = False ):
    print("cython im2col gpu ode set to: " + str(gpu_enabled) )
    global np
    np = numpy_or_cupy
    global cython_mode
    global GPU_mode
    global cython_version
    GPU_mode = gpu_enabled
    cython_version = cython_mode and GPU_mode == False
    print("cython_mode / GPU_mode: " + str(cython_mode) + "/" + str(GPU_mode) + " so cython_version is: " + str(cython_version))


def im2col_indices(x, field_height, field_width, p_h=1, p_w=1, stride_h=1, stride_w=1):
    global cython_version
    global DEBUG_FLAG
    if cython_version : 
        if DEBUG_FLAG is False : 
            print("Cython function called")
            DEBUG_FLAG = True
        return(im2col_cython(x, field_height, field_width, p_h, p_w, stride_h, stride_w) )
    else :
        return( im2col_indices_slow(x, field_height, field_width, p_h, p_w, stride_h, stride_w) )

    
def col2im_indices(dx_cols, x_shape, field_height=3, field_width=3, p_h=1, p_w=1, stride_h=1, stride_w=1):
    global cython_version
    
    
    if cython_version : 
        return( col2im_cython(dx_cols, x_shape[0],x_shape[1], x_shape[2], x_shape[3], field_height, field_width, p_h, p_w, stride_h, stride_w) )
    else :
        return( col2im_indices_slow(dx_cols, x_shape, field_height, field_width, p_h, p_w, stride_h, stride_w) )

###############################################################################
# CONV utils
###############################################################################

# -> this function returns the indices of the original image that could be used to generate the stretched out version
# if all the minibatches are the same size, then this function could be cached
# or we could juts keep it cached, and keep checking if minibatch size is same, and if yes, then keep reusing it
def get_im2col_indices(x_shape, field_height, field_width, p_h=1, p_w=1, stride_h=1, stride_w=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape # get the 4 dimensions of the input matrix
    #print("x_shape is: " + str(x_shape))
    assert (H + 2 * p_h - field_height) % stride_h == 0 # check if image dimensions are compatible with the desired patches
    assert (W + 2 * p_w - field_width) % stride_w == 0
    out_height = int((H + 2 * p_h - field_height) / stride_h + 1) # get the final output matrix sizes, IE Orig Height + twice the padding (for each side) minus a filter, divided by the number of strides
    out_width = int((W + 2 * p_w - field_width) / stride_w + 1)

    i0 = np.repeat(np.arange(field_height), field_width) # for a filter size of 3, this creates a column vector of: 0,0,0,1,1,1,2,2,2 
    i0 = np.tile(i0, C) # tile repeats the original Array C number of times: IE if we had multiple colour channels this would repeat it 3 times
    i1 = stride_h * np.repeat(np.arange(out_height), out_width) # column vector of: 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3
    j0 = np.tile(np.arange(field_width), field_height * C) # for a filter size of 3, this creates 0,1,2,0,1,2,0,1,2
    j1 = stride_w * np.tile(np.arange(out_width), out_height) # for an out_size this creates , 0,3,0,3
    i = i0.reshape(-1, 1) + i1.reshape(1, -1) # creates a 9x4 matrix, out of the i0 and i1 coordinates
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # creates a 9x4 matrix, out of the j0 and j1

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1) # k is a 9x1 column vector, of 0s

    return (k.astype(int), i.astype(int), j.astype(int))




# Goes through an image, and finds all the patches of given size (eg 3x3), then stretches those out into columns and retuns a matrix
def im2col_indices_slow(x, field_height, field_width, p_h=1, p_w=1, stride_h=1, stride_w=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
   # p_h = field_height > 1 and padding or 0
  #  p_w = field_width > 1 and padding or 0

    x_padded = np.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant') # adds a ring of 0s around the original image

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, p_h,p_w, stride_h, stride_w)

    cols = x_padded[:, k, i, j] # extract the stretched image out from the original
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1) # reshape it
    return cols






# turns an image that was in patches into back into an original
# NOTE: the patches will be copied OVER each other, IE image -> im2col_indices -> col2im_indices will NOT get back the original image
def col2im_indices_slow(cols, x_shape, field_height=3, field_width=3, p_h=1, p_w=1, stride_h=1, stride_w=1):
    global GPU_mode 
    targetDtype = cols.dtype
    if GPU_mode : targetDtype = 'float32' # CUPY's scatter_add only works for float32, and NOT float16 (or float64), so we will have to cast it to f32 then back to what it was originally
    
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape # get the 4 dimensions of the input matrix
    H_padded, W_padded = H + 2 * p_h, W + 2 * p_w
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=targetDtype) # dtype='float32') # , dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, p_h, p_w, stride_h, stride_w)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    #np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    #print("col2im_indices_slow k TYPE is : " + str(k.dtype))
    #print("col2im_indices_slow i TYPE is : " + str(i.dtype))
    #print("col2im_indices_slow j TYPE is : " + str(j.dtype))
    #print("col2im_indices_slow x_padded TYPE is : " + str(x_padded.dtype))
    #print("col2im_indices_slow cols_reshaped TYPE is : " + str(cols_reshaped.dtype))
    
    #import numpy
    #cols_reshaped_numpy = np.asnumpy(cols_reshaped)
    #x_padded_numpy = numpy.zeros((N, C, H_padded, W_padded), dtype=cols.dtype) 
    #k_numpy = np.asnumpy(k)
    #i_numpy = np.asnumpy(i)
    #j_numpy = np.asnumpy(j)
    #numpy.add.at(x_padded_numpy, (slice(None), k_numpy, i_numpy, j_numpy), cols_reshaped_numpy)
    #print("x_padded_numpy TYPE is : " + str(x_padded_numpy.dtype))
   # print("using Float16 still works for numpy!!: " + str(x_padded_numpy), flush=True)
    
    
    if GPU_mode :  
        np.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped) # scatter_add is cupy's equivalent to np.add.at : https://media.readthedocs.org/pdf/cupy/v1.0.1/cupy.pdf
        x_padded = x_padded.astype(cols.dtype) # cast it back for cupy runs
    else :     np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    
#    if p_h == 0 and p_w == 0:
#        return x_padded
#    return x_padded[:, :, p_h:-p_h, p_w:-p_w]
    
    if p_h == 0 and p_w == 0:
        return x_padded
    else :
        if p_h == 0 :
            return x_padded[:, :, :, p_w:-p_w]
        elif  p_w == 0 :
            return x_padded[:, :, p_h:-p_h, :]
        else :
            return x_padded[:, :, p_h:-p_h, p_w:-p_w]
        

#def col2im_indices_old(cols, x_shape, field_height=3, field_width=3, padding=1,
#                   stride=1):
#    """ An implementation of col2im based on fancy indexing and np.add.at """
#    N, C, H, W = x_shape
#    H_padded, W_padded = H + 2 * padding, W + 2 * padding
#    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
#    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
#                                 stride)
#    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
#    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
#    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
#    if padding == 0:
#        return x_padded
#    return x_padded[:, :, padding:-padding, padding:-padding]


                         
                            
def col2im_6d(cols,  N,  C,  H,  W, HH, WW, p_h, p_w, stride_h, stride_w):
    x = np.empty((N, C, H, W), dtype=cols.dtype)
    out_h = int ( (H + 2 * p_h - HH) / stride_h + 1 )
    out_w = int ( (W + 2 * p_w - WW) / stride_w + 1 )
    x_padded = np.zeros((N, C, H + 2 * p_h, W + 2 * p_w),dtype=cols.dtype)

    for n in range(N):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for h in range(out_h):
                        for w in range(out_w):
                            x_padded[n, c, stride_h * h + hh, stride_w * w + ww] += cols[c, hh, ww, n, h, w]  

    if p_h == 0 and p_w == 0:
        return x_padded
    else :
        if p_h == 0 :
            return x_padded[:, :, :, p_w:-p_w]
        elif  p_w == 0 :
            return x_padded[:, :, p_h:-p_h, :]
        else :
            return x_padded[:, :, p_h:-p_h, p_w:-p_w]
