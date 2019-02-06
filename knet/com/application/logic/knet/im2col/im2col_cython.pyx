import numpy as np
cimport numpy as np
cimport cython

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def im2col_cython(np.ndarray[DTYPE_t, ndim=4] x, int field_height,
                  int field_width, int p_h, int p_w, int stride_h, int stride_w):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    
    cdef int HH = (H + 2 * p_h - field_height) / stride_h + 1
    cdef int WW = (W + 2 * p_w - field_width) / stride_w + 1

    #cdef int p = padding
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.pad(x,
            ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant')

    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.zeros(
            (C * field_height * field_width, N * HH * WW),
            dtype=x.dtype)

    # Moving the inner loop to a C function with no bounds checking works, but does
    # not seem to help performance in any measurable way.

    im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW,
                        field_height, field_width, p_h, p_w, stride_h, stride_w)
    return cols


@cython.boundscheck(False)
cdef int im2col_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int field_height, int field_width, int p_h, int p_w, int stride_h, int stride_w) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for yy in range(HH):
            for xx in range(WW):
                for ii in range(field_height):
                    for jj in range(field_width):
                        row = c * field_width * field_height + ii * field_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride_h * yy + ii, stride_w * xx + jj]



def col2im_cython(np.ndarray[DTYPE_t, ndim=2] cols, int N, int C, int H, int W,
                  int field_height, int field_width, int p_h, int p_w, int stride_h, int stride_w):
    cdef np.ndarray x = np.empty((N, C, H, W), dtype=cols.dtype)
    cdef int HH = (H + 2 * p_h - field_height) / stride_h + 1
    cdef int WW = (W + 2 * p_w - field_width) / stride_w + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * p_h, W + 2 * p_w),
                                        dtype=cols.dtype)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_cython_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, p_h, p_w, stride_h, stride_w)
    
    if p_h == 0 and p_w == 0:
        return x_padded
    else :
        if p_h == 0 :
            return x_padded[:, :, :, p_w:-p_w]
        elif  p_w == 0 :
            return x_padded[:, :, p_h:-p_h, :]
        else :
            return x_padded[:, :, p_h:-p_h, p_w:-p_w]
        
#    if p_h > 0 or p_w > 0:
#        return x_padded[:, :, p_h:-p_h, p_w:-p_w]
#    return x_padded


@cython.boundscheck(False)
cdef int col2im_cython_inner(np.ndarray[DTYPE_t, ndim=2] cols,
                             np.ndarray[DTYPE_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int field_height, int field_width, int p_h, int p_w, int stride_h, int stride_w) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            x_padded[i, c, stride_h * yy + ii, stride_w * xx + jj] += cols[row, col]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef col2im_6d_cython_inner(np.ndarray[DTYPE_t, ndim=6] cols,
                            np.ndarray[DTYPE_t, ndim=4] x_padded,
                            int N, int C, int H, int W, int HH, int WW,
                            int out_h, int out_w, int p_h, int p_w, int stride_h, int stride_w):

    cdef int c, hh, ww, n, h, w
    for n in range(N):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for h in range(out_h):
                        for w in range(out_w):
                            x_padded[n, c, stride_h * h + hh, stride_w * w + ww] += cols[c, hh, ww, n, h, w]
    

def col2im_6d_cython(np.ndarray[DTYPE_t, ndim=6] cols, int N, int C, int H, int W,
        int HH, int WW, int p_h, int p_w, int stride_h, int stride_w):
    cdef np.ndarray x = np.empty((N, C, H, W), dtype=cols.dtype)
    cdef int out_h = (H + 2 * p_h - HH) / stride_h + 1
    cdef int out_w = (W + 2 * p_w - WW) / stride_w + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * p_h, W + 2 * p_w),
                                                  dtype=cols.dtype)

    col2im_6d_cython_inner(cols, x_padded, N, C, H, W, HH, WW, out_h, out_w, p_h, p_w, stride_h, stride_w)

    if p_h == 0 and p_w == 0:
        return x_padded
    else :
        if p_h == 0 :
            return x_padded[:, :, :, p_w:-p_w]
        elif  p_w == 0 :
            return x_padded[:, :, p_h:-p_h, :]
        else :
            return x_padded[:, :, p_h:-p_h, p_w:-p_w]

#    if p_h > 0 or p_w > 0:
#        return x_padded[:, :, p_h:-p_h, p_w:-p_w]
#    return x_padded 
