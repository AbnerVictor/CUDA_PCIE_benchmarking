from numba import cuda, float32
import numba
import numpy
import math
import time

TPB = 1024 # number of threads per Block, strongly depends on gpu architecture
SM = 1920 # number of SM your GPU equipped with

TPB_per_dim = int(numpy.floor(numpy.sqrt(TPB)))  # blockDim.x for a 2D block
BPG_per_dim = int(numpy.floor(numpy.sqrt(SM)))

@numba.jit(nopython=True)
def rotate_img_cpu(IN, degree, OUT):
    