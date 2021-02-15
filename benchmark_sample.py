#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
https://blog.csdn.net/Chermack/article/details/107296924
"""
from numba import cuda, float32, uint8
import numba
import numpy
import math
import time

TPB = 32 # Threads per block per dim, or blockDim.x
BPG = 42 # Blocks per grid per dim, or gridDim.x

@numba.jit(nopython=True)
def matmul_cpu(A, B, C):
    for y in range(B.shape[1]):
        for x in range(A.shape[0]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[x, k] * B[k, y]
            C[x, y] = tmp


@cuda.jit()
def matmul_gpu(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


@cuda.jit()
def matmul_shared_mem(A, B, C):
    sA = cuda.shared.array((32, 32), dtype=float32)
    sB = cuda.shared.array((TPB, TPB), dtype=float32)
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if x >= C.shape[0] and y >= C.shape[1]:
        return
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()
    C[x, y] = tmp


if __name__ == '__main__':

    print(cuda.gpus)
    cuda.select_device(0)

    A = numpy.full((TPB * BPG, TPB * BPG), 3, numpy.float)
    B = numpy.full((TPB * BPG, TPB * BPG), 4, numpy.float)
    C_cpu = numpy.full((A.shape[0], B.shape[1]), 0, numpy.float)

    print("start processing in CPU")
    start_cpu = time.time()
    matmul_cpu(A, B, C_cpu)
    end_cpu = time.time()
    time_cpu = end_cpu - start_cpu
    print("CPU time: " + str(time_cpu))

    # start in GPU
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)

    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))
    C_shared_mem = cuda.device_array((A.shape[0], B.shape[1]))

    threads_per_block = (TPB, TPB) # initialize a 2D block of shape (TPB, TPB), threads count = TPB*TPB

    blocks_per_grid_x = int(math.ceil(A.shape[0] / threads_per_block[0]))
    # blocks per grid should be defined according to TPB and the workload of the specific task
    blocks_per_grid_y = int(math.ceil(B.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    # init a 2D grid of shape (blocks_per_grid_x, blocks_per_grid_y)

    print("start processing in GPU")
    start_gpu = time.time()
    matmul_gpu[blocks_per_grid, threads_per_block](A_global_mem, B_global_mem, C_global_mem)
    cuda.synchronize()
    end_gpu = time.time()
    time_gpu = end_gpu - start_gpu
    print("GPU time(Global memory):" + str(time_gpu))
    C_global_gpu = C_global_mem.copy_to_host()

    start_gpu_shared_memory = time.time()
    matmul_shared_mem[blocks_per_grid, threads_per_block](A_global_mem, B_global_mem, C_shared_mem)
    cuda.synchronize()
    end_gpu_shared_memory = time.time()

    time_gpu_shared = end_gpu_shared_memory - start_gpu_shared_memory
    print("GPU time(shared memory):" + str(time_gpu_shared))
    C_shared_gpu = C_global_mem.copy_to_host()

    cuda.close() # close the device