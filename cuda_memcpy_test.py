import pycuda.driver as cuda
from pycuda.tools import PageLockedMemoryPool
import numpy as np
import time
import ctypes
import pdb
from queue import Queue
from threading import Thread
from pycuda.tools import make_default_context
import matplotlib.pyplot as plt

import threading
# Initialize CUDA
cuda.init()

global ctx
ctx = make_default_context() # will initialize the first device it finds
dev = ctx.get_device()

def _finish_up():
    global ctx
    ctx.pop()
    ctx = None

    from pycuda.tools import clear_context_caches
    clear_context_caches()

import atexit
atexit.register(_finish_up)

num_elems = [5000000, 10000000, 20000000, 30000000, 40000000]

# prints pci_bus_id, device name and device id for installed GPUs. You can run the Linux lspci command on the bus_id to
# obtain information about the number of PCIe lanes on that bus. This will give you the expected bandwidth
def print_device_info():
    driver_ver = cuda.get_version()
    print("CUDA Driver Version: {0}.{1}.{2}".format(driver_ver[0], driver_ver[1], driver_ver[2]))
    num_cuda_devices = cuda.Device.count()
    for i in range(0, num_cuda_devices):
        dev = cuda.Device(i)
        pci_bus_id = dev.pci_bus_id()
        dev_name = dev.name()
        print("device id: {0}, device name: {1}, bus_id: {2}".format(i, dev_name, pci_bus_id))

# Helper function to copy src array to destination using ctypes memmove
def copy_np_to_pinned_memory(src, dest):
    src_ = src.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    dest_ = dest.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sz = src.size * ctypes.sizeof(ctypes.c_float)
    ctypes.memmove(dest_, src_, sz)

# This function measures the time taken to transfer data from host-to-device (h2d) when:
# 1. source is in unpinned (pagaeable) memory
# 2. source is in pinned memory. In this case, we also measure time taken to transfer data
# from unpinned to pinned memory.
# Times are measured for different data sizes and plotted. Data transfer bandwidth is also calculated from
# the transfer times.
def compare_performance():
    # a quick warm up..
    n = 25000000
    a = np.random.randn(n).astype(np.float32)
    # allocate space on GPU
    mem_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(mem_gpu, a)
    # free space on GPU
    mem_gpu.free()
    h2d_nopin = []
    h2d_nopin_bw = []
    # measure timing without pinning
    for n in num_elems:
        # the data to be transferred
        a = np.random.randn(n).astype(np.float32)
        # allocate space on GPU
        mem_gpu = cuda.mem_alloc(a.nbytes)
        # only measure h2d transfer time
        start = time.perf_counter()
        cuda.memcpy_htod(mem_gpu, a)
        te = time.perf_counter() - start #te: time elapsed
        h2d_nopin.append(te)
        h2d_nopin_bw.append(a.nbytes/(10**9 * (te))) # convert to a bandwidth
        # free space on GPU
        mem_gpu.free()
    # now do pinning and measure time to pin and time to transfer
    h2h_pinned = [] # records the transfer time from unpinned -> pinned memory
    h2d_pin = [] # records the host to device transfer time with data in pinned memory.
    h2d_pin_total = [] # records the total (sum of the previous two)
    h2d_pin_bw = [] #h2d_pin, converted to a bandwidth (GB/sec)
    for i, n in enumerate(num_elems):
        a = np.random.randn(n).astype(np.float32)
        # allocate space on GPU
        mem_gpu = cuda.mem_alloc(a.nbytes)
        # allocate page locked memory
        a_pin = cuda.register_host_memory(a)
        # copy data from np array to pinned memory and measure transfer time
        start = time.perf_counter()
        copy_np_to_pinned_memory(a, a_pin)
        te = time.perf_counter() - start  # te: time elapsed
        h2h_pinned.append(te)
        # measure h2d transfer time
        start = time.perf_counter()
        cuda.memcpy_htod(mem_gpu, a_pin)
        te = time.perf_counter() - start #te: time elapsed
        h2d_pin.append(te)
        h2d_pin_bw.append(a.nbytes / (10**9 * te))
        h2d_pin_total.append(h2d_pin[i] + h2h_pinned[i])
        # free allocated pinned memory
        a_pin.base.unregister()
        # free space on GPU
        mem_gpu.free()

    fig = plt.figure()
    num_elems_mb = [x*4/10**6 for x in num_elems]

    plt.plot(num_elems_mb, h2d_nopin, 'g', label='h2d transfer_time (no pinning)')
    plt.plot(num_elems_mb, h2d_pin, 'r', label='h2d transfer_time (with pinning)')
    plt.plot(num_elems_mb, h2h_pinned, 'b', label='h2h transfer_time')
    plt.plot(num_elems_mb, h2d_pin_total, 'k', label='h2d transfer_time (with pinning, total)')
    plt.legend()
    plt.xlabel('data size (MB)')
    plt.ylabel('time (sec)')
    plt.show()


if __name__ == '__main__':
    print_device_info()
    compare_performance()