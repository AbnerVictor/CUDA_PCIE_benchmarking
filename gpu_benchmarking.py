from numba import cuda, float32
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numba
import numpy
import math
import time
from pynvml import *

print("=================== GPU basic info ====================")

nvmlInit()  # 初始化
print("Driver: ", nvmlSystemGetDriverVersion())  # 显示驱动信息

# 查看设备
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("GPU", i, ":", nvmlDeviceGetName(handle))

gpu_device_selected = 0
handle = nvmlDeviceGetHandleByIndex(gpu_device_selected)  # select gpu

print('Current link: pcie gen', nvmlDeviceGetCurrPcieLinkGeneration(handle), 'x',
      nvmlDeviceGetCurrPcieLinkWidth(handle))

SM = 1  # number of SM your GPU equipped with

target_FPS = 300

# CAS parameters
sharpness_pref = 0.8
developMaximum = -0.125 - (sharpness_pref * (0.2 - 0.125))

RUN_SINGLE_CPU = False
RUN_SINGLE_GPU = False
RUN_CONTINUOUS_GPU = True

@numba.jit(nopython=True)
def cas_img_cpu(INPUT: numpy.ndarray, OUT: numpy.ndarray):
    h, w, d = INPUT.shape
    for row in range(1, int(h) - 1):
        for col in range(1, int(w) - 1):
            window = INPUT[row - 1:row + 2, col - 1:col + 2, :]
            min_g = min(window[0, 1, 1], window[1, 0, 1], window[1, 2, 1], window[2, 1, 1], window[1, 1, 1])
            max_g = max(window[0, 1, 1], window[1, 0, 1], window[1, 2, 1], window[2, 1, 1], window[1, 1, 1]) + 0.001
            BAS = 0
            d_max_g = 1 - max_g
            if d_max_g < min_g:
                BAS = d_max_g / max_g
            else:
                BAS = min_g / max_g
            _w = BAS ** 0.5 * developMaximum
            OUT[row, col, :] = (_w * (window[0, 1, :] + window[1, 0, :] + window[1, 2, :] + window[2, 1, :]) + window[1,
                                                                                                               1,
                                                                                                               :]) / (
                                           _w * 4 + 1)


@cuda.jit()
def cas_img_gpu(INPUT: numpy.ndarray, OUT: numpy.ndarray, _developMaximum):
    row, col = cuda.grid(2)
    if 0 < row < INPUT.shape[0] - 1 and 0 < col < INPUT.shape[1] - 1:
        min_g = min(INPUT[row - 1, col, 1], INPUT[row, col - 1, 1], INPUT[row, col + 1, 1], INPUT[row + 1, col, 1],
                    INPUT[row, col, 1])
        max_g = max(INPUT[row - 1, col, 1], INPUT[row, col - 1, 1], INPUT[row, col + 1, 1], INPUT[row + 1, col, 1],
                    INPUT[row, col, 1]) + 0.001
        d_max_g = 1 - max_g
        BAS = 0.
        if d_max_g < min_g:
            BAS = d_max_g / max_g
        else:
            BAS = min_g / max_g
        _w = BAS ** 0.5 * _developMaximum[0]

        def cas(w, a, b, c, d, e):
            return (w * (a + b + c + d) + e) / (w * 4 + 1)

        OUT[row, col, 0] = cas(_w, INPUT[row - 1, col, 0], INPUT[row, col - 1, 0], INPUT[row, col + 1, 0],
                               INPUT[row + 1, col, 0], INPUT[row, col, 0])
        OUT[row, col, 1] = cas(_w, INPUT[row - 1, col, 1], INPUT[row, col - 1, 1], INPUT[row, col + 1, 1],
                               INPUT[row + 1, col, 1], INPUT[row, col, 1])
        OUT[row, col, 2] = cas(_w, INPUT[row - 1, col, 2], INPUT[row, col - 1, 2], INPUT[row, col + 1, 2],
                               INPUT[row + 1, col, 2], INPUT[row, col, 2])


print("\n=================== task info ====================")

img = numpy.ascontiguousarray(numpy.array(Image.open('test_img_HD.png'))[:, :, :3])

print('read texture:', img.shape, img.dtype)

h, w, d = img.shape
MegaBytes_per_frame = h * w * d / (1024 * 1024)

TPB = (32, 32)  # threads per warp, strongly depends on gpu architecture
BPG = (int(numpy.ceil(h / TPB[0])), int(numpy.ceil(w / TPB[1])))


IN = img / 255.0  # normalize the image

print(MegaBytes_per_frame, 'MB of data per raw frame')
print(MegaBytes_per_frame * target_FPS, 'MB of data per second')


if RUN_SINGLE_CPU:
    print("\n - Starting in CPU")

    # Starting in CPU
    OUT_cpu = numpy.zeros(IN.shape, dtype=float)
    cpu_start = time.time()
    cas_img_cpu(IN, OUT_cpu)
    cpu_end = time.time()
    OUT_cpu = numpy.where(OUT_cpu < 1, numpy.where(OUT_cpu > 0, OUT_cpu, 0.0), 1.0)
    print("CPU single image CAS time: " + str(cpu_end - cpu_start))
    plt.imshow(OUT_cpu)
    plt.show()
    mpimg.imsave('OUT_cpu.png', OUT_cpu)

if RUN_SINGLE_GPU:

    print("\n - Starting in GPU")

    # Starting in GPU
    cuda.select_device(gpu_device_selected)

    vram_info = nvmlDeviceGetMemoryInfo(handle)
    print('VRAM usage: ', vram_info.used / (1024 * 1024), 'MB')

    # host to device copy
    copy_host_2_device_start = time.time()
    IN_global_mem = cuda.to_device(IN)  # texture
    devMax_global_mem = cuda.to_device(numpy.array([developMaximum]))

    OUT_global_mem = cuda.device_array((h, w, d), dtype=float)
    copy_host_2_device_end = time.time()

    print('host to device copy time:', copy_host_2_device_end - copy_host_2_device_start)

    vram_info = nvmlDeviceGetMemoryInfo(handle)
    print('VRAM usage: ', vram_info.used / (1024 * 1024), 'MB')

    cas_img_gpu[BPG, TPB](IN_global_mem, OUT_global_mem, devMax_global_mem)
    cuda.synchronize()

    gpu_cas_start = time.time()
    cas_img_gpu[BPG, TPB](IN_global_mem, OUT_global_mem, devMax_global_mem)
    cuda.synchronize()
    gpu_cas_end = time.time()
    print('GPU single image CAS time:', gpu_cas_end - gpu_cas_start)

    copy_device_2_host_start = time.time()
    OUT_global_gpu = OUT_global_mem.copy_to_host()
    OUT_global_gpu = numpy.where(OUT_global_gpu < 1, numpy.where(OUT_global_gpu > 0, OUT_global_gpu, 0.0), 1.0)
    copy_device_2_host_end = time.time()
    print('host to device copy time:', copy_device_2_host_end - copy_device_2_host_start)

    plt.imshow(OUT_global_gpu)
    plt.show()
    mpimg.imsave('OUT_gpu.png', OUT_global_gpu)

if RUN_CONTINUOUS_GPU:

    print("\n=================== Continuous Run ====================")

    IN_global_mem = cuda.to_device(IN)  # texture
    devMax_global_mem = cuda.to_device(numpy.array([developMaximum]))

    OUT_global_mem = cuda.device_array((h, w, d), dtype=float)

    fps = 0
    last_time = time.time()

    # Rendering
    while 1:
        current_time = time.time()
        if current_time - last_time < 1:
            if fps >= target_FPS:
                continue
            fps += 1
        else:
            last_time = current_time
            print('\rFPS:', fps, end='')
            fps = 0
        cas_img_gpu[BPG, TPB](IN_global_mem, OUT_global_mem, devMax_global_mem)
        cuda.synchronize()
        # OUT_global_mem.copy_to_host()

nvmlShutdown()
