from numba import cuda, float32
from PIL import Image
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

TPB = 1024  # number of threads per Block, strongly depends on gpu architecture
SM = 1920  # number of SM your GPU equipped with

target_FPS = 300

TPB_per_dim = int(numpy.floor(numpy.sqrt(TPB)))  # blockDim.x for a 2D block
BPG_per_dim = int(numpy.floor(numpy.sqrt(SM)))


@numba.jit(nopython=True)
def scale_img_cpu(IN, OUT):
    pass


@cuda.jit()
def rotate_img_30_deg_gpu(IN, OUT):
    pass


print("\n=================== task info ====================")

img = numpy.ascontiguousarray(numpy.array(Image.open('test_img_UHD.png'))[:, :, :3])

print('read texture:', img.shape, img.dtype)

h, w, d = img.shape
MegaBytes_per_frame = h * w * d / (1024 * 1024)
plt.imshow(img)
plt.show()

print(MegaBytes_per_frame, 'MB of data per raw frame')
print(MegaBytes_per_frame * target_FPS, 'MB of data per second')

fps = 0
last_time = time.time()

print("\n=================== runtime info ====================")
# host to device copy
cuda.select_device(gpu_device_selected)

vram_info = nvmlDeviceGetMemoryInfo(handle)
print('VRAM usage: ', vram_info.used / (1024 * 1024), 'MB')

copy_host_2_device_start = time.time()
IN_global_mem = cuda.to_device(img)
copy_host_2_device_end = time.time()
print('host to device copy time:', copy_host_2_device_end - copy_host_2_device_start)

vram_info = nvmlDeviceGetMemoryInfo(handle)
print('VRAM usage: ', vram_info.used / (1024 * 1024), 'MB')

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

nvmlShutdown()
