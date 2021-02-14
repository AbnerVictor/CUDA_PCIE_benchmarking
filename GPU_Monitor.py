from pynvml import *
import numpy
import sys
from time import sleep

nvmlInit()  # 初始化
print("Driver: ", nvmlSystemGetDriverVersion())  # 显示驱动信息

# 查看设备
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("GPU", i, ":", nvmlDeviceGetName(handle))

gpu_device_selected = 0
handle = nvmlDeviceGetHandleByIndex(gpu_device_selected)  # select gpu
print('Current link: x', nvmlDeviceGetCurrPcieLinkWidth(handle), 'gen', nvmlDeviceGetCurrPcieLinkGeneration(handle))


def pcie_throughput_monitor():
    TX = 0
    RX = 0
    for i in range(0, 25):
        TX += nvmlDeviceGetPcieThroughput(handle, 0)
        RX += nvmlDeviceGetPcieThroughput(handle, 1)
    print('\rPCIE throughput TX', numpy.round(2*TX / 1024 / 1024, 3),
          'MB/s RX', numpy.round(2*RX / 1024 / 1024, 3), 'MB/s', end='', flush=True)

def vram_monitor():
    vram_info = nvmlDeviceGetMemoryInfo(handle)
    print('VRAM usage: ', vram_info.used / (1024 * 1024), 'MB')

try:
    while True:
        pcie_throughput_monitor()
except KeyboardInterrupt:
    pass

nvmlShutdown()