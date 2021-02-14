from pynvml import *
import numpy

nvmlInit()  # 初始化
print("Driver: ", nvmlSystemGetDriverVersion())  # 显示驱动信息

# 查看设备
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("GPU", i, ":", nvmlDeviceGetName(handle))

gpu_device_selected = 0
handle = nvmlDeviceGetHandleByIndex(gpu_device_selected)  # select gpu

def pcie_monitor():
    print('Current link: x', nvmlDeviceGetCurrPcieLinkWidth(handle), 'gen', nvmlDeviceGetCurrPcieLinkGeneration(handle))
    print('PCIE throughput TX', numpy.round(50 * nvmlDeviceGetPcieThroughput(handle, 0) / 1024 / 1024, 3),
          'MB/s RX', numpy.round(50 * nvmlDeviceGetPcieThroughput(handle, 1) / 1024 / 1024, 3), 'MB/s')

def vram_monitor():
    vram_info = nvmlDeviceGetMemoryInfo(handle)
    print('VRAM usage: ', vram_info.used / (1024 * 1024), 'MB')

while True:
    pcie_monitor()
    vram_monitor()


nvmlShutdown()