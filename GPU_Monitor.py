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

KB = 1024
MB = KB * KB
GB = KB * MB

pci_bw = {
        # Keys = PCIe-Generation, Values = Max PCIe Lane BW (per direction)
        # [Note: Using specs at https://en.wikipedia.org/wiki/PCI_Express]
        1: (250.0 * MB),
        2: (500.0 * MB),
        3: (985.0 * MB),
        4: (1969.0 * MB),
        5: (3938.0 * MB),
        6: (7877.0 * MB),
    }

print('Current link speed', nvmlDeviceGetCurrPcieLinkWidth(handle) * pci_bw[nvmlDeviceGetCurrPcieLinkGeneration(handle)] / MB, 'MB/s')


def pcie_throughput_monitor():
    TX = 0
    RX = 0
    for i in range(0, 50):
        TX = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)
        # RX = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)
    print('\rPCIE throughput TX', numpy.round(TX / KB, 3), 'MB/s',
          'RX', numpy.round(2*RX / KB, 3), 'MB/s', end='', flush=True)

def vram_monitor():
    vram_info = nvmlDeviceGetMemoryInfo(handle)
    print('VRAM usage: ', vram_info.used / (1024 * 1024), 'MB')

try:
    while True:
        pcie_throughput_monitor()
except KeyboardInterrupt:
    pass

nvmlShutdown()