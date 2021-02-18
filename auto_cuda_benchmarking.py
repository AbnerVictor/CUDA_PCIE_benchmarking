from gpu_bench_lib import *
from pynvml import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

if __name__ == '__main__':

    nvmlInit()  # 初始化
    print("Driver: ", nvmlSystemGetDriverVersion())  # 显示驱动信息

    # 查看设备
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("GPU", i, ":", nvmlDeviceGetName(handle))

    gpu_selected = 0
    # handler = nvmlDeviceGetHandleByIndex(gpu_selected)  # select gpu
    handle = nvmlDeviceGetHandleByIndex(gpu_selected)  # select gpu
    print('Current link: x', nvmlDeviceGetCurrPcieLinkWidth(handle), 'gen', nvmlDeviceGetCurrPcieLinkGeneration(handle))

    handler = None

    sharpness_pref = 0.8
    TPB = (16, 16)
    number_of_streams = 1

    TEST_FPS = False
    timeout1 = 10

    TEST_COPY = True
    DUMMY_DATA_SHAPE = [1920, 1080, 85]
    timeout2 = 10

    ###########################################################################

    IN = image_read('test_img_HD.png', to_float=False)

    # OUT_cpu = cpu_run(IN, sharpness_pref)
    # plt.imshow(OUT_cpu)
    # plt.show()
    # mpimg.imsave('OUT_cpu.png', OUT_cpu)

    OUT_gpu_single_run = gpu_single_run(IN, gpu_selected, sharpness_pref, TPB)
    plt.imshow(OUT_gpu_single_run)
    plt.show()
    mpimg.imsave('OUT_gpu.png', OUT_gpu_single_run)


    if TEST_FPS:
        OUT_gpu_continuous_run_multi_stream = gpu_continuous_run_multi_stream_new(IN, gpu_selected, sharpness_pref, TPB,
                                                                              number_of_streams=number_of_streams,
                                                                              ENABLE_CONTINUOUS_DEVICE_2_HOST=False,
                                                                              ENABLE_CONTINUOUS_HOST_2_DEVICE=False,
                                                                              timeout=timeout1, nvml_handler=handler)

        plt.imshow(OUT_gpu_continuous_run_multi_stream, vmax=255)
        plt.show()
        mpimg.imsave('OUT_gpu_multi_run.png', OUT_gpu_continuous_run_multi_stream)

        gpu_continuous_run_multi_stream_new(IN, gpu_selected, sharpness_pref, TPB,
                                        number_of_streams=number_of_streams,
                                        ENABLE_CONTINUOUS_DEVICE_2_HOST=False,
                                        ENABLE_CONTINUOUS_HOST_2_DEVICE=True,
                                        timeout=timeout1, nvml_handler=handler)

        gpu_continuous_run_multi_stream_new(IN, gpu_selected, sharpness_pref, TPB,
                                        number_of_streams=number_of_streams,
                                        ENABLE_CONTINUOUS_DEVICE_2_HOST=True,
                                        ENABLE_CONTINUOUS_HOST_2_DEVICE=False,
                                        timeout=timeout1, nvml_handler=handler)

        gpu_continuous_run_multi_stream_new(IN, gpu_selected, sharpness_pref, TPB,
                                        number_of_streams=number_of_streams,
                                        ENABLE_CONTINUOUS_DEVICE_2_HOST=True,
                                        ENABLE_CONTINUOUS_HOST_2_DEVICE=True,
                                        timeout=timeout1, nvml_handler=handler)

    if TEST_COPY:

        run_continuous_copy(DUMMY_DATA_SHAPE=DUMMY_DATA_SHAPE, gpu_device_selected=gpu_selected,
                            ENABLE_CONTINUOUS_HOST_2_DEVICE=True, ENABLE_CONTINUOUS_DEVICE_2_HOST=False,
                            timeout=timeout2)

        run_continuous_copy(DUMMY_DATA_SHAPE=DUMMY_DATA_SHAPE, gpu_device_selected=gpu_selected,
                            ENABLE_CONTINUOUS_HOST_2_DEVICE=False, ENABLE_CONTINUOUS_DEVICE_2_HOST=True,
                            timeout=timeout2)

        run_continuous_copy_bidirect(DUMMY_DATA_SHAPE=DUMMY_DATA_SHAPE, gpu_device_selected=gpu_selected, timeout=timeout2)

        # run_continuous_copy(DUMMY_DATA_SHAPE=DUMMY_DATA_SHAPE, gpu_device_selected=gpu_selected,
        #                     ENABLE_CONTINUOUS_HOST_2_DEVICE=True, ENABLE_CONTINUOUS_DEVICE_2_HOST=True,
        #                     timeout=timeout2)


