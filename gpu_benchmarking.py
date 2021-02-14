from numba import cuda, float32
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numba
import numpy
import time

######################################## PARAMETERS ############################################

gpu_device_selected = 0

target_FPS = 300

# CAS parameters
sharpness_pref = 0.8
developMaximum = -0.125 - (sharpness_pref * (0.2 - 0.125))

RUN_SINGLE_CPU = False
RUN_SINGLE_GPU = False

RUN_CONTINUOUS_GPU_SINGLE_STREAM = True

RUN_CONTINUOUS_GPU_MULTI_STREAM = False
number_of_streams = 10 # takes up more VRAM

ENABLE_CONTINUOUS_HOST_2_DEVICE = True

ENABLE_CONTINUOUS_DEVICE_2_HOST = False

######################################## PARAMETERS ############################################


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
            OUT[row, col, :] = (_w * (window[0, 1, :] + window[1, 0, :] + window[1, 2, :] + window[2, 1, :]) + window[1, 1, :]) \
                               / (_w * 4 + 1)

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

img = numpy.ascontiguousarray(numpy.array(Image.open('test_img_UHD.png'))[:, :, :3])
IN = img / 255.0  # normalize the image, float64 now

h, w, d = IN.shape

TPB = (32, 32)  # threads per warp, strongly depends on gpu architecture
BPG = (int(numpy.ceil(h / TPB[0])), int(numpy.ceil(w / TPB[1])))

print('read texture:', IN.shape, IN.dtype)

MegaBytes_per_frame = h * w * d * 64 / (8 * 1024 * 1024)
print(MegaBytes_per_frame, 'MB of data per raw frame')
# print(MegaBytes_per_frame * target_FPS, 'MB of data per second')


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

    # host to device copy
    copy_host_2_device_start = time.time()
    IN_global_mem = cuda.to_device(IN)  # texture
    devMax_global_mem = cuda.to_device(numpy.array([developMaximum]))

    OUT_global_mem = cuda.device_array((h, w, d), dtype=float)
    copy_host_2_device_end = time.time()

    print('host to device copy time:', copy_host_2_device_end - copy_host_2_device_start)


    cas_img_gpu[BPG, TPB](IN_global_mem, OUT_global_mem, devMax_global_mem)
    cuda.synchronize()

    gpu_cas_start = time.time()
    cas_img_gpu[BPG, TPB](IN_global_mem, OUT_global_mem, devMax_global_mem)
    cuda.synchronize()
    gpu_cas_end = time.time()
    print('GPU single image CAS time:', gpu_cas_end - gpu_cas_start)

    copy_device_2_host_start = time.time()
    OUT_gpu = OUT_global_mem.copy_to_host()
    copy_device_2_host_end = time.time()

    OUT_gpu = numpy.where(OUT_gpu < 1, numpy.where(OUT_gpu > 0, OUT_gpu, 0.0), 1.0)
    print('host to device copy time:', copy_device_2_host_end - copy_device_2_host_start)

    plt.imshow(OUT_gpu)
    plt.show()
    mpimg.imsave('OUT_gpu.png', OUT_gpu)

if RUN_CONTINUOUS_GPU_SINGLE_STREAM:

    print("\n=================== Continuous run, single stream ====================")

    IN_global_mem = cuda.to_device(IN)  # texture
    devMax_global_mem = cuda.to_device(numpy.array([developMaximum]))
    OUT_global_mem = cuda.device_array((h, w, d), dtype=float)

    fps = 0
    last_time = time.time()

    # Rendering
    try:
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

            if ENABLE_CONTINUOUS_HOST_2_DEVICE:
                IN_global_mem = cuda.to_device(IN)  # texture

            cas_img_gpu[BPG, TPB](IN_global_mem, OUT_global_mem, devMax_global_mem)

            if ENABLE_CONTINUOUS_DEVICE_2_HOST:
                OUT_gpu = OUT_global_mem.copy_to_host()
            cuda.synchronize()
    except KeyboardInterrupt:
        pass

if RUN_CONTINUOUS_GPU_MULTI_STREAM:
    print("\n=================== Continuous run, multi stream ====================")

    stream_list = [numba.cuda.stream() for i in range(0, number_of_streams)]
    # print(stream_list)
    IN_global_mem_list = []
    devMax_global_mem_list = []
    OUT_global_mem_list = []

    for i in range(0, number_of_streams):
        # Host to device
        IN_global_mem_list += [cuda.to_device(IN, stream=stream_list[i])]
        devMax_global_mem_list += [cuda.to_device(numpy.array([developMaximum]), stream=stream_list[i])]
        OUT_global_mem_list += [cuda.device_array((h, w, d), stream=stream_list[i], dtype=float)]

    fps = 0
    last_time = time.time()

    try:
        while 1:
            current_time = time.time()
            if current_time - last_time < 1.0:
                if fps >= target_FPS:
                    continue
                fps += 1
            else:
                last_time = current_time
                print('\rFPS:', fps*number_of_streams, end='')
                fps = 0

            for i in range(0, number_of_streams):
                # Host to device
                if ENABLE_CONTINUOUS_HOST_2_DEVICE:
                    IN_global_mem_list[i] = cuda.to_device(IN, stream=stream_list[i])

                # Kernel
                cas_img_gpu[BPG, TPB, stream_list[i]](IN_global_mem_list[i], OUT_global_mem_list[i], devMax_global_mem_list[i])

                # Device to host
                if ENABLE_CONTINUOUS_DEVICE_2_HOST:
                    OUT_gpu = OUT_global_mem_list[i].copy_to_host()
            cuda.synchronize()

    except KeyboardInterrupt:
        pass


