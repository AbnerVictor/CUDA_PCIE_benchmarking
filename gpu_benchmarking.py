from numba import cuda, float32
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numba
import numpy
import time

######################################## PARAMETERS ############################################

gpu_device_selected = 0

target_FPS = 1000

input_img = 'test_img_UHD.png'

# CAS parameters
sharpness_pref = 0.8
developMaximum = -0.125 - (sharpness_pref * (0.2 - 0.125))

TPB = (16, 16)  # threads per warp, strongly depends on gpu architecture

RUN_SINGLE_CPU = False
RUN_SINGLE_GPU = False

ENABLE_CONTINUOUS_HOST_2_DEVICE = False
ENABLE_CONTINUOUS_DEVICE_2_HOST = False

RUN_CONTINUOUS_GPU_SINGLE_STREAM = False
RUN_CONTINUOUS_GPU_MULTI_STREAM = True
number_of_streams = 1

RUN_CONTINUOUS_COPY = False
DUMMY_DATA_SHAPE = [1920, 1080, number_of_streams*3]
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


######################################## WORK ############################################

print("\n=================== task info ====================")

img = numpy.ascontiguousarray(numpy.array(Image.open(input_img))[:, :, :3])
IN = img / 255.0  # normalize the image, float64 now

h, w, d = IN.shape

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
                numpy.where(OUT_gpu < 1, numpy.where(OUT_gpu > 0, OUT_gpu, 0.0), 1.0)

            cuda.synchronize()
    except KeyboardInterrupt:
        pass

if RUN_CONTINUOUS_GPU_MULTI_STREAM:
    print("\n=================== Continuous run, multi stream ====================")

    stream_list = [numba.cuda.stream() for i in range(0, number_of_streams)]

    slice_height = int(numpy.ceil(h / number_of_streams))
    print('stream count:', number_of_streams, 'slice height:', slice_height)

    BPG = (int(numpy.ceil(slice_height / TPB[0])), int(numpy.ceil(w / TPB[1])))

    # print(stream_list)
    IN_global_mem_list = []
    devMax_global_mem_list = []
    OUT_global_mem_list = []
    OUT_gpu_list = []

    for i in range(0, number_of_streams):
        # Host to device
        slice_start = i * slice_height - 1 if i != 0 else 0
        if (1+i) * slice_height < h:
            slice_end = (1+i) * slice_height + 1
        else:
            slice_end = h
        IN_global_mem_list += [cuda.to_device(IN[slice_start:slice_end, :, :], stream=stream_list[i])]
        devMax_global_mem_list += [cuda.to_device(numpy.array([developMaximum]), stream=stream_list[i])]
        OUT_global_mem_list += [cuda.device_array((slice_end-slice_start, w, d), stream=stream_list[i], dtype=float)]
        OUT_gpu_list += [numpy.zeros((slice_end-slice_start, w, d), dtype=float)]

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
                print('\rFPS:', fps, end='')
                fps = 0

            for i in range(0, number_of_streams):
                # Host to device
                if ENABLE_CONTINUOUS_HOST_2_DEVICE:
                    # Host to device
                    slice_start = i * slice_height - 1 if i != 0 else 0
                    if slice_start + slice_height < h:
                        slice_end = slice_start + slice_height + 1
                    else:
                        slice_end = h
                    IN_global_mem_list[i] = cuda.to_device(IN[slice_start:slice_end, :, :], stream=stream_list[i])

                # Kernel
                cas_img_gpu[BPG, TPB, stream_list[i]](IN_global_mem_list[i], OUT_global_mem_list[i], devMax_global_mem_list[i])

                # Device to host
                if ENABLE_CONTINUOUS_DEVICE_2_HOST:
                    OUT_gpu = OUT_global_mem_list[i].copy_to_host()
                    OUT_gpu_list[i] = numpy.where(OUT_gpu < 1, numpy.where(OUT_gpu > 0, OUT_gpu, 0.0), 1.0)[1:-1, :, :]

            cuda.synchronize()

            # for i in range(0, number_of_streams):
            #     print(OUT_gpu_list[i].shape)
            #     plt.imshow(OUT_gpu_list[i])
            #     plt.show()

            # plt.imshow(numpy.vstack(OUT_gpu_list))
            # plt.show()
            # mpimg.imsave('OUT_mul_stream_GPU.png', numpy.vstack(OUT_gpu_list))
            # break
    except KeyboardInterrupt:
        pass


if RUN_CONTINUOUS_COPY:
    print("\n=================== Continuous copy ====================")
    print('copy data shape:', DUMMY_DATA_SHAPE, 'size:', numpy.round(DUMMY_DATA_SHAPE[0]*DUMMY_DATA_SHAPE[1]*DUMMY_DATA_SHAPE[2]/(1024*1024), 3), 'MB/frame')
    dummy_in_data = numpy.full(DUMMY_DATA_SHAPE, 255, dtype='uint8')
    if ENABLE_CONTINUOUS_DEVICE_2_HOST:
        dummy_out_data_global_mem = cuda.device_array(DUMMY_DATA_SHAPE, dtype=numpy.uint8)
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
                print('\rFPS:', fps, end='')
                fps = 0

            # Host to device
            if ENABLE_CONTINUOUS_HOST_2_DEVICE:
                dummy_in_data_global_mem = cuda.to_device(dummy_in_data)

            # Device to host
            if ENABLE_CONTINUOUS_DEVICE_2_HOST:
                dummy_out_data = dummy_out_data_global_mem.copy_to_host()

            cuda.synchronize()

    except KeyboardInterrupt:
        pass
