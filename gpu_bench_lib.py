from numba import cuda, float32, int8
from multiprocessing import Process
from pynvml import *
from PIL import Image
import numba
import numpy
import time


################################################ Algorithm ####################################

@numba.jit(nopython=True)
def cas_img_cpu(INPUT: numpy.ndarray, OUT: numpy.ndarray, developMaximum):
    h, w, d = INPUT.shape
    def reformat(IN):
        for i in range(IN.shape[0]):
            if IN[i] < 0:
                IN[i] = numpy.uint8(0)
            if IN[i] > 255:
                IN[i] = numpy.uint8(255)
            else:
                IN[i] = numpy.uint8(IN[i])
        return IN
    for row in range(1, int(h) - 1):
        for col in range(1, int(w) - 1):
            window = INPUT[row - 1:row + 2, col - 1:col + 2, :]
            min_g = min(window[0, 1, 1], window[1, 0, 1], window[1, 2, 1], window[2, 1, 1], window[1, 1, 1])
            max_g = max(window[0, 1, 1], window[1, 0, 1], window[1, 2, 1], window[2, 1, 1], window[1, 1, 1]) + 1
            BAS = 0
            d_max_g = 255 - max_g
            if d_max_g < min_g:
                BAS = d_max_g / max_g
            else:
                BAS = min_g / max_g
            _w = BAS ** 0.5 * developMaximum
            OUT[row, col, :] = \
                reformat((_w * (window[0, 1, :] + window[1, 0, :] + window[1, 2, :] + window[2, 1, :]) + window[1, 1, :]) / (_w * 4 + 1))

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

@cuda.jit()
def cas_img_gpu_optimized(INPUT: numpy.ndarray, OUT: numpy.ndarray, _developMaximum: float32, offset: int = 0):
    row, col = cuda.grid(2)

    def cas(w, a, b, c, d, e):
        res = float32(w * (a + b + c + d) + e) / float32(w * 4 + 1)
        if res > 255:
            return numpy.uint8(255)
        if res < 0:
            return numpy.uint8(0)
        return numpy.uint8(res)

    if 0 < row < INPUT.shape[0] - 1 and 0 < col < INPUT.shape[1] - 1:

        min_g = min(INPUT[row - 1, col, 1], INPUT[row, col - 1, 1], INPUT[row, col + 1, 1], INPUT[row + 1, col, 1],
                    INPUT[row, col, 1])
        max_g = max(INPUT[row - 1, col, 1], INPUT[row, col - 1, 1], INPUT[row, col + 1, 1], INPUT[row + 1, col, 1],
                    INPUT[row, col, 1]) + 1
        d_max_g = 255 - max_g

        if d_max_g < min_g:
            BAS = float32(d_max_g) / float32(max_g)
        else:
            BAS = float32(min_g) / float32(max_g)
        _w = float32(BAS ** 0.5 * _developMaximum)

        OUT[row + offset, col, 0] = cas(_w, INPUT[row - 1, col, 0], INPUT[row, col - 1, 0], INPUT[row, col + 1, 0],
                                        INPUT[row + 1, col, 0], INPUT[row, col, 0])
        OUT[row + offset, col, 1] = cas(_w, INPUT[row - 1, col, 1], INPUT[row, col - 1, 1], INPUT[row, col + 1, 1],
                                        INPUT[row + 1, col, 1], INPUT[row, col, 1])
        OUT[row + offset, col, 2] = cas(_w, INPUT[row - 1, col, 2], INPUT[row, col - 1, 2], INPUT[row, col + 1, 2],
                                        INPUT[row + 1, col, 2], INPUT[row, col, 2])

@cuda.jit()
def cas_img_gpu_optimized_shared_mem(INPUT: numpy.ndarray, OUT: numpy.ndarray, _developMaximum, offset):
    shared_INPUT = cuda.shared.array((33, 33, 3), dtype=numpy.uint8)  # allocated shared memory

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row, col = cuda.grid(2)
    # print(tx, ty, row, col)

    def cas(w, a, b, c, d, e):
        res = float32(w * (a + b + c + d) + e) / float32(w * 4 + 1)
        if res > 255:
            return numpy.uint8(255)
        if res < 0:
            return numpy.uint8(0)
        return numpy.uint8(res)

    if 0 <= row < INPUT.shape[0] and 0 <= col < INPUT.shape[1]:
        shared_INPUT[tx, ty, 0] = INPUT[row, col, 0]
        shared_INPUT[tx, ty, 1] = INPUT[row, col, 1]
        shared_INPUT[tx, ty, 2] = INPUT[row, col, 2]
        cuda.syncthreads()  # waiting for copy to shared mem

    if 0 < row < INPUT.shape[0] - 1 and 0 < col < INPUT.shape[1] - 1:

        min_g = min(shared_INPUT[tx - 1, ty, 1], shared_INPUT[tx, ty - 1, 1], shared_INPUT[tx, ty + 1, 1],
                    shared_INPUT[tx + 1, ty, 1],
                    shared_INPUT[tx, ty, 1])
        max_g = max(shared_INPUT[tx - 1, ty, 1], shared_INPUT[tx, ty - 1, 1], shared_INPUT[tx, ty + 1, 1],
                    shared_INPUT[tx + 1, ty, 1],
                    shared_INPUT[tx, ty, 1]) + 1
        d_max_g = 255 - max_g

        if d_max_g < min_g:
            BAS = float32(d_max_g) / float32(max_g)
        else:
            BAS = float32(min_g) / float32(max_g)
        _w = float32(BAS ** 0.5 * _developMaximum)

        OUT[row + offset, col, 0] = cas(_w, shared_INPUT[tx - 1, ty, 0], shared_INPUT[tx, ty - 1, 0],
                                        shared_INPUT[tx, ty + 1, 0], shared_INPUT[tx + 1, ty, 0],
                                        shared_INPUT[tx, ty, 0])
        OUT[row + offset, col, 1] = cas(_w, shared_INPUT[tx - 1, ty, 1], shared_INPUT[tx, ty - 1, 1],
                                        shared_INPUT[tx, ty + 1, 1], shared_INPUT[tx + 1, ty, 1],
                                        shared_INPUT[tx, ty, 1])
        OUT[row + offset, col, 2] = cas(_w, shared_INPUT[tx - 1, ty, 2], shared_INPUT[tx, ty - 1, 2],
                                        shared_INPUT[tx, ty + 1, 2], shared_INPUT[tx + 1, ty, 2],
                                        shared_INPUT[tx, ty, 2])

################################################ Utility ####################################


def image_read(input_img, to_float=True):
    img = numpy.ascontiguousarray(numpy.array(Image.open(input_img))[:, :, :3])
    h, w, d = img.shape

    if to_float:
        IN = img / 255.0  # normalize the image, float64 now
        MegaBytes_per_frame = h * w * d * 64 / (8 * 1024 * 1024)
    else:
        IN = numpy.array(img, dtype='uint8')
        MegaBytes_per_frame = h * w * d / (1024 * 1024)

    print('read texture:', IN.shape, IN.dtype)
    print(MegaBytes_per_frame, 'MB of data per raw frame')
    # print(MegaBytes_per_frame * target_FPS, 'MB of data per second')

    return IN


def read_utilization(handler):
    gpu_uti = nvmlDeviceGetUtilizationRates(handler)
    mem_info = nvmlDeviceGetMemoryInfo(handler)
    return gpu_uti.gpu, gpu_uti.memory, mem_info.used


def cpu_run(IN, sharpness_pref=0.8):
    developMaximum = -0.125 - (sharpness_pref * (0.2 - 0.125))

    print("\n - Starting in CPU")

    # Starting in CPU
    OUT_cpu = numpy.zeros(IN.shape, dtype=numpy.uint8)
    cpu_start = time.time()
    cas_img_cpu(IN, OUT_cpu, developMaximum)
    cpu_end = time.time()
    print("CPU single image CAS time: " + str(cpu_end - cpu_start))
    return OUT_cpu


def gpu_single_run(IN, gpu_device_selected=0, sharpness_pref=0.8, TPB=(16, 16)):
    print("\n - Starting in GPU")
    developMaximum = float32(-0.125 - (sharpness_pref * (0.2 - 0.125)))

    h, w, d = IN.shape
    BPG = (int(numpy.ceil(h / TPB[0])), int(numpy.ceil(w / TPB[1])))

    OUT_gpu = numpy.empty((h, w, d), dtype='uint8')

    # Starting in GPU
    cuda.select_device(gpu_device_selected)


    # host to device copy
    copy_host_2_device_start = time.time()
    # Pin memory
    cuda.pinned([IN, OUT_gpu])
    IN_global_mem = cuda.to_device(IN)  # texture

    OUT_global_mem = cuda.device_array((h, w, d), dtype='uint8')
    copy_host_2_device_end = time.time()

    print('host to device copy time:', copy_host_2_device_end - copy_host_2_device_start)

    cas_img_gpu_optimized_shared_mem[BPG, TPB](IN_global_mem, OUT_global_mem, developMaximum, 0)
    cuda.synchronize()

    gpu_cas_start = time.time()
    cas_img_gpu_optimized_shared_mem[BPG, TPB](IN_global_mem, OUT_global_mem, developMaximum, 0)
    cuda.synchronize()
    gpu_cas_end = time.time()
    print('GPU single image CAS time:', gpu_cas_end - gpu_cas_start)

    copy_device_2_host_start = time.time()
    OUT_global_mem.copy_to_host(OUT_gpu)
    copy_device_2_host_end = time.time()
    print('device to host copy time:', copy_device_2_host_end - copy_device_2_host_start)

    return OUT_gpu


def gpu_continuous_run_single_stream(IN, gpu_device_selected=0, sharpness_pref=0.8, TPB=(16, 16), target_FPS=1000,
                                     ENABLE_CONTINUOUS_HOST_2_DEVICE=False, ENABLE_CONTINUOUS_DEVICE_2_HOST=False,
                                     timeout=-1):
    print("\n=================== Continuous run, single stream ====================")

    print('ENABLE_CONTINUOUS_HOST_2_DEVICE', ENABLE_CONTINUOUS_HOST_2_DEVICE, 'ENABLE_CONTINUOUS_DEVICE_2_HOST',
          ENABLE_CONTINUOUS_DEVICE_2_HOST)

    developMaximum = -0.125 - (sharpness_pref * (0.2 - 0.125))

    h, w, d = IN.shape
    BPG = (int(numpy.ceil(h / TPB[0])), int(numpy.ceil(w / TPB[1])))

    # Starting in GPU
    cuda.select_device(gpu_device_selected)

    IN_global_mem = cuda.to_device(IN)  # texture
    devMax_global_mem = cuda.to_device(numpy.array([developMaximum]))
    OUT_global_mem = cuda.device_array((h, w, d), dtype=float)

    OUT_gpu = None

    fps = 0
    avg_fps = 0
    last_time = time.time()
    init_time = last_time
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
                avg_fps += fps
                fps = 0
                if current_time - init_time > timeout > 1:
                    OUT_gpu = OUT_global_mem.copy_to_host()
                    numpy.where(OUT_gpu < 1, numpy.where(OUT_gpu > 0, OUT_gpu, 0.0), 1.0)
                    avg_fps /= int(current_time - init_time)
                    raise TimeoutError

            if ENABLE_CONTINUOUS_HOST_2_DEVICE:
                IN_global_mem = cuda.to_device(IN)  # texture

            cas_img_gpu[BPG, TPB](IN_global_mem, OUT_global_mem, devMax_global_mem)

            if ENABLE_CONTINUOUS_DEVICE_2_HOST:
                OUT_gpu = OUT_global_mem.copy_to_host()
                numpy.where(OUT_gpu < 1, numpy.where(OUT_gpu > 0, OUT_gpu, 0.0), 1.0)

            cuda.synchronize()
    except KeyboardInterrupt:
        pass
    except TimeoutError:
        print('Average FPS:', avg_fps, 'in', timeout, 's running time')
        pass
    return OUT_gpu


def gpu_continuous_run_multi_stream(IN, gpu_device_selected=0, sharpness_pref=0.8, TPB=(16, 16), number_of_streams=5,
                                    target_FPS=1000, ENABLE_CONTINUOUS_HOST_2_DEVICE=False,
                                    ENABLE_CONTINUOUS_DEVICE_2_HOST=False, timeout=-1, nvml_handler=None):
    print("\n=================== Continuous run, multi stream ====================")

    print('ENABLE_CONTINUOUS_HOST_2_DEVICE', ENABLE_CONTINUOUS_HOST_2_DEVICE, 'ENABLE_CONTINUOUS_DEVICE_2_HOST',
          ENABLE_CONTINUOUS_DEVICE_2_HOST)

    developMaximum = -0.125 - (sharpness_pref * (0.2 - 0.125))
    h, w, d = IN.shape

    stream_list = [numba.cuda.stream() for i in range(0, number_of_streams)]

    slice_height = int(numpy.ceil(h / number_of_streams))
    print('stream count:', number_of_streams, 'slice height:', slice_height)

    BPG = (int(numpy.ceil(slice_height / TPB[0])), int(numpy.ceil(w / TPB[1])))

    # Starting in GPU
    cuda.select_device(gpu_device_selected)

    # print(stream_list)
    IN_global_mem_list = []
    devMax_global_mem_list = []
    OUT_global_mem_list = []
    OUT_gpu_list = []

    for i in range(0, number_of_streams):
        # Host to device
        slice_start = i * slice_height - 1 if i != 0 else 0
        if (1 + i) * slice_height < h:
            slice_end = (1 + i) * slice_height + 1
        else:
            slice_end = h
        IN_global_mem_list += [cuda.to_device(IN[slice_start:slice_end, :, :], stream=stream_list[i])]
        devMax_global_mem_list += [cuda.to_device(numpy.array([developMaximum]), stream=stream_list[i])]
        OUT_global_mem_list += [cuda.device_array((slice_end - slice_start, w, d), stream=stream_list[i], dtype=float)]
        OUT_gpu_list += [numpy.zeros((slice_end - slice_start, w, d), dtype=float)]

    fps = 0
    avg = [0, 0, 0, 0]
    last_time = time.time()
    init_time = last_time

    try:
        while 1:
            current_time = time.time()
            if current_time - last_time < 1.0:
                if fps >= target_FPS:
                    continue
                fps += 1
                avg[0] += 1
            else:
                last_time = current_time
                if nvml_handler is not None:
                    gpu_usage, mem_usage, mem_used = read_utilization(nvml_handler)
                    mem_used = numpy.round(mem_used / 1024 / 1024, 3)
                    print('\rFPS: {}, GPU usage {}%, mem usage {}%, mem used {}MB'
                          .format(fps, gpu_usage, mem_usage, mem_used), end='')
                    avg[1] += gpu_usage
                    avg[2] += mem_usage
                    avg[3] += mem_used
                else:
                    print('\rFPS:', fps, end='')
                fps = 0
                if current_time - init_time > timeout > 1:
                    avg = numpy.round(numpy.array(avg) / (current_time - init_time), 3)
                    for i in range(0, number_of_streams):
                        OUT_gpu = OUT_global_mem_list[i].copy_to_host()
                        OUT_gpu_list[i] = numpy.where(OUT_gpu < 1, numpy.where(OUT_gpu > 0, OUT_gpu, 0.0), 1.0)[
                                          1:-1, :, :]
                    raise TimeoutError

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
                cas_img_gpu[BPG, TPB, stream_list[i]](IN_global_mem_list[i], OUT_global_mem_list[i],
                                                      devMax_global_mem_list[i])

                # Device to host
                if ENABLE_CONTINUOUS_DEVICE_2_HOST:
                    OUT_gpu = OUT_global_mem_list[i].copy_to_host()
                    OUT_gpu_list[i] = numpy.where(OUT_gpu < 1, numpy.where(OUT_gpu > 0, OUT_gpu, 0.0), 1.0)[1:-1, :, :]

            cuda.synchronize()

    except KeyboardInterrupt:
        pass
    except TimeoutError:
        print('\rAverage FPS: {}, GPU usage {}%, mem usage {}%, mem used {}MB in {}s running time'
              .format(avg[0], avg[1], avg[2], avg[3], timeout))
        pass

    return numpy.vstack(OUT_gpu_list)

def gpu_continuous_run_multi_stream_new(IN, gpu_device_selected=0, sharpness_pref=0.8, TPB=(16, 16),
                                        number_of_streams=5,
                                        target_FPS=1000, ENABLE_CONTINUOUS_HOST_2_DEVICE=False,
                                        ENABLE_CONTINUOUS_DEVICE_2_HOST=False, timeout=-1, nvml_handler=None):
    print("\n=================== Continuous run, multi stream ====================")

    print('ENABLE_CONTINUOUS_HOST_2_DEVICE', ENABLE_CONTINUOUS_HOST_2_DEVICE, 'ENABLE_CONTINUOUS_DEVICE_2_HOST',
          ENABLE_CONTINUOUS_DEVICE_2_HOST)



    developMaximum = float32(-0.125 - (sharpness_pref * (0.2 - 0.125)))
    h, w, d = IN.shape

    slice_height = int(numpy.floor(h / number_of_streams)) + 2
    print('stream count:', number_of_streams, 'slice height:', slice_height)

    BPG = (int(numpy.ceil(slice_height / TPB[0])), int(numpy.ceil(w / TPB[1])))

    # Starting in GPU
    cuda.select_device(gpu_device_selected)

    # print(stream_list)
    stream_list = []

    IN_global_mem_list = []
    cuda.pinned(IN)

    OUT_global_mem = cuda.device_array((h, w, d), dtype='uint8')
    OUT_gpu = numpy.empty((h, w, d), dtype=numpy.uint8)
    cuda.pinned(OUT_gpu)

    # Host to device
    for i in range(0, number_of_streams):
        stream_list += [cuda.stream()]
        slice_start = i * slice_height - 1 if i != 0 else 0
        if (1 + i) * slice_height < h:
            slice_end = (1 + i) * slice_height + 1
        else:
            slice_end = h
        IN_global_mem_list += [cuda.to_device(IN[slice_start:slice_end, :, :], stream=stream_list[i])]

    fps = 0
    avg = [0, 0, 0, 0]
    last_time = time.time()
    init_time = last_time

    try:
        while 1:
            current_time = time.time()
            if current_time - last_time < 1.0:
                if fps >= target_FPS:
                    continue
                fps += 1
                avg[0] += 1
            else:
                last_time = current_time
                if nvml_handler is not None:
                    gpu_usage, mem_usage, mem_used = read_utilization(nvml_handler)
                    mem_used = numpy.round(mem_used / 1024 / 1024, 3)
                    print('\rFPS: {}, GPU usage {}%, mem usage {}%, mem used {}MB'
                          .format(fps, gpu_usage, mem_usage, mem_used), end='')
                    avg[1] += gpu_usage
                    avg[2] += mem_usage
                    avg[3] += mem_used
                else:
                    print('\rFPS:', fps, end='')
                fps = 0
                if current_time - init_time > timeout > 1:
                    avg = numpy.round(numpy.array(avg) / (current_time - init_time), 3)
                    OUT_gpu = OUT_global_mem.copy_to_host()
                    raise TimeoutError

            for i in range(0, number_of_streams):

                slice_start = i * slice_height - 1 if i != 0 else 0
                if (1 + i) * slice_height < h:
                    slice_end = (1 + i) * slice_height + 1
                else:
                    slice_end = h

                # Host to device
                if ENABLE_CONTINUOUS_HOST_2_DEVICE:
                    # Host to device
                    cuda.to_device(IN[slice_start:slice_end, :, :], stream=stream_list[i], to=IN_global_mem_list[i])

                # Kernel
                cas_img_gpu_optimized_shared_mem[BPG, TPB, stream_list[i]](IN_global_mem_list[i], OUT_global_mem, developMaximum,
                                                                slice_start)

            cuda.synchronize()

            # Device to host
            if ENABLE_CONTINUOUS_DEVICE_2_HOST:
                OUT_global_mem.copy_to_host(OUT_gpu)

    except KeyboardInterrupt:
        pass
    except TimeoutError:
        print('\rAverage FPS: {}, GPU usage {}%, mem usage {}%, mem used {}MB in {}s running time'
              .format(avg[0], avg[1], avg[2], avg[3], timeout))
        pass

    return OUT_gpu


def run_continuous_copy(DUMMY_DATA_SHAPE=None, target_FPS=1000, ENABLE_CONTINUOUS_DEVICE_2_HOST=True,
                        ENABLE_CONTINUOUS_HOST_2_DEVICE=False, timeout=-1):
    if DUMMY_DATA_SHAPE is None:
        DUMMY_DATA_SHAPE = [1920, 1080, 600]

    print("\n=================== Continuous copy ====================")

    print('ENABLE_CONTINUOUS_HOST_2_DEVICE', ENABLE_CONTINUOUS_HOST_2_DEVICE, 'ENABLE_CONTINUOUS_DEVICE_2_HOST',
          ENABLE_CONTINUOUS_DEVICE_2_HOST)

    print('copy data shape:', DUMMY_DATA_SHAPE, 'size:',
          numpy.round(DUMMY_DATA_SHAPE[0] * DUMMY_DATA_SHAPE[1] * DUMMY_DATA_SHAPE[2] / (1024 * 1024), 3), 'MB/frame')

    if ENABLE_CONTINUOUS_HOST_2_DEVICE:
        dummy_in_data = numpy.full(DUMMY_DATA_SHAPE, 3, dtype=numpy.uint8)
        cuda.pinned(dummy_in_data)
        dummy_in_data_global_mem = cuda.device_array(DUMMY_DATA_SHAPE, dtype=numpy.uint8)
        h2d_stream = cuda.stream()

    if ENABLE_CONTINUOUS_DEVICE_2_HOST:
        dummy_out_data_global_mem = cuda.device_array(DUMMY_DATA_SHAPE, dtype=numpy.uint8)
        dummy_out_data = numpy.empty(shape=DUMMY_DATA_SHAPE, dtype=numpy.uint8)
        cuda.pinned(dummy_out_data)
        d2h_stream = cuda.stream()

    fps = 0
    avg = [0, 0, 0, 0]
    last_time = time.time()
    init_time = last_time

    try:
        while 1:
            current_time = time.time()
            if current_time - last_time < 1.0:
                if fps >= target_FPS:
                    continue
                fps += 1
                avg[0] += 1
            else:
                last_time = current_time
                print('\rFPS:', fps, end='')
                fps = 0
                if current_time - init_time > timeout > 1:
                    avg = numpy.round(numpy.array(avg) / (current_time - init_time), 3)
                    raise TimeoutError

            # Host to device
            if ENABLE_CONTINUOUS_HOST_2_DEVICE:
                cuda.to_device(dummy_in_data, stream=h2d_stream, to=dummy_in_data_global_mem)

            # Device to host
            if ENABLE_CONTINUOUS_DEVICE_2_HOST:
                dummy_out_data_global_mem.copy_to_host(dummy_out_data, stream=d2h_stream)

            # cuda.synchronize()
    except KeyboardInterrupt:
        pass
    except TimeoutError:
        print('\rAverage FPS: {} in {}s running time'
              .format(avg[0], timeout))
    pass

def copy(is_h2d=False, is_d2h=False, DUMMY_DATA_SHAPE=None, target_FPS=None, timeout=None):
    print(cuda.devices.get_context())
    if is_h2d:
        dummy_in_data = numpy.full(DUMMY_DATA_SHAPE, 3, dtype=numpy.uint8)
        cuda.pinned(dummy_in_data)
        dummy_in_data_global_mem = cuda.device_array(DUMMY_DATA_SHAPE, dtype=numpy.uint8)
        h2d_stream = cuda.stream()

    if is_d2h:
        dummy_out_data_global_mem = cuda.to_device(numpy.full(DUMMY_DATA_SHAPE, 3, dtype=numpy.uint8))
        dummy_out_data = numpy.empty(shape=DUMMY_DATA_SHAPE, dtype=numpy.uint8)
        cuda.pinned(dummy_out_data)
        d2h_stream = cuda.stream()

    fps = 0
    avg = 0
    last_time = time.time()
    init_time = last_time
    try:
        while 1:
            current_time = time.time()
            if current_time - last_time < 1.0:
                if fps >= target_FPS:
                    continue
                fps += 1
                avg += 1
            else:
                last_time = current_time
                print('\rh2d' if is_h2d else '\rd2h', 'FPS:', fps, end='')
                fps = 0
                if current_time - init_time > timeout > 1:
                    avg = numpy.round(numpy.array(avg) / (current_time - init_time), 3)
                    raise TimeoutError
            if is_h2d:
                cuda.to_device(dummy_in_data, stream=h2d_stream, to=dummy_in_data_global_mem)
            if is_d2h:
                dummy_out_data_global_mem.copy_to_host(dummy_out_data, stream=d2h_stream)
    except TimeoutError:
        print('\nh2d' if is_h2d else '\nd2h', 'Average FPS: {} in {}s running time'.format(avg, timeout))


def run_continuous_copy_new(DUMMY_DATA_SHAPE=None, target_FPS=1000, ENABLE_CONTINUOUS_DEVICE_2_HOST=True,
                        ENABLE_CONTINUOUS_HOST_2_DEVICE=False, timeout=-1):

    if DUMMY_DATA_SHAPE is None:
        DUMMY_DATA_SHAPE = [1920, 1080, 600]

    print("\n=================== Continuous copy ====================")

    print('ENABLE_CONTINUOUS_HOST_2_DEVICE', ENABLE_CONTINUOUS_HOST_2_DEVICE, 'ENABLE_CONTINUOUS_DEVICE_2_HOST',
          ENABLE_CONTINUOUS_DEVICE_2_HOST)

    print('copy data shape:', DUMMY_DATA_SHAPE, 'size:',
          numpy.round(DUMMY_DATA_SHAPE[0] * DUMMY_DATA_SHAPE[1] * DUMMY_DATA_SHAPE[2] / (1024 * 1024), 3), 'MB/frame')

    if ENABLE_CONTINUOUS_HOST_2_DEVICE:
        h2d_t = Process(target=copy, args=(True, False, DUMMY_DATA_SHAPE, target_FPS, timeout))
        h2d_t.start()

    if ENABLE_CONTINUOUS_DEVICE_2_HOST:
        d2h_t = Process(target=copy, args=(False, True, DUMMY_DATA_SHAPE, target_FPS, timeout))
        d2h_t.start()

    if ENABLE_CONTINUOUS_HOST_2_DEVICE:
        h2d_t.join(timeout+2)
    if ENABLE_CONTINUOUS_DEVICE_2_HOST:
        d2h_t.join(timeout+2)