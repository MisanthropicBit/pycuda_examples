#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A small demo showcasing pycuda's built-in reduction kernel."""

import numpy as np
import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
import pycuda.reduction


if __name__ == '__main__':
    array_size = 10e6

    start, end = driver.Event(), driver.Event()

    # Allocate some random numbers
    h_array = np.random.randint(0, array_size, size=array_size)\
        .astype(np.int32)
    true_max = h_array.max()  # For verification

    # Allocate device memory and transfer host array
    d_array = gpuarray.to_gpu(h_array)

    # Create a reduction kernel
    reduction_kernel =\
        pycuda.reduction.ReductionKernel(np.int32, neutral="-1",
                                         reduce_expr="max(a, b)",
                                         map_expr="in[i]",
                                         arguments="int* const in")

    start.record()

    # Launch the reduction kernel
    gpu_max = reduction_kernel(d_array).get()

    end.record()
    end.synchronize()
    print("Took {}ms".format(start.time_till(end)))

    # Sanity check
    print("True max is {}\nGPU max is {}".format(true_max, gpu_max))
    assert gpu_max == true_max
