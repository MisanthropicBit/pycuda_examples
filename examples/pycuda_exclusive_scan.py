#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A small demo showcasing pycuda's built-in reduction kernel."""

import numpy as np
import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
import pycuda.scan


if __name__ == '__main__':
    array_size = 1000000

    start, end = driver.Event(), driver.Event()

    # Allocate some random numbers
    h_array = np.random.randint(1, 101, size=array_size).astype(np.uint32)
    true_sum = h_array.cumsum()-h_array  # For verification

    # Allocate device memory and transfer host array
    d_array = gpuarray.to_gpu(h_array)

    # Create a reduction kernel
    scan_kernel =\
        pycuda.scan.ExclusiveScanKernel(np.int32, neutral="0", scan_expr="a+b")

    start.record()

    # Launch the reduction kernel
    scan_kernel(d_array).get(h_array)

    end.record()
    end.synchronize()
    print("Took {}ms".format(start.time_till(end)))

    # Sanity check
    print("True sum {}\nGPU sum: {}".format(true_sum[-1], h_array[-1]))
    assert true_sum[-1] == h_array[-1]
    assert np.all(true_sum == h_array)
