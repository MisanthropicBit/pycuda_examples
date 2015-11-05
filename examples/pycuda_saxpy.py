#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A small demo showcasing a saxpy computation."""

from __future__ import print_function

import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray

saxpy_kernel = """__global__ void saxpy_kernel(const float* const x,
                                               const float* const y,
                                               float a,
                                               float* const result,
                                               size_t size) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = tid + blockIdx.x * blockDim.x;

    if (gid < size) {
        result[gid] = a * x[gid] + y[gid];
    }
}"""

if __name__ == '__main__':
    array_size = 10e5

    start, end = driver.Event(), driver.Event()

    # Same host-side allocation as last time
    h_x = np.random.uniform(1., 101., size=array_size).astype(np.float32)
    h_y = np.random.uniform(1., 101., size=array_size).astype(np.float32)
    h_result = np.empty_like(h_y)
    h_a = np.float32(0.234)

    # Transfer host array to device memory
    d_x = gpuarray.to_gpu(h_x)
    d_y = gpuarray.to_gpu(h_y)
    d_result = gpuarray.to_gpu(d_y)

    # Launch implicit kernel and retrieve result in one line
    start.record()
    d_result = h_a * d_x + d_y
    end.record()
    end.synchronize()
    print("Took {}ms".format(start.time_till(end)))

    d_result.get(h_result)

    # Verify the result
    assert np.all(h_a * h_x + h_y == h_result)
