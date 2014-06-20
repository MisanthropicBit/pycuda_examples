#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A small demo showcasing how to use pycuda."""

from __future__ import print_function

import sys
import numpy as np
import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as driver


double_kernel_source = pycuda.compiler.SourceModule("""
__global__ void double_kernel(int* const d_a, const size_t size) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = tid + blockIdx.x * blockDim.x;

    if (gid < size) {
        d_a[gid] *= 2;
    }
}""")


if __name__ == '__main__':
    array_size = 1000000

    start, end = driver.Event(), driver.Event()

    # Allocate a host array of 1 million signed 32-bit integers
    h_array = np.random.randint(1, 101, size=array_size).astype(np.int32)
    h_array_copy = h_array.copy()  # For verification

    # -------------------------------------------------------------------------

    # Transfer the host array to device memory (DeviceAllocation)
    d_array = driver.to_device(h_array)

    # -------------------------------------------------------------------------

    # Retrieve the kernel from the source module
    double_kernel = double_kernel_source.get_function('double_kernel')

    start.record()
    # Launch the kernel
    double_kernel(d_array, np.uint32(array_size),
                  grid=((array_size+256-1)//256, 1, 1), block=(256, 1, 1))
    end.record()
    end.synchronize()
    print("Took {}ms".format(start.time_till(end)))

    # Transfer the results back to host memory
    driver.memcpy_dtoh(h_array, d_array)

    # Verify the result
    assert np.all(h_array_copy * 2 == h_array)

    # Visual sanity check
    print(h_array_copy*2-h_array)
