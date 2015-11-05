#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A small demo showcasing pycuda.gpuarray objects."""

from __future__ import print_function

import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray


if __name__ == '__main__':
    array_size = 10e5

    start, end = driver.Event(), driver.Event()

    # Same host-side allocation as last time
    h_array = np.random.randint(1, 101, size=array_size).astype(np.uint32)
    h_array_copy = h_array.copy()  # For verification

    # Transfer host array to device memory
    d_array = gpuarray.to_gpu(h_array)

    # Launch implicit kernel and retrieve result in one line
    start.record()
    d_array = 2 * d_array
    end.record()
    end.synchronize()
    print("Took {}ms".format(start.time_till(end)))

    d_array.get(h_array)

    # Verify the result
    assert np.all(h_array_copy * 2 == h_array)

    # Visual sanity check
    print(h_array_copy*2-h_array)
