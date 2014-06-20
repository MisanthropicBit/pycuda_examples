#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Doubling the integers in a numpy array."""

from __future__ import print_function

import sys
import time
import numpy as np


if __name__ == '__main__':
    array_size = 10000000
    array = np.random.randint(1, 101, size=array_size)
    timer = time.clock if sys.platform.startswith('win') else time.time

    start = timer()
    result = array * 2
    end = timer()

    print("Execution time: {}ms".format((end-start)*1000.0))

    # Verify the result
    assert np.all(array * 2 == result)

    # Visual sanity check
    print(array*2-result)
