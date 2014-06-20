PyCUDA Examples
===============

A collection of PyCUDA examples, originally made for `PyCON DK <http://pycon.dk/>`_.
Refer to the `PyCUDA wiki page <http://wiki.tiker.net/PyCuda/Examples>`_ for more examples.

Examples with explanations:
---------------------------

``numpy_simple.py``
    Double the values in a signed integer array (CPU performance reference)

``pycuda_simple1.py``
    Double the values in a signed integer array using explicit memory allocations and transfers.

``pycuda_simple2.py``
    Same as ``pycuda_simple1.py``, but using ``pycuda.driver`` functions for memory transfers.

``pycuda_simple3.py``
    Same as ``pycuda_simple1-2.py``, but using ``pycuda.driver.InOut``.

``pycuda_gpuarray.py``
    Same as the three previous examples, using ``pycuda.gpuarray.gpuarray``.

``pycuda_reduction.py``
    Maximum-reduction of an array using ``pycuda.reduction.ReductionKernel``.

``pycuda_exclusive_scan.py``
    Perform an exclusive scan on an array using ``pycuda.scan.ExclusiveScanKernel``.
