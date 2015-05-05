# CUDNN

<!---
[![Build Status](https://travis-ci.org/denizyuret/CUDNN.jl.svg?branch=master)](https://travis-ci.org/denizyuret/CUDNN.jl)
--->

## Overview

This is a Julia wrapper for the NVIDIA cuDNN GPU accelerated deep
learning library.  It consists of a low level interface and a high
level interface.  The low level interface wraps each function in
libcudnn.so in a Julia function in libcudnn.jl, each data type in
cudnn.h in a Julia datatype in types.jl.  These were generated
semi-automatically using [Clang](https://github.com/ihnorton/Clang.jl)
and are well documented in the [cuDNN Library User
Guide](https://developer.nvidia.com/cuDNN).  The high level interface
introduces some Julia datatypes and provides reasonable defaults for
arguments when possible.  I will mostly describe the high level
interface below.

## Types

We introduce two data types: Tensor and Filter.  Tensors and Filters
are almost identical data structures except there is no stride option
for the filter constructor.  CUDNN docs say "Filters layout must be
contiguous in memory."  We introduce AbstractTensor as their parent
for common operations, and employ
[CudaArray](https://github.com/JuliaGPU/CUDArt.jl)'s for their data.
```
abstract AbstractTensor
immutable Tensor <: AbstractTensor; data::CudaArray; desc::cudnnTensorDescriptor_t; end
immutable Filter <: AbstractTensor; data::CudaArray; desc::cudnnFilterDescriptor_t; end
```

Tensors and Filters can be constructed using Array's, CudaArray's, or
by specifying the element type and dimensions.
```
Tensor(T::Type, dims::Dims)
Tensor(T::Type, dims::Integer...)
Tensor(a::Array)
Tensor(a::CudaArray)
```

Similar constructors also exist for Filters.  Currently only Float32
and Float64 are supported for element types, and only 4-D Tensors and
Filters are supported by the majority of CUDNN functions.  5-D Tensor
operations (to support 3-D point clouds) are under development.

The default order of Tensor dimensions in Julia are (W,H,C,N) with W
the being fastest changing dimension.  These stand for width, height,
channels, and number of images for image applications.  Note that the
C library documentation refers to this order as NCHW.  Similarly the
default order of Filter dimensions in Julia are (W,H,C,K) standing for
width, height, number of input feature maps, and number of output
feature maps respectively.

The following array operations are supported for Tensors and Filters:
`eltype`, `ndims`, size, strides, stride, zeros, ones, similar, copy.