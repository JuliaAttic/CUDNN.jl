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
for common operations, and employ CudaArray's for their data.

```
abstract AbstractTensor
immutable Tensor <: AbstractTensor; data::CudaArray; desc::cudnnTensorDescriptor_t; end
immutable Filter <: AbstractTensor; data::CudaArray; desc::cudnnFilterDescriptor_t; end
```

