
using CuArrays
using CUDAapi
import NNlib: conv2d, conv2d_grad_w, conv2d_grad_x, pool, pool_grad

const Cptr = Ptr{Void}
macro gs(); if false; esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))); end; end

include("init.jl")
include("const.jl")
include("descriptors.jl")
include("utils.jl")
include("conv.jl")
include("pool.jl")
include("batchnorm.jl")
# include("dropout.jl")
include("softmax.jl")
