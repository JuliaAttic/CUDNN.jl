

using CuArrays

const Cptr = Ptr{Void}
macro gs(); if false; esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))); end; end

include("gpu.jl")
include("common.jl")
include("conv.jl")
include("pool.jl")

# See if we have a gpu at initialization:
function __init__()
    try
        r = gpu(true)        
    catch e
        gpu(false)
    end
end
