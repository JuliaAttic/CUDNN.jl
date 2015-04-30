module CUDNN
using CUDArt

const libcudnn = find_library(["libcudnn"])
isempty(libcudnn) && error("CUDNN library cannot be found")

include("libcudnn_types.jl")
include("libcudnn.jl")
include("tensor.jl")

# setup cudnn handle
cudnnhandle = cudnnHandle_t[0]
cudnnCreate(cudnnhandle)
# destroy cudnn handle at julia exit
atexit(()->cudnnDestroy(cudnnhandle[1]))

end # module
