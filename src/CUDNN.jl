module CUDNN
using CUDArt

const libcudnn = find_library(["libcudnn"])
isempty(libcudnn) && error("CUDNN library cannot be found")

include("libcudnn_types.jl")
include("libcudnn.jl")
include("tensor.jl")

# setup cudnn handle
cudnnHandlePtr = cudnnHandle_t[0]
cudnnCreate(cudnnHandlePtr)
cudnnHandle = cudnnHandlePtr[1]
# destroy cudnn handle at julia exit
atexit(()->cudnnDestroy(cudnnHandle))

end # module
