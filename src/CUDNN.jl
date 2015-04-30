module CUDNN

const libcudnn = find_library(["libcudnn"])
isempty(libcudnn) && error("CUDNN library cannot be found")
include("libcudnn_types.jl")
include("libcudnn.jl")

end # module
