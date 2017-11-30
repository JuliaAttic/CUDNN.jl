

const toolkit = CUDAapi.find_toolkit()
const libcudnn = CUDAapi.find_library("cudnn", toolkit)
const Cptr = Ptr{Void}

macro cuda(lib,fun,x...)        # give an error if library missing, or if error code!=0
    if Libdl.find_library(["lib$lib"], []) != ""
        fx = Expr(:call, :ccall, ("$fun","lib$lib"), :UInt32, x...)        
        msg = "$lib.$fun error "
        err = gensym()
        # esc(:(if ($err=$fx) != 0; warn($msg, $err); Base.show_backtrace(STDOUT, backtrace()); end))
        esc(:(if ($err=$fx) != 0; error($msg, $err); end; @gs))
    else
        Expr(:call,:error,"Cannot find lib$lib, please install it and rerun Pkg.build(\"Knet\").")
    end
end

macro cuda1(lib,fun,x...)       # return -1 if library missing, error code if run
    if Libdl.find_library(["lib$lib"], []) != ""
        fx = Expr(:call, :ccall, ("$fun","lib$lib"), :UInt32, x...)
        err = gensym()
        esc(:($err=$fx; @gs; $err))
    else
        -1
    end
end



const CUDNN_VERSION = Ref{Int}(-1)

function cudnn_version()
    if CUDNN_VERSION[] == -1
        CUDNN_VERSION[] = Int(ccall((:cudnnGetVersion,:libcudnn),Csize_t,()))
    end
    return CUDNN_VERSION[]
end



const CUDNN_HANDLES = Array{Ptr{Void}}(0)


function cudnn_create_handle()
    handleP = Cptr[0]
    @cuda(cudnn, cudnnCreate, (Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cuda(cudnn,cudnnDestroy,(Cptr,), handle))    
    return handle
end


# TODO: handle multiple GPUs
function cudnnhandle()
    if isempty(CUDNN_HANDLES)
        handle = cudnn_create_handle()
        push!(CUDNN_HANDLES, handle) 
    end
    return CUDNN_HANDLES[1]
end
