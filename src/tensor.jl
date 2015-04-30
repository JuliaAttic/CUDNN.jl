# High level interface to CUDNN Tensors

type Tensor
    data::CudaArray
    desc::cudnnTensorDescriptor_t
    # TODO: maybe have a constructor that takes dims only
    function Tensor(a::Array)
        dt = (eltype(a) == Float64 ? CUDNN_DATA_DOUBLE :
              eltype(a) == Float32 ? CUDNN_DATA_FLOAT :
              error("Supported data types are Float32 and Float64"))
        c = CudaArray(a)
        d = cudnnTensorDescriptor_t[0]
        # TODO: We should make sure this gets deallocated properly:
        cudnnCreateTensorDescriptor(d)
        # Default order in CUDNN is n,c,h,w with w fastest changing
        # Default order in Julia is a[w,h,c,n] with w fastest changing
        # I think the reverse below gives what CUDNN expects
        cudnnSetTensorNdDescriptor(d[1], dt, ndims(a), Cint[reverse(size(a))...], Cint[reverse(strides(a))...])
        new(c, d[1])
    end
end

# Basic array functions
# TODO: Some of these create unnecessary host arrays
# TODO: Make these work with InplaceOps.jl
Base.eltype(t::Tensor)=eltype(t.data)
Base.zeros(t::Tensor)=Tensor(zeros(to_host(t)))
Base.ones(t::Tensor)=Tensor(ones(to_host(t)))
Base.similar(t::Tensor)=Tensor(to_host(t))
Base.copy(t::Tensor)=Tensor(to_host(t))
Base.copy!(dest::Tensor,src::Tensor)=cudnnTransformTensor(1, src, 0, dest)
Base.fill!(src::Tensor,value::Number)=cudnnSetTensor(src,value)
Base.scale!(src::Tensor, alpha::Number)=cudnnScaleTensor(src,alpha)

# Transfer to Julia array
CUDArt.to_host(t::Tensor)=to_host(t.data)

# Free allocated memory
CUDArt.free(t::Tensor)=(free(t.data); cudnnDestroyTensorDescriptor(t.desc))

# Read the tensor descriptor (mostly for debugging)
cudnnGetTensorNdDescriptor(t::Tensor)=cudnnGetTensorNdDescriptor(t.desc)
function cudnnGetTensorNdDescriptor(td::cudnnTensorDescriptor_t, nbDimsRequested=8)
    dataType = cudnnDataType_t[0]
    nbDims = Array(Cint, 1)
    dimA = Array(Cint, nbDimsRequested)
    strideA = Array(Cint, nbDimsRequested)
    cudnnGetTensorNdDescriptor(td,nbDimsRequested,dataType,nbDims,dimA,strideA)
    return (dataType[1], nbDims[1], dimA[1:nbDims[1]], strideA[1:nbDims[1]])
    # nbDimsRequested > 8 gives error
end

# alpha * src + beta * dest -> dest
function cudnnTransformTensor(alpha::Number, src::Tensor, beta::Number=0, dest::Tensor=ones(src))
    alphaPtr = eltype(src)[alpha]
    betaPtr = eltype(dest)[beta]
    cudnnTransformTensor(cudnnHandle, alphaPtr, src.desc, src.data.ptr, betaPtr, dest.desc, dest.data.ptr)
    return dest
end

# Refer to cudnn doc to see what different add modes do
function cudnnAddTensor(mode::cudnnAddMode_t, alpha::Number, bias::Tensor, beta::Number, src::Tensor)
    alphaPtr = eltype(bias)[alpha]
    betaPtr = eltype(src)[beta]
    cudnnAddTensor(cudnnHandle, mode, alphaPtr, bias.desc, bias.data.ptr, betaPtr, src.desc, src.data.ptr)
    return src
end

# src .= value
function cudnnSetTensor(src::Tensor, value::Number)
    valuePtr = eltype(src)[value]
    cudnnSetTensor(cudnnHandle, src.desc, src.data.ptr, valuePtr)
    return src
end

# src .*= alpha
function cudnnScaleTensor(src::Tensor, alpha::Number)
    alphaPtr = eltype(src)[alpha]
    cudnnScaleTensor(cudnnHandle, src.desc, src.data.ptr, alphaPtr)
    return src
end

