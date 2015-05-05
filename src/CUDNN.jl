module CUDNN
using CUDArt

const libcudnn = find_library(["libcudnn"])
isempty(libcudnn) && error("CUDNN library cannot be found")

# These are semi-automatically generated using Clang with wrap_cudnn.jl:
include("types.jl")
include("libcudnn.jl")

# Setup cudnn handle
cudnnHandlePtr = cudnnHandle_t[0]
cudnnCreate(cudnnHandlePtr)
cudnnHandle = cudnnHandlePtr[1]
# destroy cudnn handle at julia exit
atexit(()->cudnnDestroy(cudnnHandle))

### High level interface to CUDNN:

# Datatypes: AbstractTensor, Tensor, Filter, PoolingDescription

abstract AbstractTensor

immutable Tensor <: AbstractTensor
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

# A Filter is identical to a Tensor as a data structure.  The cudnn
# constructors are the same except there is no stride option for
# filters.  Docs say "Filters layout must be contiguous in memory."

immutable Filter <: AbstractTensor
    data::CudaArray
    desc::cudnnFilterDescriptor_t
    function Filter(a::Array)
        dt = (eltype(a) == Float64 ? CUDNN_DATA_DOUBLE :
              eltype(a) == Float32 ? CUDNN_DATA_FLOAT :
              error("Supported data types are Float32 and Float64"))
        c = CudaArray(a)
        d = cudnnFilterDescriptor_t[0]
        cudnnCreateFilterDescriptor(d)
        cudnnSetFilterNdDescriptor(d[1], dt, ndims(a), Cint[reverse(size(a))...])
        new(c, d[1])
    end
end

# Basic array functions
# TODO: Some of these create unnecessary host arrays, need arrayless constructor.
# TODO: Make these work with InplaceOps.jl
Base.eltype(t::AbstractTensor)=eltype(t.data)
Base.ndims(t::AbstractTensor)=ndims(t.data)
Base.size(t::AbstractTensor)=size(t.data)
Base.strides(t::AbstractTensor)=strides(to_host(t)) # CUDArt does not have strides?
Base.size(t::AbstractTensor,n)=size(t.data,n)
Base.stride(t::AbstractTensor,n)=stride(t.data,n)
Base.zeros{T<:AbstractTensor}(t::T)=T(zeros(eltype(t), size(t)))
Base.ones{T<:AbstractTensor}(t::T)=T(ones(eltype(t), size(t)))
Base.similar{T<:AbstractTensor}(t::T)=T(Array(eltype(t), size(t)))
Base.copy{T<:AbstractTensor}(t::T)=T(to_host(t))
# TODO: these should be available for Filters as well:
Base.copy!(dest::Tensor,src::Tensor)=cudnnTransformTensor(1, src, 0, dest)
Base.fill!(src::Tensor,value::Number)=cudnnSetTensor(src,value)
Base.scale!(src::Tensor, alpha::Number)=cudnnScaleTensor(src,alpha)

# CUDArt functions
CUDArt.to_host(t::AbstractTensor)=to_host(t.data)
CUDArt.free(t::Tensor)=(free(t.data); cudnnDestroyTensorDescriptor(t.desc))
CUDArt.free(t::Filter)=(free(t.data); cudnnDestroyFilterDescriptor(t.desc))

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

# Read the filter descriptor (mostly for debugging)
cudnnGetFilterNdDescriptor(t::Filter)=cudnnGetFilterNdDescriptor(t.desc)
function cudnnGetFilterNdDescriptor(td::cudnnFilterDescriptor_t, nbDimsRequested=8)
    dataType = cudnnDataType_t[0]
    nbDims = Array(Cint, 1)
    dimA = Array(Cint, nbDimsRequested)
    cudnnGetFilterNdDescriptor(td,nbDimsRequested,dataType,nbDims,dimA)
    return (dataType[1], nbDims[1], dimA[1:nbDims[1]])
    # nbDimsRequested > 8 gives error
end

# For cudnn functions that require a pointer to a number
ptr(x,a)=eltype(a)[x]

# alpha * src + beta * dest -> dest
# Both beta and dest optional, beta=0 if not specified, dest is allocated with ones if not specified.
# These defaults give the expected answers for (alpha * src) and (alpha * src + beta)
# The doc says no in-place, i.e. src and dest should not overlap.  
# My experiments say otherwise but I will go with the doc just in case.

function cudnnTransformTensor(alpha::Number, src::Tensor, beta::Number=0, dest::Tensor=ones(src); handle=cudnnHandle)
    cudnnTransformTensor(handle, 
                         ptr(alpha,src), src.desc, src.data.ptr, 
                         ptr(beta,dest), dest.desc, dest.data.ptr)
    return dest
end

# Refer to cudnn doc to see what different add modes do

function cudnnAddTensor(mode::cudnnAddMode_t, alpha::Number, bias::Tensor, beta::Number, src::Tensor; handle=cudnnHandle)
    cudnnAddTensor(handle, mode, ptr(alpha,bias), bias.desc, bias.data.ptr, ptr(beta,src), src.desc, src.data.ptr)
    return src
end

# src .= value

function cudnnSetTensor(src::Tensor, value::Number; handle=cudnnHandle)
    cudnnSetTensor(handle, src.desc, src.data.ptr, ptr(value,src))
    return src
end

# src .*= alpha

function cudnnScaleTensor(src::Tensor, alpha::Number; handle=cudnnHandle)
    cudnnScaleTensor(handle, src.desc, src.data.ptr, ptr(alpha,src))
    return src
end

# Apply activation fn to each element of src (dest optional, in-place by default)
# mode is one of CUDNN_ACTIVATION_{SIGMOID,RELU,TANH}

function cudnnActivationForward(src::Tensor, dest::Tensor=src; handle=cudnnHandle, 
                                mode=CUDNN_ACTIVATION_RELU, alpha=1.0, beta=0.0)
    cudnnActivationForward(handle, mode, 
                           ptr(alpha,src), src.desc, src.data.ptr, 
                           ptr(beta,dest), dest.desc, dest.data.ptr)
    return dest
end

# Compute activation fn gradient.  The naming is a bit confusing.  In
# KUnet terminology, if we have input x and output y going forward, we
# get dy, the gradient wrt output y, and compute dx, the gradient wrt
# input x (dx overwrites dy by default).  In cudnn, x=dest, y=src,
# dx=destDiff, dy=srcDiff.  The gradient calculation should not need
# x=dest.  cudnn seems to set dx=0 whereever x=0.

function cudnnActivationBackward(src::Tensor, srcDiff::Tensor, dest::Tensor, destDiff::Tensor=srcDiff; 
                                 handle=cudnnHandle, mode=CUDNN_ACTIVATION_RELU, alpha=1.0, beta=0.0) 
    cudnnActivationBackward(handle, mode, 
                            ptr(alpha,src), src.desc, src.data.ptr, 
                            srcDiff.desc, srcDiff.data.ptr, 
                            dest.desc, dest.data.ptr,
                            ptr(beta,destDiff), destDiff.desc, destDiff.data.ptr)
    return destDiff
end

# In KUnet I implement softmax as a normalization function on the
# final output which is typically a small number of units representing
# classes.  CUDNN softmax acts on Tensors?  What does that mean?  Doc says:
# mode=INSTANCE: The softmax operation is computed per image (N) across the dimensions C,H,W.
# mode=CHANNEL: The softmax operation is computed per spatial location (H,W) per image (N) across the dimension C.
# From caffe docs: image date blob has NCHW dims.
# Regular feature vectors (that are input to InnerProduct layers) have NC11 dims. (N vectors holding C dims each)
# Convolution parameter blob has NCHW where N output filters, C input channels, HxW images.
# Inner product weight matrix has 11HW for H output and W input dims.
# Caffe softmax takes pred and label blobs and computes a single number.
# cudnnSoftmaxForward seems to just exponentiate and normalize.
# This is more like an activation function rather than a loss function.
# It takes a NC11 tensor (N vectors, holding C unnormalized logp each) and returns an NC11 tensor with normalized p.
# Note that an NC11 tensor is constructed from a julia array of size (1,1,C,N).
# Can we do in-place?  It turns out yes (undocumented).

function cudnnSoftmaxForward(src::Tensor, dest::Tensor=src; 
                             handle=cudnnHandle,
                             algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                             mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                             alpha=1.0, beta=0.0)
    cudnnSoftmaxForward(handle, algorithm, mode,
                        ptr(alpha, src), src.desc, src.data.ptr,
                        ptr(beta, dest), dest.desc, dest.data.ptr)
    return dest
end

# If SoftmaxForward is implemented as an activation fn rather than a
# loss fn, it stands to reason that its backward partner is similar.
# If the input to the forward pass was x, array of unnormalized logp,
# and forward output was y, array of normalized p, then I am going to
# guess below that src=y, srcDiff=dy, and destDiff=dx.  Nothing in
# docs about in-place ops but I will assume it is possible as the
# structure is similar to cudnnActivateBackward (except dest=x is not in
# the arglist).
# 
# But what does it compute?  Given a one-of-k vector of correct
# answers ai we have: dJ/dxi = (yi - ai), so what is srcDiff?  Do they
# really mean dJ/dyi?  Who computes dJ/dyi?  What idiot wrote this
# documentation without a single mathematical expression?
#
# OK, if src is normalized probabilities output by the model, and we
# assume y1 is the probability of the correct answer we have
# J=-log(y1) and we should specify srcDiff as dJ/dy1 = -1/y1 and
# dJ/dyi=1/y1 (for i!=1).  If we do, we should get destDiff
# dJ/dx1=y1-1 and dJ/dxi=yi (for i!=1).  We get twice those answers.
# I have no idea where the factor of 2 comes from.  I recommend
# dividing dJ/dyi with 2 for srcDiff to get the correct derivatives.
# You should also divide all dJ/dyi by the number of instances (N) to
# make learningRate specify the same step size regardless of batch
# size.

function cudnnSoftmaxBackward(src::Tensor, srcDiff::Tensor, destDiff::Tensor=srcDiff;
                              handle=cudnnHandle,
                              algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                              mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                              alpha=1.0, beta=0.0)
    cudnnSoftmaxBackward(handle, algorithm, mode,
                         ptr(alpha, src), src.desc, src.data.ptr,
                         srcDiff.desc, srcDiff.data,
                         ptr(beta, destDiff), destDiff.desc, destDiff.data.ptr)
    return destDiff
end

# The cudnn documentation again does not tell anything about how the
# pooling parameters are used.  According to the caffe source
# (pooling_layer.cpp) here is what the parameters in the
# PoolingDescriptor mean: windowHeight and windowWidth give the size
# of the pooling area.  To compute the pooled output entry at position
# (ph,pw), we look at the input entries in the range:
#
# hstart = ph*verticalStride - verticalPadding
# wstart = pw*horizontalStride - horizontalPadding
# hend = hstart + windowHeight
# wend = wstart + windowWidth
#
# This is 0 based, the end points are exclusive, C style.
# The ranges are truncated if they are outside the input size.
# This makes the output size:
#
# pooledHeight = ceil((inputHeight + 2 * verticalPadding - windowHeight) / verticalStride) + 1
# pooledWidth = ceil((inputWidth + 2 * horizontalPadding - windowWidth) / horizontalStride) + 1
#
# If the output tensor is smaller, cudnn fills it with the upper left
# region of the pooled output.  If the output tensor is larger, it is
# filled up to the size of the input + 2 * padding, with the rest of
# the entries set to -inf (for max pooling) and NaN (for avg pooling).
# Actually there is a single row/col of NaNs and 0s outside.
#
# CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING is buggy when padding >
# 0 or else I have no idea what it is doing.

immutable PoolingDescriptor; dims; padding; stride; mode; ptr;
    function PoolingDescriptor(dims; padding=zeros(dims), stride=dims, mode=CUDNN_POOLING_MAX)
        @assert in(mode, (CUDNN_POOLING_MAX, 
                          CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 
                          CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING))
        @assert length(dims) == length(padding) == length(stride)
        pd = Array(cudnnPoolingDescriptor_t, 1)
        cudnnCreatePoolingDescriptor(pd)
        cudnnSetPoolingNdDescriptor(pd[1],mode,length(dims),Cint[dims...],Cint[padding...],Cint[stride...])
        new(dims, padding, stride, mode, pd[1])
    end
end

CUDArt.free(pd::PoolingDescriptor)=cudnnDestroyPoolingDescriptor(pd.ptr)

function cudnnPoolingForward(pd::PoolingDescriptor, src::Tensor, dest::Tensor; 
                             handle=cudnnHandle, alpha=1.0, beta=0.0)
    cudnnPoolingForward(handle, pd.ptr, 
                        ptr(alpha,src), src.desc, src.data.ptr,
                        ptr(beta,dest), dest.desc, dest.data.ptr)
    return dest
end
                             
# Read info from gpu for debugging
function cudnnGetPoolingNdDescriptor(pd::PoolingDescriptor)
    nd = length(pd.dims)
    m = cudnnPoolingMode_t[0]
    n = Cint[0]
    s = Array(Cint, nd)
    p = Array(Cint, nd)
    t = Array(Cint, nd)
    cudnnGetPoolingNdDescriptor(pd.ptr, nd, m, n, s, p, t)
    inttuple(x)=tuple(Int[x...]...)
    (m[1], n[1], inttuple(s), inttuple(p), inttuple(t))
end

# This does not seem to exist in libcudnn.so even though it is declared in cudnn.h and the docs
# It is also in the static library but not the .so!  Weird.
#
# function cudnnGetPoolingNdForwardOutputDim(pd::PoolingDescriptor, input::Tensor)
#     nbDims = ndims(input)
#     outputTensorDimA = Array(Cint, nbDims)
#     cudnnGetPoolingNdForwardOutputDim(pd.ptr, input.desc, nbDims, outputTensorDimA)
#     tuple(outputTensorDimA...)
# end


function cudnnPoolingBackward(pd::PoolingDescriptor, src::Tensor, srcDiff::Tensor, dest::Tensor, destDiff::Tensor; 
                              handle=cudnnHandle, alpha=1.0, beta=0.0)
    cudnnPoolingBackward(handle, pd.ptr, 
                         ptr(alpha,src), src.desc, src.data.ptr, 
                         srcDiff.desc, srcDiff.data.ptr, 
                         dest.desc, dest.data.ptr,
                         ptr(beta,destDiff), destDiff.desc, destDiff.data.ptr)
    return destDiff
end



end # module
