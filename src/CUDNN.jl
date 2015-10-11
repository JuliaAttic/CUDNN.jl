### High level interface to CUDNN:
# 0. Init
# 1. Descriptors
# 2. Convolution
# 3. Pooling
# 4. Other tensor functions
# 5. Helper functions


### 0. INIT ##################################################

module CUDNN
using Compat
using CUDArt

const libcudnn = Libdl.find_library(["libcudnn"])
isempty(libcudnn) && error("CUDNN library cannot be found")
const CUDNN_VERSION = convert(Int, ccall((:cudnnGetVersion,libcudnn),Csize_t,()))
export CUDNN_VERSION

export cudnnTransformTensor, cudnnAddTensor, cudnnSetTensor, cudnnScaleTensor
export CUDNN_ADD_IMAGE, CUDNN_ADD_SAME_HW, CUDNN_ADD_FEATURE_MAP, CUDNN_ADD_SAME_CHW, CUDNN_ADD_SAME_C, CUDNN_ADD_FULL_TENSOR
export cudnnActivationForward, cudnnActivationBackward
export CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH
export cudnnSoftmaxForward, cudnnSoftmaxBackward
export cudnnPoolingForward, cudnnPoolingBackward
export cudnnConvolutionForward
export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION
export cudnnConvolutionBackwardBias, cudnnConvolutionBackwardFilter, cudnnConvolutionBackwardData
export CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
export cudnnGetConvolutionNdForwardOutputDim, cudnnGetPoolingNdForwardOutputDim, cudnnGetConvolutionForwardWorkspaceSize

import Base: unsafe_convert, conv2, strides
import CUDArt: free

# This is missing from CUDArt:
type cudaStream; end
typealias cudaStream_t Ptr{cudaStream}

# This is missing from cudnn.h:
const CUDNN_DIM_MAX = 8

# error handling function
function cudnnCheck(status)
    if status == CUDNN_STATUS_SUCCESS
        return nothing
    end
    # Because try/finally may disguise the source of the problem,
    # let's show a backtrace here
    warn("CUDNN error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    println()
    throw(cudnnGetErrorString(status))
end

# These are semi-automatically generated using Clang with wrap_cudnn.jl:
include("types.jl")
include("libcudnn.jl")

# Conversion between Julia and CUDNN datatypes
cudnnDataType(::AbstractCudaArray{Float16})=CUDNN_DATA_HALF
cudnnDataType(::AbstractCudaArray{Float32})=CUDNN_DATA_FLOAT
cudnnDataType(::AbstractCudaArray{Float64})=CUDNN_DATA_DOUBLE
cudnnDataType(::Type{Float16})=CUDNN_DATA_HALF
cudnnDataType(::Type{Float32})=CUDNN_DATA_FLOAT
cudnnDataType(::Type{Float64})=CUDNN_DATA_DOUBLE
juliaDataType(a)=(a==CUDNN_DATA_HALF ? Float16 :
                  a==CUDNN_DATA_FLOAT ? Float32 :
                  a==CUDNN_DATA_DOUBLE ? Float64 : error())

# Setup default cudnn handle
cudnnHandlePtr = cudnnHandle_t[0]
cudnnCreate(cudnnHandlePtr)
cudnnHandle = cudnnHandlePtr[1]
# destroy cudnn handle at julia exit
atexit(()->cudnnDestroy(cudnnHandle))


### 1. DESCRIPTORS ##################################################

type TD; ptr; end
type FD; ptr; end
type CD; ptr; end
type PD; ptr; end

free(td::TD)=cudnnDestroyTensorDescriptor(td.ptr)
free(fd::FD)=cudnnDestroyFilterDescriptor(fd.ptr)
free(cd::CD)=cudnnDestroyConvolutionDescriptor(cd.ptr)
free(pd::PD)=cudnnDestroyPoolingDescriptor(pd.ptr)

unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TD)=td.ptr
unsafe_convert(::Type{cudnnFilterDescriptor_t}, fd::FD)=fd.ptr
unsafe_convert(::Type{cudnnConvolutionDescriptor_t}, cd::CD)=cd.ptr
unsafe_convert(::Type{cudnnPoolingDescriptor_t}, pd::PD)=pd.ptr

function TD(a::AbstractCudaArray, dims=ndims(a))
    (sz, st) = tensorsize(a, dims)
    d = cudnnTensorDescriptor_t[0]
    cudnnCreateTensorDescriptor(d)
    cudnnSetTensorNdDescriptor(d[1], cudnnDataType(a), length(sz), sz, st)
    this = TD(d[1])
    finalizer(this, free)
    return this
end

function FD(a::AbstractCudaArray, dims=ndims(a))
    # The only difference of a FilterDescriptor is no strides.
    (sz, st) = tensorsize(a, dims)
    d = cudnnFilterDescriptor_t[0]
    cudnnCreateFilterDescriptor(d)
    cudnnSetFilterNdDescriptor(d[1], cudnnDataType(a), length(sz), sz)
    this = FD(d[1])
    finalizer(this, free)
    return this
end

function CD(nd, padding, stride, upscale, mode, xtype)
    cd = cudnnConvolutionDescriptor_t[0]
    cudnnCreateConvolutionDescriptor(cd)
    CUDNN_VERSION >= 3000 ?
    cudnnSetConvolutionNdDescriptor(cd[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,cudnnDataType(xtype)) :
    cudnnSetConvolutionNdDescriptor(cd[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode)
    this = CD(cd[1])
    finalizer(this, free)
    return this
end

CD(; ndims=2, padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION, eltype=Float32, o...)=CD(ndims, padding, stride, upscale, mode, eltype)
CD(a::AbstractCudaArray; o...)=CD(; ndims=ndims(a)-2, eltype=eltype(a), o...)

function PD(nd, window, padding, stride, mode)
    pd = cudnnPoolingDescriptor_t[0]
    cudnnCreatePoolingDescriptor(pd)
    cudnnSetPoolingNdDescriptor(pd[1],mode,nd,pdsize(window,nd),pdsize(padding,nd),pdsize(stride,nd))
    this = PD(pd[1])
    finalizer(this, free)
    return this
end

PD(; ndims=2, window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX, o...)=PD(ndims,window,padding,stride,mode)
PD(a::AbstractCudaArray; o...)=PD(; ndims=ndims(a)-2, o...)

# This is missing from CUDNN although mentioned in the documentation
const CUDNN_MAX_DIM = 8

function cudnnGetTensorNdDescriptor(td::TD)
    nd=CUDNN_MAX_DIM
    dataType = cudnnDataType_t[0]
    np = Array(Cint, 1)
    dimA = Array(Cint, nd)
    strideA = Array(Cint, nd)
    cudnnGetTensorNdDescriptor(td,nd,dataType,np,dimA,strideA)
    n = np[1]
    return (dataType[1], n, inttuple(dimA[1:n]), inttuple(strideA[1:n]))
end

function cudnnGetFilterNdDescriptor(fd::FD)
    nd=CUDNN_MAX_DIM
    dataType = cudnnDataType_t[0]
    np = Array(Cint, 1)
    dimA = Array(Cint, nd)
    cudnnGetFilterNdDescriptor(fd,nd,dataType,np,dimA)
    n = np[1]
    return (dataType[1], n, inttuple(dimA[1:n]))
end

function cudnnGetConvolutionNdDescriptor(cd::CD)
    nd = CUDNN_DIM_MAX - 2
    np = Cint[0]
    p = Array(Cint, nd)
    s = Array(Cint, nd)
    u = Array(Cint, nd)
    m = cudnnConvolutionMode_t[0]
    if CUDNN_VERSION >= 3000
        t = cudnnDataType_t[0]
        cudnnGetConvolutionNdDescriptor(cd, nd, np, p, s, u, m, t); n = np[1];
        (n, inttuple(p[1:n]), inttuple(s[1:n]), inttuple(u[1:n]), m[1], juliaDataType(t[1]))
    else
        # Only nd<3 works
        cudnnGetConvolutionNdDescriptor(cd, 2, np, p, s, u, m); n = np[1];
        (n, inttuple(p[1:n]), inttuple(s[1:n]), inttuple(u[1:n]), m[1])
    end
end

function cudnnGetPoolingNdDescriptor(pd::PD)
    nd = CUDNN_MAX_DIM - 2
    np = Cint[0]
    m = cudnnPoolingMode_t[0]
    s = Array(Cint, nd)
    p = Array(Cint, nd)
    t = Array(Cint, nd)
    cudnnGetPoolingNdDescriptor(pd, nd, m, np, s, p, t)
    n = np[1]
    (m[1], n, inttuple(s[1:n]), inttuple(p[1:n]), inttuple(t[1:n]))
end


### 2. CONVOLUTION ##################################################

# CUDNN/Caffe sizes for various arrays in column-major notation:
# conv x: (W,H,C,N): W,H=image size, C=channels, N=instances
# conv w: (X,Y,C,K): X,Y=filter size, C=input channels, K=output channels
# conv y: (W-X+1,H-Y+1,K,N)
# conv b: (1,1,K,1)
# mmul x: (1,1,C,N): C=features, N=instances
# mmul w: (1,1,K,C)
# mmul y: (1,1,K,N)
# mmul b: (1,1,K,1)

# This function returns the dimensions of the resulting n-D tensor of a nbDims-2-D
# convolution, given the convolution descriptor, the input tensor descriptor and the filter
# descriptor This function can help to setup the output tensor and allocate the proper
# amount of memory prior to launch the actual convolution.
# Each dimension of the (nbDims-2)-D images of the output tensor is computed as
# followed:
#  outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride;

function cudnnGetConvolutionNdForwardOutputDim(src::AbstractCudaArray, filter::AbstractCudaArray; cd=nothing, o...)
    cd1 = (cd == nothing ? CD(src; o...) : cd)
    nbDims = ndims(src)
    outputDim = Array(Cint, nbDims)
    cudnnGetConvolutionNdForwardOutputDim(cd1, TD(src), FD(filter), nbDims, outputDim)
    # cd1 === cd || free(cd1)
    tuple(Int[reverse(outputDim)...]...)
end

# These are related to convolution algorithm selection:
# cudnnGetConvolutionForwardAlgorithm
# cudnnGetConvolutionForwardWorkspaceSize
#
# From the v2 release notes it seems like IMPLICIT_PRECOMP_GEMM is a
# good default especially with our memory limitations.
#
# Forward convolution is now implemented via several different algorithms, and 
# the interface allows the application to choose one of these algorithms specifically 
# or to specify a strategy (e.g., prefer fastest, use no additional working space) by 
# which the library should select the best algorithm automatically. The four 
# algorithms currently given are as follows:
# o IMPLICIT_GEMM corresponds to the sole algorithm that was provided in 
# cuDNN Release 1; it is the only algorithm that supports all input sizes while 
# using no additional working space.
# o IMPLICIT_PRECOMP_GEMM is a modification of this approach that uses a 
# small amount of working space (specifically, C*R*S*sizeof(int) bytes, 
# where C is the number of input feature maps and R and S are the filter height 
# and width, respectively, for the case of 2D convolutions) to achieve 
# significantly higher performance in most cases. This algorithm achieves its 
# highest performance when zero-padding is not used.
# o GEMM is an “im2col”-based approach, explicitly expanding the input data 
# in memory and then using an otherwise-pure matrix multiplication that 
# obeys cuDNN’s input and output stridings, avoiding a separate transposition 
# step on the input or output. Note that this algorithm requires significant
# working space, though it may be faster than either of the two “implicit 
# GEMM” approaches in some cases. As such, the PREFER_FASTEST
# forward convolution algorithm preference may sometimes recommend this 
# approach. When memory should be used more sparingly, 
# SPECIFY_WORKSPACE_LIMIT can be used instead of PREFER_FASTEST
# to ensure that the algorithm recommended will not require more than a given 
# amount of working space.
# o DIRECT is a placeholder for a future implementation of direct convolution.

function cudnnGetConvolutionForwardAlgorithm(src::AbstractCudaArray, filter::AbstractCudaArray, dest::AbstractCudaArray; 
                                             handle=cudnnHandle,
                                             preference=CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, memoryLimitInbytes=10^9,
                                             cd=nothing, o...)
    cd1 = (cd == nothing ? CD(src; o...) : cd)
    algo = cudnnConvolutionFwdAlgo_t[0]
    cudnnGetConvolutionForwardAlgorithm(handle,TD(src),FD(filter),cd1,TD(dest),preference,memoryLimitInbytes,algo)
    # cd1 === cd || free(cd1)
    return algo[1]
end

function cudnnGetConvolutionForwardWorkspaceSize(src::AbstractCudaArray, filter::AbstractCudaArray, dest::AbstractCudaArray;
                                                 handle=cudnnHandle, algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                                 cd=nothing, o...)
    cd1 = (cd == nothing ? CD(src; o...) : cd)
    sizeInBytes = Csize_t[0]
    cudnnGetConvolutionForwardWorkspaceSize(handle,TD(src),FD(filter),cd1,TD(dest),algorithm,sizeInBytes)
    # cd1 === cd || free(cd1)
    return Int(sizeInBytes[1])
end

function cudnnConvolutionForward(src, filter, dest=nothing;
                                 handle=cudnnHandle, alpha=1.0, beta=0.0, 
                                 algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                 workSpace=C_NULL, workSpaceSizeInBytes=0,
                                 cd=nothing, o...)
    cd1 = (cd == nothing ? CD(src; o...) : cd)
    @assert eltype(filter) == eltype(src)
    osize = cudnnGetConvolutionNdForwardOutputDim(src,filter; cd=cd1)
    (dest == nothing) && (dest = CudaArray(eltype(src), osize))
    @assert osize == size(dest)
    @assert eltype(dest) == eltype(src)
    wsize = cudnnGetConvolutionForwardWorkspaceSize(src, filter, dest; algorithm=algorithm, cd=cd1)
    if ((wsize > 0) && (workSpace == C_NULL || workSpaceSizeInBytes < wsize))
        workSpaceSizeInBytes = wsize
        ws = CudaArray(Int8, workSpaceSizeInBytes)
    else
        ws = workSpace
    end
    cudnnConvolutionForward(handle,
                            cptr(alpha,src),TD(src),src,
                            FD(filter),filter,
                            cd1,algorithm,ws,workSpaceSizeInBytes,
                            cptr(beta,dest),TD(dest),dest)
    ws === workSpace || free(ws)
    # cd1 === cd || free(cd1)
    return dest
end


# conv2(x,w) in Julia performs 2-D convolution with padding equal to
# one less than the w dimensions.

Base.conv2{T}(src::AbstractCudaArray{T}, filter::AbstractCudaArray{T}, dest=nothing)=
    cudnnConvolutionForward(src, filter, dest; cd=CD(src; padding = (size(filter,1)-1, size(filter,2)-1)))

# n=h=w=1 for dest and c same as input.  CUDNN seems to assume a
# single scalar bias per output channel, i.e. the same number is added
# to every image and every pixel on the same output channel.  Say our
# input was x, we convolved it going forward with w, added bias b, and
# got tensor y as a result.  i.e. y=w*x+b where * means convolution
# and + means broadcasting addition. Now we get dJ/dy=src, and we want
# dJ/db=dest.  Well, we simply need to add up all dy where a
# particular b was added to get db.  This means dest is just the sum
# of src entries for every channel.
function cudnnConvolutionBackwardBias(src, dest=CudaArray(eltype(src),(1,1,size(src,3),1));
                                      handle=cudnnHandle, alpha=1.0, beta=0.0)
    cudnnConvolutionBackwardBias(handle,cptr(alpha,src),TD(src,4),src,cptr(beta,dest),TD(dest,4),dest)
    return dest
end

# I am guessing if y=w*x+b going forward, the arguments below
# correspond to src=x, diff=dy, grad=dw.
function cudnnConvolutionBackwardFilter(src, diff, grad;
                                        handle=cudnnHandle, alpha=1.0, beta=0.0, 
                                        algo=CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                        workSpace=C_NULL, workSpaceSizeInBytes=0,
                                        cd = nothing, o...)
    cd1 = (cd == nothing ? CD(src; o...) : cd)
    CUDNN_VERSION >= 3000 ? 
    cudnnConvolutionBackwardFilter(handle,
                                   cptr(alpha,src),TD(src),src,
                                   TD(diff),diff,cd1,
                                   algo, workSpace, workSpaceSizeInBytes,
                                   cptr(beta,grad),FD(grad),grad) :
    cudnnConvolutionBackwardFilter(handle,
                                   cptr(alpha,src),TD(src),src,
                                   TD(diff),diff,cd1,
                                   cptr(beta,grad),FD(grad),grad)
    # cd1 === cd || free(cd1)
    return grad
end

# I am guessing if y=w*x+b going forward, the arguments below
# correspond to filter=w, diff=dy, grad=dx.
function cudnnConvolutionBackwardData(filter, diff, grad;
                                      handle=cudnnHandle, alpha=1.0, beta=0.0, 
                                      algo=CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                      workSpace=C_NULL, workSpaceSizeInBytes=0,
                                      cd = nothing, o...)
    cd1 = (cd == nothing ? CD(filter; o...) : cd)
    CUDNN_VERSION >= 3000 ? 
    cudnnConvolutionBackwardData(handle,cptr(alpha,diff),
                                 FD(filter),filter,
                                 TD(diff),diff,cd1,
                                 algo, workSpace, workSpaceSizeInBytes,
                                 cptr(beta,grad),TD(grad),grad) :
    cudnnConvolutionBackwardData(handle,cptr(alpha,diff),
                                 FD(filter),filter,
                                 TD(diff),diff,cd1,
                                 cptr(beta,grad),TD(grad),grad)
    # cd1 === cd || free(cd1)
    return grad
end

### 3. POOLING ##################################################

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

function cudnnPoolingForward(src, dest; 
                             handle=cudnnHandle, alpha=1.0, beta=0.0,
                             pd = nothing, o...)
    pd1 = (pd == nothing ? PD(src; o...) : pd)
    cudnnPoolingForward(handle, pd1, 
                        cptr(alpha,src), TD(src), src,
                        cptr(beta,dest), TD(dest), dest)
    # pd1 === pd || free(pd1)
    return dest
end

function cudnnPoolingBackward(src, srcDiff, dest, destDiff; 
                              handle=cudnnHandle, alpha=1.0, beta=0.0,
                              pd = nothing, o...)
    pd1 = (pd == nothing ? PD(src; o...) : pd)
    cudnnPoolingBackward(handle, pd1, 
                         cptr(alpha,src), TD(src), src, 
                         TD(srcDiff), srcDiff, 
                         TD(dest), dest,
                         cptr(beta,destDiff), TD(destDiff), destDiff)
    # pd1 === pd || free(pd1)
    return destDiff
end

# This does not seem to exist in libcudnn.so even though it is declared in cudnn.h and the docs
# It is also in the static library but not the .so!  Weird.
#
# function cudnnGetPoolingNdForwardOutputDim(pd::PoolingDescriptor, input::AbstractCudaArray)
#     nbDims = ndims(input)
#     outputTensorDimA = Array(Cint, nbDims)
#     cudnnGetPoolingNdForwardOutputDim(pd, TD(input), nbDims, outputTensorDimA)
#     tuple(outputTensorDimA...)
# end

function cudnnGetPoolingNdForwardOutputDim(pd::PD, input::AbstractCudaArray)
    dims = [size(input)...]
    (mode, pdims, window, padding, stride) = cudnnGetPoolingNdDescriptor(pd)
    for i=1:length(dims)-2
        dims[i] = 1 + ceil((dims[i] + 2*padding[i] - window[i]) / stride[i])
    end
    tuple(dims...)
end



### 4. OTHER TENSOR FUNCTIONS ##################################################

# alpha * src + beta * dest -> dest
# Both beta and dest optional, beta=0 if not specified, dest is allocated with ones if not specified.
# These defaults give the expected answers for (alpha * src) and (alpha * src + beta)
# The doc says no in-place, i.e. src and dest should not overlap.  
# My experiments say otherwise but I will go with the doc just in case.

function cudnnTransformTensor(alpha::Number, src::AbstractCudaArray, beta::Number=0, dest::AbstractCudaArray=ones(src); 
                              handle=cudnnHandle)
    cudnnTransformTensor(handle, 
                         cptr(alpha,src), TD(src,4), src, 
                         cptr(beta,dest), TD(dest,4), dest)
    return dest
end

# Refer to cudnn doc to see what different add modes do

function cudnnAddTensor(bias::AbstractCudaArray, src::AbstractCudaArray;
                        handle=cudnnHandle, alpha=1.0, beta=1.0, mode=CUDNN_ADD_SAME_C)
    cudnnAddTensor(handle, mode, cptr(alpha,bias), TD(bias,4), bias, cptr(beta,src), TD(src,4), src)
    return src
end

# src .= value

function cudnnSetTensor(src::AbstractCudaArray, value::Number; handle=cudnnHandle)
    cudnnSetTensor(handle, TD(src,4), src, cptr(value,src))
    return src
end

# src .*= alpha

function cudnnScaleTensor(src::AbstractCudaArray, alpha::Number; handle=cudnnHandle)
    cudnnScaleTensor(handle, TD(src,4), src, cptr(alpha,src))
    return src
end

# Apply activation fn to each element of src (dest optional, in-place by default)
# mode is one of CUDNN_ACTIVATION_{SIGMOID,RELU,TANH}

function cudnnActivationForward(src::AbstractCudaArray, dest::AbstractCudaArray=src; handle=cudnnHandle, 
                                mode=CUDNN_ACTIVATION_RELU, alpha=1.0, beta=0.0)
    cudnnActivationForward(handle, mode, 
                           cptr(alpha,src), TD(src,4), src, 
                           cptr(beta,dest), TD(dest,4), dest)
    return dest
end

# Compute activation fn gradient.  The naming is a bit confusing.  In
# KUnet terminology, if we have input x and output y going forward, we
# get dy, the gradient wrt output y, and compute dx, the gradient wrt
# input x (dx overwrites dy by default).  In cudnn, x=dest, y=src,
# dx=destDiff, dy=srcDiff.  The gradient calculation should not need
# x=dest.  cudnn seems to set dx=0 whereever x=0.

function cudnnActivationBackward(src::AbstractCudaArray, srcDiff::AbstractCudaArray, dest::AbstractCudaArray=src, destDiff::AbstractCudaArray=srcDiff; 
                                 handle=cudnnHandle, mode=CUDNN_ACTIVATION_RELU, alpha=1.0, beta=0.0) 
    cudnnActivationBackward(handle, mode, 
                            cptr(alpha,src), TD(src,4), src, 
                            TD(srcDiff,4), srcDiff, 
                            TD(dest,4), dest,
                            cptr(beta,destDiff), TD(destDiff,4), destDiff)
    return destDiff
end

# In KUnet I implement softmax as a normalization function on the
# final output which is typically a small number of units representing
# classes.  CUDNN softmax acts on Tensors?  What does that mean?  Doc says:
# mode=INSTANCE: The softmax operation is computed per image (N) across the dimensions C,H,W.
# mode=CHANNEL: The softmax operation is computed per spatial location (H,W) per image (N) across the dimension C.
# From caffe docs: image data blob has NCHW dims.
# Regular feature vectors (that are input to InnerProduct layers) have NC11 dims. (N vectors holding C dims each)
# Convolution parameter blob has NCHW where N output filters, C input channels, HxW images.
# Inner product weight matrix has 11HW (or is it CK11) for H output and W input dims.
# Caffe softmax takes pred and label blobs and computes a single number.
# cudnnSoftmaxForward seems to just exponentiate and normalize.
# This is more like an activation function rather than a loss function.
# It takes a NC11 tensor (N vectors, holding C unnormalized logp each) and returns an NC11 tensor with normalized p.
# Note that an NC11 tensor is constructed from a julia array of size (1,1,C,N).
# Can we do in-place?  It turns out yes (undocumented).

function cudnnSoftmaxForward(src::AbstractCudaArray, dest::AbstractCudaArray=src; 
                             handle=cudnnHandle,
                             algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                             mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                             alpha=1.0, beta=0.0)
    cudnnSoftmaxForward(handle, algorithm, mode,
                        cptr(alpha, src), TD(src,4), src,
                        cptr(beta, dest), TD(dest,4), dest)
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
# size.  (Or maybe one should divide the loss J instead).

function cudnnSoftmaxBackward(src::AbstractCudaArray, srcDiff::AbstractCudaArray, destDiff::AbstractCudaArray=srcDiff;
                              handle=cudnnHandle,
                              algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                              mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                              alpha=1.0, beta=0.0)
    cudnnSoftmaxBackward(handle, algorithm, mode,
                         cptr(alpha, src), TD(src,4), src,
                         TD(srcDiff,4), srcDiff,
                         cptr(beta, destDiff), TD(destDiff,4), destDiff)
    return destDiff
end


### 5. HELPER FUNCTIONS ##################################################

# This is missing from CUDArt:
Base.strides(a::AbstractCudaArray)=map(i->stride(a,i), tuple(1:ndims(a)...))

# For low level cudnn functions that require a pointer to a number
cptr(x,a::CudaArray{Float64})=Float64[x]
cptr(x,a::CudaArray{Float32})=Float32[x]
cptr(x,a::CudaArray{Float16})=Float32[x]

inttuple(x)=tuple(Int[x...]...)

# size and strides in Julia are in column-major format, e.g. (W,H,C,N)
# whereas cudnn uses row-major, e.g. NCHW
# most cudnn ops still work only on 4-D arrays
# so we may need to modify size and strides

function tensorsize(a, dims)
    sz = Cint[reverse(size(a))...]
    st = Cint[reverse(strides(a))...]
    if length(sz) == 1 < dims
        unshift!(sz, 1)
        unshift!(st, 1)
    end
    while length(sz) < dims
        push!(sz, 1)
        push!(st, 1)
    end
    while length(sz) > dims
        d = pop!(sz)
        sz[length(sz)] *= d
        pop!(st)
        st[length(st)] = 1
    end
    (sz, st)
end

function cdsize(w, nd)
    isa(w,Integer) ? Cint[fill(w,nd)...] :
    length(w)!=nd ? error("Dimension mismatch") :
    Cint[reverse(w)...]
end

pdsize(w, nd)=Cint[reverse(psize(w,nd))...]
psize(w, nd)=(isa(w,Integer)  ? fill(w,nd) : length(w) != nd ? error("Dimension mismatch") : w)


end # module


### DEAD CODE:

# I am no longer keeping default conv or pool descriptors.  These
# descriptors are created and destroyed by the operations that need
# them.  Thus the following functions have been modified:

# function cudnnConvolutionForward(src::AbstractCudaArray, filter::AbstractCudaArray, dest=nothing;
#                                  handle=cudnnHandle, alpha=1.0, beta=0.0, 
#                                  convDesc=defaultConvolutionDescriptor,
#                                  algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
#                                  workSpace=C_NULL, workSpaceSizeInBytes=0)
#     @assert eltype(filter) == eltype(src)
#     osize = cudnnGetConvolutionNdForwardOutputDim(src,filter;convDesc=convDesc)
#     (dest == nothing) && (dest = CudaArray(eltype(src), osize))
#     @assert osize == size(dest)
#     @assert eltype(dest) == eltype(src)
#     wsize = cudnnGetConvolutionForwardWorkspaceSize(src, filter, dest; algorithm=algorithm)
#     if ((wsize > 0) && (workSpace == C_NULL || workSpaceSizeInBytes < wsize))
#         workSpaceSizeInBytes = wsize
#         workSpace = CudaArray(Int8, workSpaceSizeInBytes)
#     end
#     cudnnConvolutionForward(handle,
#                             cptr(alpha,src),TD(src),src,
#                             FD(filter),filter,
#                             convDesc,algorithm,workSpace,workSpaceSizeInBytes,
#                             cptr(beta,dest),TD(dest),dest)
#     return dest
# end

# type ConvolutionDescriptor; ptr; padding; stride; upscale; mode; datatype; end

# # TODO: accept single number args for padding etc.
# function ConvolutionDescriptor(; padding=(0,0), 
#                                stride=map(x->1,padding), 
#                                upscale=map(x->1,padding), 
#                                mode=CUDNN_CONVOLUTION,
#                                datatype=Float64)
#     @assert in(mode, (CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION))
#     # @assert length(padding) == length(stride) == length(upscale)
#     cd = cudnnConvolutionDescriptor_t[0]
#     cudnnCreateConvolutionDescriptor(cd)
#     cudnnSetConvolutionNdDescriptor(cd[1],
#                                     length(padding),
#                                     Cint[reverse(padding)...],
#                                     Cint[reverse(stride)...],
#                                     Cint[reverse(upscale)...],
#                                     mode, cudnnDataType(datatype))
#     this = ConvolutionDescriptor(cd[1], padding, stride, upscale, mode, datatype)
#     finalizer(this, free)
#     this
# end

# free(cd::ConvolutionDescriptor)=cudnnDestroyConvolutionDescriptor(cd.ptr)
# unsafe_convert(::Type{cudnnConvolutionDescriptor_t}, cd::ConvolutionDescriptor)=cd.ptr

# const defaultConvolutionDescriptor = ConvolutionDescriptor()

# type PoolingDescriptor; ptr; dims; padding; stride; mode; end

# # TODO: allow initialization with a single number

# function PoolingDescriptor(dims; padding=map(x->0,dims), stride=dims, mode=CUDNN_POOLING_MAX)
#     @assert in(mode, (CUDNN_POOLING_MAX, 
#                       CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 
#                       CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING))
#     @assert length(dims) == length(padding) == length(stride)
#     pd = cudnnPoolingDescriptor_t[0]
#     cudnnCreatePoolingDescriptor(pd)
#     cudnnSetPoolingNdDescriptor(pd[1],mode,length(dims),Cint[reverse(dims)...],Cint[reverse(padding)...],Cint[reverse(stride)...])
#     this = PoolingDescriptor(pd[1], dims, padding, stride, mode)
#     finalizer(this, free)
#     this
# end

# free(pd::PoolingDescriptor)=cudnnDestroyPoolingDescriptor(pd.ptr)
# unsafe_convert(::Type{cudnnPoolingDescriptor_t}, pd::PoolingDescriptor)=pd.ptr

# function cudnnPoolingForward(pd::PoolingDescriptor, src::AbstractCudaArray, dest::AbstractCudaArray; 
#                              handle=cudnnHandle, alpha=1.0, beta=0.0)
#     cudnnPoolingForward(handle, pd, 
#                         cptr(alpha,src), TD(src), src,
#                         cptr(beta,dest), TD(dest), dest)
#     return dest
# end
                             
# # Read info from gpu for debugging
# function cudnnGetPoolingNdDescriptor(pd::PoolingDescriptor)
#     nd = length(pd.dims)
#     m = cudnnPoolingMode_t[0]
#     n = Cint[0]
#     s = Array(Cint, nd)
#     p = Array(Cint, nd)
#     t = Array(Cint, nd)
#     cudnnGetPoolingNdDescriptor(pd, nd, m, n, s, p, t)
#     inttuple(x)=tuple(Int[x...]...)
#     (m[1], n[1], inttuple(s), inttuple(p), inttuple(t))
# end

# type TD; ptr; end
# type FD; ptr; end
# free(td::TD)=cudnnDestroyTensorDescriptor(td.ptr)
# free(fd::FD)=cudnnDestroyFilterDescriptor(fd.ptr)
# unsafe_convert(::Type{cudnnTensorDescriptor_t}, td::TD)=td.ptr
# unsafe_convert(::Type{cudnnFilterDescriptor_t}, fd::FD)=fd.ptr

