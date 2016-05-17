using Base.Test
using CUDArt
using CUDNN

include("test_libcudnn.jl")

@show CUDNN_VERSION # this is set at runtime in CUDNN.jl, not fixed in types.jl

# Uncomment this if you want lots of messages:
# Base.Test.default_handler(r::Base.Test.Success) = info("$(r.expr)")
# Base.Test.default_handler(r::Base.Test.Failure) = info("FAIL: $(r.expr)")

# See which operations support which dimensions:
function testdims()
    for n=3:8
        @show n
        try
            dims = [(n+1):-1:2]
            x = CudaArray(rand(dims...))
            y = CudaArray(rand(dims...))
            dx = CudaArray(rand(dims...))
            dy = CudaArray(rand(dims...))
            CUDNN.cudnnActivationBackward(CUDNN.CUDNN_ACTIVATION_RELU, y, dy, x, dx)
            CUDNN.cudnnActivationForward(CUDNN.CUDNN_ACTIVATION_RELU, x)
            CUDNN.cudnnScaleTensor(x, pi)
            CUDNN.cudnnSetTensor(x, pi)
            CUDNN.cudnnTransformTensor(2.0, x, 3.0, y)
            bdim = ones(dims); bdim[end-1]=dims[end-1]
            bias = CudaArray(rand(bdim...))
            CUDNN.cudnnAddTensor(bias, x)
        catch y
            println(y)
        end
    end
end

# Or rtfm:
# The interface of cuDNN has been generalized so that data sets with other than two 
# spatial dimensions (e.g., 1D or 3D data) can be supported in future releases.
# o Note: while the interface now allows arbitrary N-dimensional tensors, most 
# routines in this release remain limited to two spatial dimensions. This may 
# be relaxed in future releases based on community feedback.
# o As a BETA preview in this release, the convolution forward, convolution 
# weight and data gradient, and cudnnSetTensor/cudnnScaleTensor routines 
# now support 3D datasets through the “Nd” interface. Note, however, that 
# these 3D routines have not yet been tuned for optimal performance. This will 
# be improved in future releases.


using CUDNN: cudnnGetVersion, cudnnGetErrorString, CUDNN_STATUS_SUCCESS
@show cudnnGetVersion()
@show bytestring(cudnnGetErrorString(CUDNN_STATUS_SUCCESS))

# TODO: handle type conflict between CUDArt and CUDNN when handling streams
# using CUDNN: cudnnCreate, cudnnDestroy, cudnnHandle_t, cudnnSetStream, cudnnGetStream, cudaStream_t
# hptr = cudnnHandle_t[0]
# cudnnCreate(hptr)
# sptr = cudaStream_t[0]
# CUDArt.rt.cudaStreamCreate(sptr) # library already checks == CUDArt.rt.cudaSuccess
# cudnnSetStream(hptr[1], sptr[1]) # library already checks == CUDNN_STATUS_SUCCESS in cudnnCheck
# tptr = cudaStream_t[0]
# cudnnGetStream(hptr[1], tptr)
# @test sptr[1] == tptr[1]
# CUDArt.rt.cudaStreamDestroy(sptr[1])
# cudnnDestroy(hptr[1])

using CUDNN: cudnnTensorDescriptor_t, cudnnCreateTensorDescriptor, cudnnSetTensor4dDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, cudnnDataType_t, cudnnGetTensor4dDescriptor
d = cudnnTensorDescriptor_t[0]
cudnnCreateTensorDescriptor(pointer(d))
cudnnSetTensor4dDescriptor(d[1], CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 2, 3, 4, 5)
dt = cudnnDataType_t[0]
for n in (:sn, :sc, :sh, :sw, :tn, :tc, :th, :tw); @eval $n = Cint[0]; end
cudnnGetTensor4dDescriptor(d[1],dt,sn,sc,sh,sw,tn,tc,th,tw)
@test (dt[1],sn[1],sc[1],sh[1],sw[1],tn[1],tc[1],th[1],tw[1]) == (CUDNN_DATA_DOUBLE, 2, 3, 4, 5, 60, 20, 5, 1)


using CUDNN: cudnnGetTensorNdDescriptor
nd=4; nbDims=Cint[0]; dimA=Array(Cint,nd); strideA=Array(Cint,nd)
cudnnGetTensorNdDescriptor(d[1],nd,dt,nbDims,dimA,strideA)
@test (nbDims, dimA, strideA) == (Cint[4], Cint[2,3,4,5], Cint[60,20,5,1])


using CUDNN: TD
x = ones(5,4,3,2)
tx = CudaArray(x)
@test to_host(tx) == x
@test cudnnGetTensorNdDescriptor(TD(tx)) == (CUDNN_DATA_DOUBLE, 4, (reverse(size(x))...), (reverse(strides(x))...))


using CUDNN: cudnnTransformTensor
y = ones(5,4,3,2)
ty = CudaArray(y)
@test to_host(cudnnTransformTensor(2, tx, 3, ty)) == 2x+3y


using CUDNN: cudnnAddTensor
b = rand(1,1,3,1)
tb = CudaArray(b)
x = rand(5,4,3,2)
tx = CudaArray(x)
@test to_host(cudnnAddTensor(tb,tx)) == x .+ b
x = rand(5,4,3,2)
tx = CudaArray(x)
@test to_host(cudnnAddTensor(tb,tx;alpha=2,beta=3)) == 2b .+ 3x


using CUDNN: cudnnSetTensor
@test to_host(cudnnSetTensor(tx, pi)) == fill!(ones(size(tx)), pi)


using CUDNN: cudnnScaleTensor
x = rand(5,4,3,2)
tx = CudaArray(x)
@test to_host(cudnnScaleTensor(tx, pi)) == x .* pi


using CUDNN: cudnnActivationForward, CUDNN_ACTIVATION_RELU
myrelu(x,y)=(copy!(y,x);for i=1:length(y); (y[i]<zero(y[i]))&&(y[i]=zero(y[i])); end; y)
x = rand(5,4,3,2) - 0.5; tx = CudaArray(x)
y = zeros(5,4,3,2); ty = CudaArray(y)
@test to_host(cudnnActivationForward(tx, ty, mode=CUDNN_ACTIVATION_RELU)) == myrelu(x, y)

using CUDNN: cudnnActivationBackward
dy = (rand(5,4,3,2) - 0.5); tdy = CudaArray(dy)
dx = zeros(5,4,3,2); tdx = CudaArray(dx)
myrelu(y,dy,dx)=(copy!(dx,dy);for i=1:length(y); (y[i]==zero(y[i]))&&(dx[i]=zero(dx[i])); end; dx)
@test to_host(cudnnActivationBackward(ty, tdy, tx, tdx, mode=CUDNN_ACTIVATION_RELU)) == myrelu(y,dy,dx)

using CUDNN: CUDNN_ACTIVATION_SIGMOID
mysigm(x,y)=(for i=1:length(y); y[i]=(1.0/(1.0+exp(-x[i]))); end; y)
epseq(x,y)=(maximum(abs(x-y)) < 1e-7)
x = rand(5,4,3,2) - 0.5; tx = CudaArray(x)
y = zeros(5,4,3,2); ty = CudaArray(y)
@test epseq(to_host(cudnnActivationForward(tx, ty, mode=CUDNN_ACTIVATION_SIGMOID)), mysigm(x, y))

dy = (rand(5,4,3,2) - 0.5); tdy = CudaArray(dy)
dx = zeros(5,4,3,2); tdx = CudaArray(dx)
mysigm(y,dy,dx)=(for i=1:length(dx); dx[i]=dy[i]*y[i]*(1.0-y[i]); end; dx)
@test epseq(to_host(cudnnActivationBackward(ty, tdy, tx, tdx, mode=CUDNN_ACTIVATION_SIGMOID)), mysigm(y,dy,dx))

using CUDNN: CUDNN_ACTIVATION_TANH
mytanh(x,y)=(for i=1:length(y); y[i]=tanh(x[i]); end; y)
x = rand(5,4,3,2) - 0.5; tx = CudaArray(x)
y = zeros(5,4,3,2); ty = CudaArray(y)
@test epseq(to_host(cudnnActivationForward(tx, ty, mode=CUDNN_ACTIVATION_TANH)), mytanh(x, y))

dy = (rand(5,4,3,2) - 0.5); tdy = CudaArray(dy)
dx = zeros(5,4,3,2); tdx = CudaArray(dx)
mytanh(y,dy,dx)=(for i=1:length(dx); dx[i]=dy[i]*(1.0+y[i])*(1.0-y[i]); end; dx)
@test epseq(to_host(cudnnActivationBackward(ty, tdy, tx, tdx, mode=CUDNN_ACTIVATION_TANH)), mytanh(y,dy,dx))


using CUDNN: cudnnSoftmaxForward, CUDNN_SOFTMAX_MODE_CHANNEL, CUDNN_SOFTMAX_MODE_INSTANCE
x = (rand(5,4,3,2) - 0.5)
tx = CudaArray(x)
Base.zeros(a::AbstractCudaArray)=cudnnSetTensor(similar(a), zero(eltype(a)))
ty = zeros(tx)
@test epseq(to_host(cudnnSoftmaxForward(tx, ty; mode=CUDNN_SOFTMAX_MODE_INSTANCE)), exp(x)./sum(exp(x), (1,2,3)))
@test epseq(to_host(cudnnSoftmaxForward(tx, ty; mode=CUDNN_SOFTMAX_MODE_CHANNEL)), exp(x)./sum(exp(x), 3))


# Do we feed the gold probabilities p or the output gradients (1-p/q) to cudnnSoftmaxBackward?
# It turns out we feed 1-p/q, which is numerically unstable.
# x: unnormalized logp
# q: normalized estimates  qi=(exp xi)/(Σ exp xj)  q=softmaxForward(x)
# p: gold answers
# J = -Σ p log q
# dJ/dq = 1-p/q
# dJ/dx = q-p

using CUDNN: cudnnSoftmaxBackward
# Model probabilities:
x = (rand(5,4) - 0.5); tx = CudaArray(reshape(x,(1,1,5,4))); tq = zeros(tx)
cudnnSoftmaxForward(tx,tq); q = squeeze(to_host(tq),(1,2))
# Gold probabilities:
z = (rand(5,4) - 0.5); tz = CudaArray(reshape(z,(1,1,5,4))); tp = zeros(tz)
cudnnSoftmaxForward(tz,tp); p = squeeze(to_host(tp),(1,2))
# Gradient wrt softmax output:
dq = 1-p./q; tdq = CudaArray(reshape(dq,(1,1,5,4)))
# Gradient wrt softmax input:
tdx = similar(tdq)
cudnnSoftmaxBackward(tq, tdq, tdx)
dx = squeeze(to_host(tdx),(1,2))
@test epseq(dx,q-p)


# Discovering what pooling does:
# using CUDNN: cudnnPoolingDescriptor_t, cudnnCreatePoolingDescriptor, cudnnSetPooling2dDescriptor
# # x = rand(5,4,1,1)
# x = reshape(Float64[1:20;], 5, 4, 1, 1)
# tx = CudaArray(x)
# pdptr = Array(cudnnPoolingDescriptor_t, 1)
# cudnnCreatePoolingDescriptor(pdptr)
# pd = pdptr[1]
# using CUDNN: CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_MAX
# #cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 3, 3, 0, 0, 1, 1)
# cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 3, 3, 0, 0, 1, 1)
# #cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_MAX, 3, 3, 0, 0, 1, 1)
# ty = CudaArray(ones(10,9,1,1))
# using CUDNN: cudnnHandle, ptr, cudnnPoolingForward
# cudnnPoolingForward(cudnnHandle, pd, ptr(1,tx), tx.desc, tx.data.ptr, ptr(0,ty), ty.desc, ty.data.ptr)
# y = to_host(ty)
# dump(x)
# dump(y)

using CUDNN: CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN
using CUDNN: PD, cudnnGetPoolingNdDescriptor
pd = PD(2, 3, 2, 1, CUDNN_POOLING_MAX)
if CUDNN_VERSION >= 4000
    @test cudnnGetPoolingNdDescriptor(pd) == (CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, (3,3), (2,2), (1,1))
else
    @test cudnnGetPoolingNdDescriptor(pd) == (CUDNN_POOLING_MAX, 2, (3,3), (2,2), (1,1))
end
# free(pd)

using CUDNN: cudnnPoolingForward, cudnnPoolingBackward, cudnnGetPoolingNdForwardOutputDim, cudnnGetPoolingNdForwardOutputDim_buggy
x = reshape(Float64[1:20;], 5, 4, 1, 1); tx = CudaArray(x)

# 1.0   6.0  11.0  16.0
# 2.0   7.0  12.0  17.0
# 3.0   8.0  13.0  18.0
# 4.0   9.0  14.0  19.0
# 5.0  10.0  15.0  20.0

# 3 size, 0 pad, 1 stride
ty1 = CudaArray(zeros(3, 2, 1, 1))
pd1 = PD(2, 3, 0, 1, CUDNN_POOLING_MAX)
@test cudnnGetPoolingNdForwardOutputDim(pd1, tx) == (3,2,1,1)
@test cudnnGetPoolingNdForwardOutputDim_buggy(pd1, tx) == (3,2,1,1)
@test squeeze(to_host(cudnnPoolingForward(tx, ty1; pd=pd1)),(3,4)) == [13 18; 14 19; 15 20.]
ty2 = CudaArray(zeros(3, 2, 1, 1))
pd2 = PD(2, 3, 0, 1, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd2, tx) == (3,2,1,1)
@test cudnnGetPoolingNdForwardOutputDim_buggy(pd2, tx) == (3,2,1,1)
@test squeeze(to_host(cudnnPoolingForward(tx, ty2; pd=pd2)),(3,4)) == [7 12; 8 13; 9 14.]
ty3 = CudaArray(zeros(3, 2, 1, 1))
pd3 = PD(2, 3, 0, 1, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd3, tx) == (3,2,1,1)
@test cudnnGetPoolingNdForwardOutputDim_buggy(pd3, tx) == (3,2,1,1)
@test squeeze(to_host(cudnnPoolingForward(tx, ty3, pd=pd3)),(3,4)) == [7 12; 8 13; 9 14.]

dy1 = reshape(Float64[1:6;], 3, 2, 1, 1); 
tdy1 = CudaArray(dy1)
tdx1 = zeros(tx)
@test squeeze(to_host(cudnnPoolingBackward(ty1, tdy1, tx, tdx1; pd=pd1)),(3,4)) == [0 0 0 0;0 0 0 0;0 0 1 4;0 0 2 5;0 0 3 6.]
tdx2 = zeros(tx)
@test epseq(squeeze(to_host(cudnnPoolingBackward(ty2, tdy1, tx, tdx2; pd=pd2)),(3,4)), [1/9 5/9 5/9 4/9;3/9 12/9 12/9 9/9;6/9 21/9 21/9 15/9;5/9 16/9 16/9 11/9;3/9 9/9 9/9 6/9])
tdx3 = zeros(tx)
@test epseq(squeeze(to_host(cudnnPoolingBackward(ty3, tdy1, tx, tdx3; pd=pd3)),(3,4)), [1/9 5/9 5/9 4/9;3/9 12/9 12/9 9/9;6/9 21/9 21/9 15/9;5/9 16/9 16/9 11/9;3/9 9/9 9/9 6/9])

# 3 size, 1 pad, 1 stride
ty4 = CudaArray(zeros(5, 4, 1, 1))
pd4 = PD(2, 3, 1, 1, CUDNN_POOLING_MAX)
@test cudnnGetPoolingNdForwardOutputDim(pd4, tx) == (5,4,1,1)
@test cudnnGetPoolingNdForwardOutputDim_buggy(pd4, tx) == (5,4,1,1)
@test squeeze(to_host(cudnnPoolingForward(tx, ty4; pd=pd4)),(3,4)) == [7 12 17 17; 8 13 18 18; 9 14 19 19; 10 15 20 20; 10 15 20 20.]
ty5 = CudaArray(zeros(5, 4, 1, 1))
pd5 = PD(2, 3, 1, 1, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd5, tx) == (5,4,1,1)
@test cudnnGetPoolingNdForwardOutputDim_buggy(pd5, tx) == (5,4,1,1)
@test squeeze(to_host(cudnnPoolingForward(tx, ty5; pd=pd5)),(3,4)) == [16/9 39/9 69/9 56/9; 3 7 12 87/9; 33/9 8 13 93/9; 39/9 9 14 11; 28/9 57/9 87/9 68/9]
ty6 = CudaArray(zeros(5, 4, 1, 1))
pd6 = PD(2, 3, 1, 1, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd6, tx) == (5,4,1,1)
@test cudnnGetPoolingNdForwardOutputDim_buggy(pd6, tx) == (5,4,1,1)
@test squeeze(to_host(cudnnPoolingForward(tx, ty6; pd=pd6)),(3,4)) == [16/4 39/6 69/6 56/4; 27/6 7 12 87/6; 33/6 8 13 93/6; 39/6 9 14 99/6; 28/4 57/6 87/6 68/4]

dy4 = reshape(Float64[1:20;], 5, 4, 1, 1); 
tdy4 = CudaArray(dy4)
tdx4 = zeros(tx)
@test squeeze(to_host(cudnnPoolingBackward(ty4, tdy4, tx, tdx4; pd=pd4)),(3,4)) == [0 0 0 0;0 1 6 11+16;0 2 7 12+17;0 3 8 13+18;0 4+5 9+10 14+15+19+20.]
tdx5 = zeros(tx)
@test epseq(squeeze(to_host(cudnnPoolingBackward(ty5, tdy4, tx, tdx5; pd=pd5)),(3,4)), [16/9 39/9 69/9 56/9; 3 7 12 87/9; 33/9 8 13 93/9; 39/9 9 14 11; 28/9 57/9 87/9 68/9])
tdx6 = zeros(tx)
@test epseq(squeeze(to_host(cudnnPoolingBackward(ty6, tdy4, tx, tdx6; pd=pd6)),(3,4)), [2.361111111111111 5.527777777777778 11.777777777777779 10.0;  3.75 8.36111111111111 17.11111111111111 14.444444444444445;  4.166666666666666 8.5 16.0 13.333333333333332;  5.972222222222222 11.472222222222221 20.22222222222222 16.666666666666668; 4.583333333333334 8.63888888888889 14.88888888888889 12.222222222222221])

# 3 size, 1 pad, 2 stride
ty7 = CudaArray(zeros(3, 3, 1, 1))
pd7 = PD(2, 3, 1, 2, CUDNN_POOLING_MAX)
@test cudnnGetPoolingNdForwardOutputDim(pd7, tx) == (3,3,1,1)
@show cudnnGetPoolingNdForwardOutputDim_buggy(pd7, tx), (3,3,1,1)
@test squeeze(to_host(cudnnPoolingForward(tx, ty7; pd=pd7)),(3,4)) == [7 17 17; 9 19 19; 10 20 20.]
# Note below that if the window falls outside the padding, the denom can be less than 9!
ty8 = CudaArray(zeros(3, 3, 1, 1))
pd8 = PD(2, 3, 1, 2, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd8, tx) == (3,3,1,1)
@show cudnnGetPoolingNdForwardOutputDim_buggy(pd8, tx), (3,3,1,1)
@test squeeze(to_host(cudnnPoolingForward(tx, ty8; pd=pd8)),(3,4)) == [16/9 69/9 33/6; 33/9 13 54/6; 28/9 87/9 39/6]
ty9 = CudaArray(zeros(3, 3, 1, 1))
pd9 = PD(2, 3, 1, 2, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd9, tx) == (3,3,1,1)
@show cudnnGetPoolingNdForwardOutputDim_buggy(pd9, tx), (3,3,1,1)
@test squeeze(to_host(cudnnPoolingForward(tx, ty9; pd=pd9)),(3,4)) == [16/4 69/6 33/2; 33/6 13 54/3; 28/4 87/6 39/2]

dy7 = reshape(Float64[1:9;], 3, 3, 1, 1); 
tdy7 = CudaArray(dy7)
tdx7 = zeros(tx)
# TODO: check this answer
@test squeeze(to_host(cudnnPoolingBackward(ty7, tdy7, tx, tdx7; pd=pd7)),(3,4)) == [0 0 0 0;0 1 0 11;0 0 0 0;0 2 0 13;0 3 0 15.]
tdx8 = zeros(tx)
# TODO: check this answer
@test epseq(squeeze(to_host(cudnnPoolingBackward(ty8, tdy7, tx, tdx8; pd=pd8)),(3,4)), [1/9 5/9 4/9 29/18; 1/3 4/3 1 7/2; 2/9 7/9 5/9 17/9; 5/9 16/9 11/9 73/18; 1/3 1 2/3 39/18])
tdx9 = zeros(tx)
@test epseq(squeeze(to_host(cudnnPoolingBackward(ty9, tdy7, tx, tdx9; pd=pd9)),(3,4)), [0.25 0.9166666666666666 0.6666666666666666 4.166666666666667; 0.5833333333333333 1.8055555555555554 1.2222222222222223 7.388888888888889; 0.3333333333333333 0.8888888888888888 0.5555555555555556 3.2222222222222223; 1.0833333333333333 2.638888888888889 1.5555555555555556 8.722222222222221; 0.75 1.75 1.0 5.5])

# 3D pooling support added in v3
x10 = reshape(Float64[1:60;], 5, 4, 3, 1, 1); tx10 = CudaArray(x10)
ty10 = CudaArray(zeros(3, 2, 1, 1, 1))
pd10 = PD(3, 3, 0, 1, CUDNN_POOLING_MAX)
# TODO: add 3d pooling tests
# @show cudnnPoolingForward(tx10, ty10; pd=pd10)


# Filters are basically the same as tensors:

using CUDNN: cudnnFilterDescriptor_t, cudnnCreateFilterDescriptor, cudnnSetFilter4dDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, cudnnDataType_t, cudnnGetFilter4dDescriptor
d = cudnnFilterDescriptor_t[0]
cudnnCreateFilterDescriptor(d)
cudnnSetFilter4dDescriptor(d[1], CUDNN_DATA_DOUBLE, 2, 3, 4, 5)
dt = cudnnDataType_t[0]
for n in (:sk, :sc, :sh, :sw); @eval $n = Cint[0]; end
cudnnGetFilter4dDescriptor(d[1],dt,sk,sc,sh,sw)
@test (dt[1],sn[1],sc[1],sh[1],sw[1],tn[1],tc[1],th[1],tw[1]) == (CUDNN_DATA_DOUBLE, 2, 3, 4, 5, 60, 20, 5, 1)


using CUDNN: cudnnGetFilterNdDescriptor
nd=4; nbDims=Cint[0]; dimA=Array(Cint,nd)
cudnnGetFilterNdDescriptor(d[1],nd,dt,nbDims,dimA)
@test (nbDims, dimA) == (Cint[4], Cint[2,3,4,5])


using CUDNN: FD
x = rand(5,4,3,2)
tx = CudaArray(x)
@test to_host(tx) == x
@test cudnnGetFilterNdDescriptor(FD(tx)) == (CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 4, (reverse(size(x))...))

# Convolution

using CUDNN: CD, cudnnGetConvolutionNdDescriptor, CUDNN_CONVOLUTION
cd = CD(2, 0, 1, 1, CUDNN_CONVOLUTION, Float32)
if CUDNN_VERSION >= 3000
@test cudnnGetConvolutionNdDescriptor(cd) == (2, (0,0), (1,1), (1,1), CUDNN_CONVOLUTION, Float32)
else # if CUDNN_VERSION >= 3000
@test cudnnGetConvolutionNdDescriptor(cd) == (2, (0,0), (1,1), (1,1), CUDNN_CONVOLUTION)
end # if CUDNN_VERSION >= 3000
# Note: upscale other than (1,1) gives unsupported error.
# Note: not sure if we need to expose the ConvolutionDescriptor or just have options for convolution.
# Note: need to understand upscale.  how often do we need non-default padding and stride?
# Note: conv vs xcor

using CUDNN: cudnnGetConvolutionNdForwardOutputDim
@test cudnnGetConvolutionNdForwardOutputDim(CudaArray(ones(12,8,3,6)), CudaArray(ones(5,4,3,2))) == (8,5,2,6)
# Does not work for dimensions other than 4 yet:
# @show cudnnGetConvolutionNdForwardOutputDim(CudaArray(ones(13,12,11,3,10)), CudaArray(ones(6,5,4,3,2)))

using CUDNN: CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
using CUDNN: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, CUDNN_CONVOLUTION_FWD_ALGO_FFT, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING

using CUDNN: cudnnGetConvolutionForwardAlgorithm
src = CudaArray(ones(102,101,3,100))
flt = CudaArray(ones(25,15,3,99))
dst = CudaArray(ones(cudnnGetConvolutionNdForwardOutputDim(src, flt)))
@test cudnnGetConvolutionForwardAlgorithm(src, flt, dst; preference=CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
@test cudnnGetConvolutionForwardAlgorithm(src, flt, dst; preference=CUDNN_CONVOLUTION_FWD_PREFER_FASTEST) == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
@test cudnnGetConvolutionForwardAlgorithm(src, flt, dst; preference=CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT) == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM

# TODO: check if the 0 answers are accurate
using CUDNN: cudnnGetConvolutionForwardWorkspaceSize
@test cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) == 0
@test cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) == 4500 # size(flt,1)*size(flt,2)*size(flt,3)*sizeof(Cint)
@test cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_GEMM) == 1812432704 # this is from v4, v3:6107400000

# These get "CUDNN_STATUS_NOT_SUPPORTED" in v4:
# @test cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_DIRECT) == 0
# @test cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_FFT) == 0
# @test cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) == 0

using CUDNN: cudnnConvolutionForward
x = reshape(Float64[1:20;], 5, 4, 1, 1); tx = CudaArray(x)
w = reshape(Float64[1:4;], 2, 2, 1, 1); tw = CudaArray(w)
@test squeeze(to_host(cudnnConvolutionForward(tx, tw)),(3,4)) == [29 79 129; 39 89 139; 49 99 149; 59 109 159.]

using CUDNN: CD, CUDNN_CROSS_CORRELATION
cdesc = CD(tx; mode=CUDNN_CROSS_CORRELATION)
@test squeeze(to_host(cudnnConvolutionForward(tx, tw; cd=cdesc)),(3,4)) == [51 101 151;61 111 161;71 121 171;81 131 181.]

using CUDNN: cudnnConvolutionBackwardBias, cudnnConvolutionBackwardFilter, cudnnConvolutionBackwardData
x = rand(5,4,3,2); tx = CudaArray(x)
w = rand(2,2,3,4); tw = CudaArray(w)
ty = cudnnConvolutionForward(tx, tw)
dy = rand(size(ty)); tdy = CudaArray(dy)
@test epseq(to_host(cudnnConvolutionBackwardBias(tdy)), sum(dy,(1,2,4)))

# TODO: put a more meaningful test here...
tdx = zeros(tx)
@test size(cudnnConvolutionBackwardData(tw, tdy, tdx)) == size(x)

tdw = zeros(tw)
@test size(cudnnConvolutionBackwardFilter(tx, tdy, tdw)) == size(w)

# Testing correspondence with conv2
x = rand(5,4); tx = CudaArray(reshape(x, (5,4,1,1)))
w = rand(3,3); tw = CudaArray(reshape(w, (3,3,1,1)))
padding=map(x->x-1,size(w))
@test epseq(squeeze(to_host(cudnnConvolutionForward(tx, tw; cd=CD(tx; padding=padding))),(3,4)), conv2(x,w))
@test epseq(squeeze(to_host(conv2(tx, tw)), (3,4)), conv2(x,w))

if CUDNN_VERSION >= 3000
    # LRN: TODO: add tests
    x = rand(5,4); tx = CudaArray(reshape(x, (5,4,1,1)))
    ty = similar(tx)
    cudnnLRNCrossChannelForward(tx,ty)
    y = squeeze(to_host(ty), (3,4))
    dy = rand(5,4); tdy = CudaArray(reshape(x, (5,4,1,1)))
    tdx = similar(tdy)
    cudnnLRNCrossChannelBackward(ty, tdy, tx, tdx)
    dx = squeeze(to_host(tdx), (3,4))


    # DivisiveNormalization
    x = rand(5,4); tx = CudaArray(reshape(x, (5,4,1,1)))
    ty = similar(tx)
    cudnnDivisiveNormalizationForward(tx,ty)
    y = squeeze(to_host(ty), (3,4))
    dy = rand(5,4); tdy = CudaArray(reshape(x, (5,4,1,1)))
    tdx = similar(tdy)
    cudnnDivisiveNormalizationBackward(ty, tdy, tdx)
    dx = squeeze(to_host(tdx), (3,4))

    # println(x)
    # println(y)
    # println(dy)
    # println(dx)
end # if CUDNN_VERSION >= 3000

:ok
