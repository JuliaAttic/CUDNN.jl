using Base.Test
using CUDArt

# Uncomment this if you want lots of messages:
Base.Test.default_handler(r::Base.Test.Success) = info("$(r.expr)")

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


using CUDNN: cudnnTensorDescriptor_t, cudnnCreateTensorDescriptor, cudnnSetTensor4dDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, cudnnDataType_t, cudnnGetTensor4dDescriptor
d = cudnnTensorDescriptor_t[0]
cudnnCreateTensorDescriptor(d)
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
x = rand(5,4,3,2)
tx = CudaArray(x)
@test to_host(tx) == x
@test cudnnGetTensorNdDescriptor(TD(tx)) == (CUDNN_DATA_DOUBLE, 4, [reverse(size(x))...], [reverse(strides(x))...])


using CUDNN: cudnnTransformTensor
y = rand(5,4,3,2)
ty = CudaArray(y)
@test to_host(cudnnTransformTensor(2, tx, 3, ty)) == 2x+3y


using CUDNN: cudnnAddTensor
b = rand(1,1,3,1)
tb = CudaArray(b)
@test to_host(cudnnAddTensor(tb,tx)) == x .+ b


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
epseq(x,y)=(maximum(abs(x-y)) < 1e-14)
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

using CUDNN: cudnnSoftmaxBackward
# If y[c,j] is the probability of the correct answer for the j'th instance:
# dy[c,j] = -1/y[c,j]
# dy[i!=c,j] = 1/y[c,j]
x = (rand(1,1,5,4) - 0.5); tx = CudaArray(x); ty = zeros(tx)
cudnnSoftmaxForward(tx,ty); y = to_host(ty)
r = rand(1:size(y,3), size(y,4)) # Random answer key
c = sub2ind((size(y,3),size(y,4)), r, 1:size(y,4)) # indices of correct answers
p = y[c] # probabilities of correct answers
dy = zeros(y)
for i=1:size(dy,3); dy[:,:,i,:] = 1./p; end  # dy = 1/p for incorrect answers
dy[c] *= -1.0 # dy = -1/p for correct answers
dy *= 0.5  # this is a cudnn bug
tdy = CudaArray(dy)
tdx = zeros(tdy)
# We should have dx = y-1 for correct answers, dx = y for wrong answers
dx = copy(y); dx[c] .-= 1
@test epseq(to_host(cudnnSoftmaxBackward(ty, tdy, tdx)), dx)

# Discovering what pooling does:
# using CUDNN: cudnnPoolingDescriptor_t, cudnnCreatePoolingDescriptor, cudnnSetPooling2dDescriptor
# # x = rand(5,4,1,1)
# x = reshape(Float64[1:20], 5, 4, 1, 1)
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

using CUDNN: CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
using CUDNN: PoolingDescriptor, free, cudnnGetPoolingNdDescriptor
pd = PoolingDescriptor((3,3); padding=(2,2), stride=(1,1), mode=CUDNN_POOLING_MAX)
@test cudnnGetPoolingNdDescriptor(pd) == (CUDNN_POOLING_MAX, length(pd.dims), pd.dims, pd.padding, pd.stride)
# free(pd)

using CUDNN: cudnnPoolingForward, cudnnPoolingBackward, cudnnGetPoolingNdForwardOutputDim
x = reshape(Float64[1:20], 5, 4, 1, 1); tx = CudaArray(x)

# 1.0   6.0  11.0  16.0
# 2.0   7.0  12.0  17.0
# 3.0   8.0  13.0  18.0
# 4.0   9.0  14.0  19.0
# 5.0  10.0  15.0  20.0

# 3 size, 0 pad, 1 stride
ty1 = CudaArray(zeros(3, 2, 1, 1))
pd1 = PoolingDescriptor((3,3); padding=(0,0), stride=(1,1), mode=CUDNN_POOLING_MAX)
@test cudnnGetPoolingNdForwardOutputDim(pd1, tx) == (3,2,1,1)
@test squeeze(to_host(cudnnPoolingForward(pd1, tx, ty1)),(3,4)) == [13 18; 14 19; 15 20.]
ty2 = CudaArray(zeros(3, 2, 1, 1))
pd2 = PoolingDescriptor((3,3); padding=(0,0), stride=(1,1), mode=CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd2, tx) == (3,2,1,1)
@test squeeze(to_host(cudnnPoolingForward(pd2, tx, ty2)),(3,4)) == [7 12; 8 13; 9 14.]
ty3 = CudaArray(zeros(3, 2, 1, 1))
pd3 = PoolingDescriptor((3,3); padding=(0,0), stride=(1,1), mode=CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd3, tx) == (3,2,1,1)
@test squeeze(to_host(cudnnPoolingForward(pd3, tx, ty3)),(3,4)) == [7 12; 8 13; 9 14.]

dy1 = reshape(Float64[1:6], 3, 2, 1, 1); 
tdy1 = CudaArray(dy1)
tdx1 = zeros(tx)
@test squeeze(to_host(cudnnPoolingBackward(pd1, ty1, tdy1, tx, tdx1)),(3,4)) == [0 0 0 0;0 0 0 0;0 0 1 4;0 0 2 5;0 0 3 6.]
tdx2 = zeros(tx)
@test epseq(squeeze(to_host(cudnnPoolingBackward(pd2, ty2, tdy1, tx, tdx2)),(3,4)), [1/9 5/9 5/9 4/9;3/9 12/9 12/9 9/9;6/9 21/9 21/9 15/9;5/9 16/9 16/9 11/9;3/9 9/9 9/9 6/9])
tdx3 = zeros(tx)
@test epseq(squeeze(to_host(cudnnPoolingBackward(pd3, ty3, tdy1, tx, tdx3)),(3,4)), [1/9 5/9 5/9 4/9;3/9 12/9 12/9 9/9;6/9 21/9 21/9 15/9;5/9 16/9 16/9 11/9;3/9 9/9 9/9 6/9])

# 3 size, 1 pad, 1 stride
ty4 = CudaArray(zeros(5, 4, 1, 1))
pd4 = PoolingDescriptor((3,3); padding=(1,1), stride=(1,1), mode=CUDNN_POOLING_MAX)
@test cudnnGetPoolingNdForwardOutputDim(pd4, tx) == (5,4,1,1)
@test squeeze(to_host(cudnnPoolingForward(pd4, tx, ty4)),(3,4)) == [7 12 17 17; 8 13 18 18; 9 14 19 19; 10 15 20 20; 10 15 20 20.]
ty5 = CudaArray(zeros(5, 4, 1, 1))
pd5 = PoolingDescriptor((3,3); padding=(1,1), stride=(1,1), mode=CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd5, tx) == (5,4,1,1)
@test squeeze(to_host(cudnnPoolingForward(pd5, tx, ty5)),(3,4)) == [16/9 39/9 69/9 56/9; 3 7 12 87/9; 33/9 8 13 93/9; 39/9 9 14 11; 28/9 57/9 87/9 68/9]
# This is buggy in the library:
ty6 = CudaArray(zeros(5, 4, 1, 1))
pd6 = PoolingDescriptor((3,3); padding=(1,1), stride=(1,1), mode=CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
cudnnPoolingForward(pd6, tx, ty6)
@test cudnnGetPoolingNdForwardOutputDim(pd6, tx) == (5,4,1,1)
# @test squeeze(to_host(cudnnPoolingForward(pd6, tx, ty6)),(3,4)) == [16/4 39/6 69/6 56/4; 27/6 7 12 87/6; 33/6 8 13 93/6; 39/6 9 14 99/6; 28/4 57/6 87/6 68/4]
# dump( squeeze(to_host(cudnnPoolingForward(pd6, tx, ty6)),(3,4)) )
# dump( [16/4 39/6 69/6 56/4; 27/6 7 12 87/6; 33/6 8 13 93/6; 39/6 9 14 99/6; 28/4 57/6 87/6 68/4] )

dy4 = reshape(Float64[1:20], 5, 4, 1, 1); 
tdy4 = CudaArray(dy4)
tdx4 = zeros(tx)
@test squeeze(to_host(cudnnPoolingBackward(pd4, ty4, tdy4, tx, tdx4)),(3,4)) == [0 0 0 0;0 1 6 11+16;0 2 7 12+17;0 3 8 13+18;0 4+5 9+10 14+15+19+20.]
tdx5 = zeros(tx)
@test epseq(squeeze(to_host(cudnnPoolingBackward(pd5, ty5, tdy4, tx, tdx5)),(3,4)), [16/9 39/9 69/9 56/9; 3 7 12 87/9; 33/9 8 13 93/9; 39/9 9 14 11; 28/9 57/9 87/9 68/9])
tdx6 = zeros(tx)
# Buggy fails test:
cudnnPoolingBackward(pd6, ty6, tdy4, tx, tdx6)
# @show (squeeze(to_host(cudnnPoolingBackward(pd6, ty6, tdy4, tx, tdx6)),(3,4)), [1/9 5/9 5/9 4/9;3/9 12/9 12/9 9/9;6/9 21/9 21/9 15/9;5/9 16/9 16/9 11/9;3/9 9/9 9/9 6/9])

# 3 size, 1 pad, 2 stride
ty7 = CudaArray(zeros(3, 3, 1, 1))
pd7 = PoolingDescriptor((3,3); padding=(1,1), stride=(2,2), mode=CUDNN_POOLING_MAX)
@test cudnnGetPoolingNdForwardOutputDim(pd7, tx) == (3,3,1,1)
@test squeeze(to_host(cudnnPoolingForward(pd7, tx, ty7)),(3,4)) == [7 17 17; 9 19 19; 10 20 20.]
# Note below that if the window falls outside the padding, the denom can be less than 9!
ty8 = CudaArray(zeros(3, 3, 1, 1))
pd8 = PoolingDescriptor((3,3); padding=(1,1), stride=(2,2), mode=CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
@test cudnnGetPoolingNdForwardOutputDim(pd8, tx) == (3,3,1,1)
@test squeeze(to_host(cudnnPoolingForward(pd8, tx, ty8)),(3,4)) == [16/9 69/9 33/6; 33/9 13 54/6; 28/9 87/9 39/6]
# This is buggy in the library:
# ty9 = CudaArray(zeros(3, 3, 1, 1))
# pd9 = PoolingDescriptor((3,3); padding=(1,1), stride=(2,2), mode=CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
# dump( squeeze(to_host(cudnnPoolingForward(pd9, tx, ty9)),(3,4)) )
# dump( [16/4 69/6 33/2; 33/6 13 54/3; 28/4 87/6 39/2] )

# 3D pooling not supported:
# x10 = reshape(Float64[1:60], 5, 4, 3, 1, 1); tx10 = CudaArray(x10)
# ty10 = CudaArray(zeros(3, 2, 1, 1, 1))
# pd10 = PoolingDescriptor((3,3,3); padding=(0,0,0), stride=(1,1,1), mode=CUDNN_POOLING_MAX)
# @show cudnnPoolingForward(pd10, tx10, ty10)


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
@test cudnnGetFilterNdDescriptor(FD(tx)) == (CUDNN_DATA_DOUBLE, 4, [reverse(size(x))...])

# Convolution

using CUDNN: ConvolutionDescriptor, cudnnGetConvolutionNdDescriptor, CUDNN_CONVOLUTION
pd = ConvolutionDescriptor(padding=(0,0), stride=(1,1), upscale=(1,1), mode=CUDNN_CONVOLUTION)
@test cudnnGetConvolutionNdDescriptor(pd) == (length(pd.padding), pd.padding, pd.stride, pd.upscale, pd.mode)
# Note: upscale other than (1,1) gives unsupported error.
# Note: not sure if we need to expose the ConvolutionDescriptor or just have options for convolution.
# Note: need to understand upscale.  how often do we need non-default padding and stride?
# Note: conv vs xcor

using CUDNN: cudnnGetConvolutionNdForwardOutputDim
@test cudnnGetConvolutionNdForwardOutputDim(CudaArray(ones(12,8,3,6)), CudaArray(ones(5,4,3,2))) == (8,5,2,6)
# Does not work for dimensions other than 4 yet:
# @show cudnnGetConvolutionNdForwardOutputDim(CudaArray(ones(13,12,11,3,10)), CudaArray(ones(6,5,4,3,2)))

using CUDNN: CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
using CUDNN: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT

using CUDNN: cudnnGetConvolutionForwardAlgorithm
src = CudaArray(ones(102,101,3,100))
flt = CudaArray(ones(25,15,3,99))
dst = CudaArray(ones(cudnnGetConvolutionNdForwardOutputDim(src, flt)))
@test cudnnGetConvolutionForwardAlgorithm(src, flt, dst; preference=CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
@test cudnnGetConvolutionForwardAlgorithm(src, flt, dst; preference=CUDNN_CONVOLUTION_FWD_PREFER_FASTEST) == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
@test cudnnGetConvolutionForwardAlgorithm(src, flt, dst; preference=CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT) == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM

using CUDNN: cudnnGetConvolutionForwardWorkspaceSize
@test cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) == 0
@test cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) == 4500 # size(flt,1)*size(flt,2)*size(flt,3)*sizeof(Cint)
@test cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_GEMM) == 6107400000
# Not supported yet:
# @show cudnnGetConvolutionForwardWorkspaceSize(src, flt, dst; algorithm=CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)

using CUDNN: cudnnConvolutionForward
x = reshape(Float64[1:20], 5, 4, 1, 1); tx = CudaArray(x)
w = reshape(Float64[1:4], 2, 2, 1, 1); tw = CudaArray(w)
@test squeeze(to_host(cudnnConvolutionForward(tx, tw)),(3,4)) == [29 79 129; 39 89 139; 49 99 149; 59 109 159.]

using CUDNN: ConvolutionDescriptor, CUDNN_CROSS_CORRELATION
cdesc = ConvolutionDescriptor(mode=CUDNN_CROSS_CORRELATION)
@test squeeze(to_host(cudnnConvolutionForward(tx, tw; convDesc=cdesc)),(3,4)) == [51 101 151;61 111 161;71 121 171;81 131 181.]

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
@test epseq(squeeze(to_host(cudnnConvolutionForward(tx, tw; convDesc=ConvolutionDescriptor(padding=padding))),(3,4)), conv2(x,w))
@test epseq(squeeze(to_host(conv2(tx, tw)), (3,4)), conv2(x,w))

:ok
