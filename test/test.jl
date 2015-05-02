using Base.Test

# Comment this out if you don't want lots of messages:
# Base.Test.default_handler(r::Base.Test.Success) = info("$(r.expr)")

# See which operations support which dimensions:
function testdims()
    for n=3:8
        @show n
        try
            dims = [(n+1):-1:2]
            x = CUDNN.Tensor(rand(dims...))
            y = CUDNN.Tensor(rand(dims...))
            dx = CUDNN.Tensor(rand(dims...))
            dy = CUDNN.Tensor(rand(dims...))
            CUDNN.cudnnActivationBackward(CUDNN.CUDNN_ACTIVATION_RELU, y, dy, x, dx)
            CUDNN.cudnnActivationForward(CUDNN.CUDNN_ACTIVATION_RELU, x)
            CUDNN.cudnnScaleTensor(x, pi)
            CUDNN.cudnnSetTensor(x, pi)
            CUDNN.cudnnTransformTensor(2.0, x, 3.0, y)
            bdim = ones(dims); bdim[1]=dims[1]; bdim[2]=dims[2]
            bias = CUDNN.Tensor(rand(bdim...))
            CUDNN.cudnnAddTensor(CUDNN.CUDNN_ADD_IMAGE, 1, bias, 1, x)
        catch y
            println(y)
        end
    end
end

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


using CUDNN: Tensor, to_host
x = rand(5,4,3,2)
tx = Tensor(x)
@test to_host(tx) == x
@test cudnnGetTensorNdDescriptor(tx) == (CUDNN_DATA_DOUBLE, 4, [reverse(size(x))...], [reverse(strides(x))...])


using CUDNN: cudnnTransformTensor
y = rand(5,4,3,2)
ty = Tensor(y)
@test to_host(cudnnTransformTensor(2, tx, 3, ty)) == 2x+3y


using CUDNN: cudnnAddTensor, CUDNN_ADD_IMAGE
b = rand(5,4,1,1)
tb = Tensor(b)
@test to_host(cudnnAddTensor(CUDNN_ADD_IMAGE, 1, tb, 1, tx)) == x .+ b


using CUDNN: cudnnSetTensor
@test to_host(cudnnSetTensor(tx, pi)) == fill!(ones(size(tx)), pi)


using CUDNN: cudnnScaleTensor
x = rand(5,4,3,2)
tx = Tensor(x)
@test to_host(cudnnScaleTensor(tx, pi)) == x .* pi


using CUDNN: cudnnActivationForward, CUDNN_ACTIVATION_RELU
myrelu(x,y)=(copy!(y,x);for i=1:length(y); (y[i]<zero(y[i]))&&(y[i]=zero(y[i])); end; y)
x = rand(5,4,3,2) - 0.5; tx = Tensor(x)
y = zeros(5,4,3,2); ty = Tensor(y)
@test to_host(cudnnActivationForward(tx, ty, mode=CUDNN_ACTIVATION_RELU)) == myrelu(x, y)

using CUDNN: cudnnActivationBackward
dy = (rand(5,4,3,2) - 0.5); tdy = Tensor(dy)
dx = zeros(5,4,3,2); tdx = Tensor(dx)
myrelu(y,dy,dx)=(copy!(dx,dy);for i=1:length(y); (y[i]==zero(y[i]))&&(dx[i]=zero(dx[i])); end; dx)
@test to_host(cudnnActivationBackward(ty, tdy, tx, tdx, mode=CUDNN_ACTIVATION_RELU)) == myrelu(y,dy,dx)

using CUDNN: CUDNN_ACTIVATION_SIGMOID
mysigm(x,y)=(for i=1:length(y); y[i]=(1.0/(1.0+exp(-x[i]))); end; y)
epseq(x,y)=(maximum(abs(x-y)) < 1e-15)
x = rand(5,4,3,2) - 0.5; tx = Tensor(x)
y = zeros(5,4,3,2); ty = Tensor(y)
@test epseq(to_host(cudnnActivationForward(tx, ty, mode=CUDNN_ACTIVATION_SIGMOID)), mysigm(x, y))

dy = (rand(5,4,3,2) - 0.5); tdy = Tensor(dy)
dx = zeros(5,4,3,2); tdx = Tensor(dx)
mysigm(y,dy,dx)=(for i=1:length(dx); dx[i]=dy[i]*y[i]*(1.0-y[i]); end; dx)
@test epseq(to_host(cudnnActivationBackward(ty, tdy, tx, tdx, mode=CUDNN_ACTIVATION_SIGMOID)), mysigm(y,dy,dx))

using CUDNN: CUDNN_ACTIVATION_TANH
mytanh(x,y)=(for i=1:length(y); y[i]=tanh(x[i]); end; y)
x = rand(5,4,3,2) - 0.5; tx = Tensor(x)
y = zeros(5,4,3,2); ty = Tensor(y)
@test epseq(to_host(cudnnActivationForward(tx, ty, mode=CUDNN_ACTIVATION_TANH)), mytanh(x, y))

dy = (rand(5,4,3,2) - 0.5); tdy = Tensor(dy)
dx = zeros(5,4,3,2); tdx = Tensor(dx)
mytanh(y,dy,dx)=(for i=1:length(dx); dx[i]=dy[i]*(1.0+y[i])*(1.0-y[i]); end; dx)
@test epseq(to_host(cudnnActivationBackward(ty, tdy, tx, tdx, mode=CUDNN_ACTIVATION_TANH)), mytanh(y,dy,dx))


using CUDNN: cudnnSoftmaxForward
x = (rand(1,1,4,5) - 0.5)       # 5 instances with 4 classes each
tx = Tensor(x)
ty = zeros(tx)
@test epseq(to_host(cudnnSoftmaxForward(tx, ty)), exp(x)./sum(exp(x), 3))
y = to_host(ty)

using CUDNN: cudnnSoftmaxBackward
# If y[c,j] is the probability of the correct answer for the j'th instance:
# dy[c,j] = -1/y[c,j]
# dy[i!=c,j] = 1/y[c,j]
r = rand(1:size(y,3), size(y,4)) # Random answer key
c = sub2ind((size(y,3),size(y,4)), r, 1:size(y,4)) # indices of correct answers
p = y[c] # probabilities of correct answers
dy = zeros(y)
for i=1:size(dy,3); dy[:,:,i,:] = 1./p; end  # dy = 1/p for incorrect answers
dy[c] *= -1.0 # dy = -1/p for correct answers
dy *= 0.5  # this is a cudnn bug
tdy = Tensor(dy)
tdx = zeros(tdy)
# We should have dx = y-1 for correct answers, dx = y for wrong answers
dx = copy(y); dx[c] .-= 1
@test epseq(to_host(cudnnSoftmaxBackward(ty, tdy, tdx)), dx)

# Discovering what pooling does:
# using CUDNN: cudnnPoolingDescriptor_t, cudnnCreatePoolingDescriptor, cudnnSetPooling2dDescriptor
# # x = rand(5,4,1,1)
# x = reshape(Float64[1:20], 5, 4, 1, 1)
# tx = Tensor(x)
# pdptr = Array(cudnnPoolingDescriptor_t, 1)
# cudnnCreatePoolingDescriptor(pdptr)
# pd = pdptr[1]
# using CUDNN: CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_MAX
# #cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 3, 3, 0, 0, 1, 1)
# cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 3, 3, 0, 0, 1, 1)
# #cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_MAX, 3, 3, 0, 0, 1, 1)
# ty = Tensor(ones(10,9,1,1))
# using CUDNN: cudnnHandle, ptr, cudnnPoolingForward
# cudnnPoolingForward(cudnnHandle, pd, ptr(1,tx), tx.desc, tx.data.ptr, ptr(0,ty), ty.desc, ty.data.ptr)
# y = to_host(ty)
# dump(x)
# dump(y)

using CUDNN: PoolingDescriptor, free, CUDNN_POOLING_MAX, cudnnGetPoolingNdDescriptor
pd = PoolingDescriptor((3,3); padding=(2,2), stride=(1,1), mode=CUDNN_POOLING_MAX)
@test cudnnGetPoolingNdDescriptor(pd) == (CUDNN_POOLING_MAX, length(pd.dims), pd.dims, pd.padding, pd.stride)
# free(pd)

# Test forward and backward:

:ok
