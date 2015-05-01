using CUDArt
using CUDNN
using KUnet

#x = reshape(Float64[1:21], (1,1,3,7))
x = rand(1,1,4,5)
ex = exp(x)
sx = sum(ex,3)
px = ex ./ sx

y = CUDNN.Tensor(x)
CUDNN.cudnnSoftmaxForward(y)
@show maximum(abs(px - to_host(y)))

dy = to_host(y)
for j=1:size(dy,4)
    #y1 = 0.5 / dy[1,1,1,j]
    i=rand(1:size(dy,3))
    y1 = 0.5 / dy[1,1,i,j]
    dy[:,:,:,j] = y1
    dy[:,:,i,j] = -y1
end
dy = CUDNN.Tensor(dy)
dx = zeros(dy)
CUDNN.cudnnSoftmaxBackward(y, dy, dx)
dump(squeeze(to_host(y), (1,2)))
dump(squeeze(to_host(dx), (1,2)))
@show maximum(map(min, abs(to_host(y)-to_host(dx)), abs(to_host(y)-1.0-to_host(dx))))

if false

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


x = CUDNN.Tensor(rand(5,4,3,2))
y = copy(x)
CUDNN.cudnnActivationForward(CUDNN.CUDNN_ACTIVATION_RELU, y)

l1 = KUnet.Layer()
y1 = to_host(x)
KUnet.relu(l1,y1)
@show isequal(y1,to_host(y))

dy = CUDNN.Tensor(rand(5,4,3,2))
dx = copy(dy)
# x0h = to_host(x); x0h[1]=0; x0=CUDNN.Tensor(x0h)
CUDNN.cudnnActivationBackward(CUDNN.CUDNN_ACTIVATION_RELU, y, dy, x, dx)

dx1 = to_host(dy)
KUnet.relu(l1,y1,dx1)
@show isequal(dx1,to_host(dx))

x = CUDNN.Tensor(ones(5,4,3,2))
@show to_host(x)
@show to_host(CUDNN.cudnnScaleTensor(x, pi))

x = CUDNN.Tensor(rand(5,4,3,2))
@show to_host(x)
@show to_host(CUDNN.cudnnSetTensor(x, 7))

src = CUDNN.Tensor(ones(5,4,3,2))
bias = CUDNN.Tensor(ones(5,4,1,1))
@show to_host(CUDNN.cudnnAddTensor(CUDNN.CUDNN_ADD_IMAGE, 1, bias, 1, src))

tx = CUDNN.Tensor(ones(2,3,4,5))
ty = CUDNN.Tensor(ones(2,3,4,5))
CUDNN.cudnnTransformTensor(2.0, tx, 3.0, ty)
@show to_host(ty)

d = CUDNN.cudnnTensorDescriptor_t[0]
CUDNN.cudnnCreateTensorDescriptor(d)
CUDNN.cudnnSetTensor4dDescriptor(d[1], CUDNN.CUDNN_TENSOR_NCHW, CUDNN.CUDNN_DATA_FLOAT, 2,3,4,5)
an=Cint[0]
ac=Cint[0]
ah=Cint[0]
aw=Cint[0]
anS=Cint[0]
acS=Cint[0]
ahS=Cint[0]
awS=Cint[0]
dt=typeof(CUDNN.CUDNN_DATA_FLOAT)[0]
CUDNN.cudnnGetTensor4dDescriptor(d[1],dt,an,ac,ah,aw,anS,acS,ahS,awS)
@show dt[1]
@show an[1]
@show ac[1]
@show ah[1]
@show aw[1]
@show anS[1]
@show acS[1]
@show ahS[1]
@show awS[1]

nd=4
nbDims=Cint[0]
dimA=Array(Cint,nd)
strideA=Array(Cint,nd)
CUDNN.cudnnGetTensorNdDescriptor(d[1],nd,dt,nbDims,dimA,strideA)
@show nbDims[1]
@show dimA
@show strideA

a = rand(Float32, 5, 4, 3, 2)
t = CUDNN.Tensor(a)

CUDNN.cudnnGetTensorNdDescriptor(d[1],nd,dt,nbDims,dimA,strideA)
@show nbDims[1]
@show dimA
@show strideA

@show CUDNN.cudnnGetTensorNdDescriptor(t.desc)

end

:ok
