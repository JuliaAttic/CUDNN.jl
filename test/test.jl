using CUDArt
using CUDNN

x = CUDNN.Tensor(ones(5,4,3,2))
@show to_host(x)
@show to_host(CUDNN.cudnnScaleTensor(x, pi))

if false

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
