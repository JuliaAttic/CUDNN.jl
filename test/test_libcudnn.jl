using CUDNN,Base.Test,CUDArt

cudnnStatus_max = 10
cudnnDataType_max = 2
cudnnNanPropagation_max = 1
cudnnTensorFormat_max = 1
cudnnOpTensorOp_max = 3
cudnnConvolutionMode_max = 1
cudnnConvolutionFwdPreference_max = 2
cudnnConvolutionFwdAlgo_max = 6
cudnnConvolutionBwdFilterPreference_max = 2
cudnnConvolutionBwdFilterAlgo_max = 3
cudnnConvolutionBwdDataPreference_max = 2
cudnnConvolutionBwdDataAlgo_max = 4
cudnnSoftmaxAlgorithm_max = 2
cudnnSoftmaxMode_max = 1
cudnnPoolingMode_max = 2
cudnnActivationMode_max = 3
cudnnLRNMode_max = 0
cudnnDivNormMode_max = 0
cudnnBatchNormMode_max = 1
cudnnSamplerType_max = 0
cudnnRNNMode_max = 3
cudnnDirectionMode_max = 1
cudnnRNNInputMode_max = 1
CUDNN_DIM_MIN = 3
# CUDNN_DIM_MAX = 8
CUDNN_ADD_MIN,CUDNN_ADD_MAX = 4,5
CUDNN_OP_MIN,CUDNN_OP_MAX = 4,5

v = Cint[0]
CUDArt.rt.cudaDriverGetVersion(v); cudaDriverVersion=v[1]
CUDArt.rt.cudaRuntimeGetVersion(v); cudaRuntimeVersion=v[1]
cudnnVersion = Int(cudnnGetVersion())
@show (cudaDriverVersion, cudaRuntimeVersion, cudnnVersion)

@test Int(cudnnGetVersion()) == 5005 # 1/131

status = CUDNN_STATUS_SUCCESS
@test bytestring(cudnnGetErrorString(status)) == "CUDNN_STATUS_SUCCESS" # 2/131

handlePtr = cudnnHandle_t[0]
@test cudnnCreate(handlePtr) == nothing # 3/131
handle = handlePtr[1]
@test handle != cudnnHandle_t(0)      # cudnnDestroy 4/131 tested at the end

using CUDArt.rt: cudaStream_t, cudaStreamCreate
streamIdPtr = cudaStream_t[0]
@test cudaStreamCreate(streamIdPtr) == nothing
streamId = streamIdPtr[1]
@test streamId != cudaStream_t(0)
@test cudnnSetStream(handle,streamId) == nothing # 5/131

streamIdPtr[1] = 0
@test cudnnGetStream(handle,streamIdPtr) == nothing# 6/131
@test streamIdPtr[1] == streamId

tensorDescPtr = cudnnTensorDescriptor_t[0]
@test cudnnCreateTensorDescriptor(tensorDescPtr) == nothing # 7/131
tensorDesc = tensorDescPtr[1]
@test tensorDesc != cudnnTensorDescriptor_t(0)

rndsize(n) = ntuple(i->rand(1:6), n)
rstride(s) = ntuple(i->prod(s[i+1:end]), length(s))
format = CUDNN_TENSOR_NCHW
dataType = CUDNN_DATA_FLOAT
n,c,h,w = rndsize(4)
@test cudnnSetTensor4dDescriptor(tensorDesc,format,dataType,n,c,h,w) == nothing # 8/131

nStride,cStride,hStride,wStride = rstride((n,c,h,w))
@test cudnnSetTensor4dDescriptorEx(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride) == nothing # 9/131

dataTypeP = cudnnDataType_t[0]
nP,cP,hP,wP,nStrideP,cStrideP,hStrideP,wStrideP = [ Cint[0] for i in 1:8 ]
@test cudnnGetTensor4dDescriptor(tensorDesc,dataTypeP,nP,cP,hP,wP,nStrideP,cStrideP,hStrideP,wStrideP) == nothing # 10/131
for (x,a) in [ (dataType,dataTypeP), (n,nP), (c,cP), (h,hP), (w,wP), (nStride,nStrideP), (cStride,cStrideP), (hStride,hStrideP), (wStride,wStrideP)]
    @test x == a[1]
end

for nbDims in CUDNN_DIM_MIN:CUDNN_DIM_MAX
    dimA = Cint[rndsize(nbDims)...]
    strideA = Cint[rstride(dimA)...]
    @test cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA) == nothing # 11/131

    nbDimsRequested = nbDims
    nbDimsP = Cint[0]
    dimP = zeros(Cint,nbDims)
    strideP = zeros(Cint,nbDims)
    @test cudnnGetTensorNdDescriptor(tensorDesc,nbDimsRequested,dataTypeP,nbDimsP,dimP,strideP) == nothing # 12/131
    @test dataTypeP[1] == dataType
    @test nbDimsP[1] == nbDims
    @test dimP == dimA
    @test strideP == strideA
end

@test cudnnDestroyTensorDescriptor(tensorDesc) == nothing # 13/131

function tensorDescriptor{T}(x::CudaArray{T})
    tensorDescPtr = cudnnTensorDescriptor_t[0]
    cudnnCreateTensorDescriptor(tensorDescPtr)
    tensorDesc = tensorDescPtr[1]
    dataType = (T==Float64 ? CUDNN_DATA_DOUBLE :
                T==Float32 ? CUDNN_DATA_FLOAT :
                T==Float16 ? CUDNN_DATA_HALF :
                error("CUDNN does not support $T"))
    nbDims = Cint(ndims(x))
    dimA = Cint[reverse(size(x))...]
    strideA = Cint[reverse(strides(x))...]
    cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA)
    tensorDesc
end

alpha = rand(Float32, 1)
beta = rand(Float32, 1)
for n=CUDNN_DIM_MIN:CUDNN_DIM_MAX
    xsize = rndsize(n)
    X = rand(Float32, xsize); x = CudaArray(X); xDesc = tensorDescriptor(x)
    Y = rand(Float32, xsize); y = CudaArray(Y); yDesc = tensorDescriptor(y)
    @test cudnnTransformTensor(handle,alpha,xDesc,x,beta,yDesc,y) == nothing # 14/131
    @test isapprox(to_host(y), alpha[1]*X + beta[1]*Y)
end

warn("The cudnnAddTensor documentation is misleading.  There are only 4 patterns cudnnAddTensor works for: nchw, 1chw, 11hw, 1c11")

for n=CUDNN_ADD_MIN:CUDNN_ADD_MAX
    for m=1:4 # 0:(2^n-1)
        csize = rndsize(n)
        C = rand(Float32, csize); c = CudaArray(C); cDesc = tensorDescriptor(c)
        # asize = ntuple(i->((1<<(i-1))&m==0 ? csize[i] : 1), n)
        asize = (m==1 ? ntuple(i->(i==n-1) ? csize[i] : 1, n) :
                 m==2 ? ntuple(i->(i==n) ? 1 : csize[i], n) :
                 m==3 ? ntuple(i->(i>=n-1) ? 1 : csize[i], n) :
                 m==4 ? csize :
                 error())
        A = rand(Float32, asize); a = CudaArray(A); aDesc = tensorDescriptor(a)
        # @show (csize,asize) # ,handle,alpha,aDesc,a,beta,cDesc,c)
        try 
            cudnnAddTensor(handle,alpha,aDesc,a,beta,cDesc,c) == nothing # 15/131
        catch e
            warn("$csize $asize failed.")
        end
        @test isapprox(to_host(c), alpha[1]*A .+ beta[1]*C)
    end
end

opTensorDescP = cudnnOpTensorDescriptor_t[0]
@test cudnnCreateOpTensorDescriptor(opTensorDescP) == nothing # 16/131
opTensorDesc = opTensorDescP[1]
@test opTensorDesc != cudnnOpTensorDescriptor_t(0)

opTensorOpP = cudnnOpTensorOp_t[0]
opTensorCompTypeP = cudnnDataType_t[0]
opTensorNanOptP = cudnnNanPropagation_t[0]

# cudnnOpTensor supports full broadcasting (unlike cudnnAddTensor), only on 4D and 5D tensors.
# Supports double and float, could not get it to work for half.  The docs are not clear:
# The data types of the input tensors A and B must match. If the data type of the destination tensor C is double, then the data type of the input tensors also must be double.
# If the data type of the destination tensor C is double, then opTensorCompType in opTensorDesc must be double. Else opTensorCompType must be float.
# If the input tensor B is the same tensor as the destination tensor C, then the input tensor A also must be the same tensor as the destination tensor C.

for opTensorOp in (CUDNN_OP_TENSOR_ADD, CUDNN_OP_TENSOR_MUL, CUDNN_OP_TENSOR_MIN, CUDNN_OP_TENSOR_MAX)
    for opTensorCompType in (CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE) # not sure about CUDNN_DATA_HALF
        for opTensorNanOpt in (CUDNN_NOT_PROPAGATE_NAN, CUDNN_PROPAGATE_NAN)
            @test cudnnSetOpTensorDescriptor(opTensorDesc,opTensorOp,opTensorCompType,opTensorNanOpt) == nothing # 17/131
            @test cudnnGetOpTensorDescriptor(opTensorDesc,opTensorOpP,opTensorCompTypeP,opTensorNanOptP) == nothing # 18/131
            @test opTensorOpP[1]==opTensorOp && opTensorCompTypeP[1]==opTensorCompType && opTensorNanOptP[1]==opTensorNanOpt
            # C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C
            # size(A)==size(C), size(B,i)==size(C,i) or 1
            T = CUDNN.juliaDataType(opTensorCompType)
            alpha1, alpha2, beta = [ rand(T,1) for i=1:3 ]
            for n in CUDNN_OP_MIN:CUDNN_OP_MAX        # Only ndims 4:5 supported
                for m in 0:(2^n-1)
                    asize = csize = rndsize(n)
                    bsize = ntuple(i->((1<<(i-1))&m==0 ? 1 : csize[i]), n)
                    A = rand(T, asize); a = CudaArray(A); aDesc = tensorDescriptor(a)
                    B = rand(T, bsize); b = CudaArray(B); bDesc = tensorDescriptor(b)
                    C = rand(T, csize); c = CudaArray(C); cDesc = tensorDescriptor(c)
                    @test cudnnOpTensor(handle,opTensorDesc,alpha1,aDesc,a,alpha2,bDesc,b,beta,cDesc,c) == nothing # 20/131
                    op = (opTensorOp == CUDNN_OP_TENSOR_ADD ? (.+) :
                          opTensorOp == CUDNN_OP_TENSOR_MUL ? (.*) :
                          opTensorOp == CUDNN_OP_TENSOR_MIN ? (x,y)->broadcast(min,x,y) :
                          opTensorOp == CUDNN_OP_TENSOR_MAX ? (x,y)->broadcast(max,x,y) : error())
                    @test isapprox(to_host(c), op(alpha1[1]*A, alpha2[1]*B) + beta[1]*C)
                end
            end
        end
    end
end
@test cudnnDestroyOpTensorDescriptor(opTensorDesc) == nothing # 19/131

for n=CUDNN_OP_MIN:CUDNN_OP_MAX
    y = CudaArray(Float32, rndsize(n)); yDesc = tensorDescriptor(y)
    valuePtr = Float32[42]
    @test cudnnSetTensor(handle,yDesc,y,valuePtr) == nothing # 21/131
    @test all(to_host(y) .== 42f0)
    @test cudnnScaleTensor(handle,yDesc,y,alpha) == nothing # 22/131
    @test all(to_host(y) .== 42*alpha[1])
end

filterDescPtr = cudnnFilterDescriptor_t[0]
@test cudnnCreateFilterDescriptor(filterDescPtr) == nothing # 23/131
filterDesc = filterDescPtr[1]
@test filterDesc != cudnnFilterDescriptor_t(0)

format = CUDNN_TENSOR_NCHW
dataType = CUDNN_DATA_FLOAT
k,c,h,w = rndsize(4)
@test cudnnSetFilter4dDescriptor(filterDesc,dataType,format,k,c,h,w) == nothing # 24/131

dataTypeP = cudnnDataType_t[0]
formatP = cudnnTensorFormat_t[0]
kP,cP,hP,wP = [ Cint[0] for i in 1:4 ]
@test cudnnGetFilter4dDescriptor(filterDesc,dataTypeP,formatP,kP,cP,hP,wP) == nothing # 25/131
for (x,a) in [ (dataType,dataTypeP), (format,formatP), (k,kP), (c,cP), (h,hP), (w,wP) ]
    @test x == a[1]
end

for nbDims in CUDNN_DIM_MIN:CUDNN_DIM_MAX
    dimA = Cint[rndsize(nbDims)...]
    @test cudnnSetFilterNdDescriptor(filterDesc,dataType,format,nbDims,dimA) == nothing # 26/131

    nbDimsRequested = nbDims
    nbDimsP = Cint[0]
    dimP = zeros(Cint,nbDims)
    @test cudnnGetFilterNdDescriptor(filterDesc,nbDimsRequested,dataTypeP,formatP,nbDimsP,dimP) == nothing # 27/131
    @test dataTypeP[1] == dataType
    @test formatP[1] == format
    @test nbDimsP[1] == nbDims
    @test dimP == dimA
end

@test cudnnDestroyFilterDescriptor(filterDesc) == nothing # 28/131

function filterDescriptor{T}(x::CudaArray{T})
    filterDescPtr = cudnnFilterDescriptor_t[0]
    cudnnCreateFilterDescriptor(filterDescPtr)
    filterDesc = filterDescPtr[1]
    dataType = (T==Float64 ? CUDNN_DATA_DOUBLE :
                T==Float32 ? CUDNN_DATA_FLOAT :
                T==Float16 ? CUDNN_DATA_HALF :
                error("CUDNN does not support $T"))
    format = CUDNN_TENSOR_NCHW
    nbDims = Cint(ndims(x))
    dimA = Cint[reverse(size(x))...]
    cudnnSetFilterNdDescriptor(filterDesc,dataType,format,nbDims,dimA)
    filterDesc
end

convDescPtr = cudnnConvolutionDescriptor_t[0]
@test cudnnCreateConvolutionDescriptor(convDescPtr) == nothing # 29/131
convDesc = convDescPtr[1]
@test convDesc != cudnnConvolutionDescriptor_t(0)

pad_h,pad_w,u,v = [ rand(1:5) for i=1:4 ]
upscalex = upscaley = 1
mode = CUDNN_CONVOLUTION
@test cudnnSetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode) == nothing # 30/131
@test cudnnSetConvolution2dDescriptor_v5(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode,dataType) == nothing # 31/131
pad_hP,pad_wP,uP,vP,upscalexP,upscaleyP = [ Cint[0] for i=1:6 ]
modeP = cudnnConvolutionMode_t[0]
@test cudnnGetConvolution2dDescriptor(convDesc,pad_hP,pad_wP,uP,vP,upscalexP,upscaleyP,modeP) == nothing # 32/131
for (a,x) in [(pad_hP,pad_h),(pad_wP,pad_w),(uP,u),(vP,v),(upscalexP,upscalex),(upscaleyP,upscaley),(modeP,mode)]
    @test a[1] == x
end
@test cudnnGetConvolution2dDescriptor_v5(convDesc,pad_hP,pad_wP,uP,vP,upscalexP,upscaleyP,modeP,dataTypeP) == nothing # 33/131
for (a,x) in [(pad_hP,pad_h),(pad_wP,pad_w),(uP,u),(vP,v),(upscalexP,upscalex),(upscaleyP,upscaley),(modeP,mode),(dataTypeP,dataType)]
    @test a[1] == x
end

t = CudaArray(Float32,rndsize(4)); tDesc = tensorDescriptor(t)
# first two filter dims need to be smaller than or equal to tensor dims + 2*pad
fsize = ntuple(i->(i==1 ? rand(1:2*pad_w+size(t,i)) : i==2 ? rand(1:2*pad_h+size(t,i)) : i==3 ? size(t,i) : rand(1:5)), 4)
f = CudaArray(Float32,fsize); fDesc = filterDescriptor(f)
@test cudnnGetConvolution2dForwardOutputDim(convDesc,tDesc,fDesc,nP,cP,hP,wP) == nothing # 34/131
# @show (pad_w,pad_h,v,u), size(t), size(f), (wP[1],hP[1],cP[1],nP[1])
# outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride
@test nP[1] == size(t,4)
@test cP[1] == size(f,4)
@test hP[1] == 1 + div(size(t,2) + 2 * pad_h - size(f,2), u)
@test wP[1] == 1 + div(size(t,1) + 2 * pad_w - size(f,1), v)

using CUDNN: juliaDataType

for arrayLength = 1:(CUDNN_DIM_MAX-2)
    padA = Cint[rndsize(arrayLength)...]
    strideA = Cint[rndsize(arrayLength)...]
    upscaleA = ones(Cint, arrayLength)
    mode = rand()<1/2 ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION
    dataType = rand()<1/3 ? CUDNN_DATA_DOUBLE : rand()<1/2 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF
    @test cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,strideA,upscaleA,mode,dataType) == nothing # 35/131 
    padP, strideP, upscaleP = map(similar, (padA, strideA, upscaleA))
    arrayLengthRequested = arrayLength
    arrayLengthP = Cint[0]
    @test cudnnGetConvolutionNdDescriptor(convDesc,arrayLengthRequested,arrayLengthP,padP,strideP,upscaleP,modeP,dataTypeP) == nothing # 36/131
    for (a,p) in [ (padA,padP), (strideA,strideP), (upscaleA,upscaleP) ]; @test a==p; end
    for (x,p) in [ (arrayLength,arrayLengthP), (mode,modeP), (dataType,dataTypeP) ]; @test x==p[1]; end

    xsize = rndsize(arrayLength+2)
    wsize = ntuple(arrayLength+2) do i
        (i == 1 ? rand(1:5) :
         i == 2 ? xsize[i] :
         rand(1:xsize[i]+2*padA[i-2]))
    end
    x = CudaArray(juliaDataType(dataType),reverse(xsize))
    w = CudaArray(juliaDataType(dataType),reverse(wsize))
    xDesc = tensorDescriptor(x)
    wDesc = filterDescriptor(w)
    nbDims = arrayLength+2
    dimA = Array(Cint,nbDims)
    # @show xsize, wsize, padA, strideA

    @test cudnnGetConvolutionNdForwardOutputDim(convDesc,xDesc,wDesc,nbDims,dimA) == nothing # 37/131
    # 38/131 is after the for loop
    # @show dimA
    @test dimA[1] == xsize[1]
    @test dimA[2] == wsize[1]
    for i=1:arrayLength; @test dimA[i+2]==1+div(xsize[i+2]+2*padA[i]-wsize[i+2],strideA[i]); end

    ysize = dimA
    y = CudaArray(juliaDataType(dataType),reverse(ysize))
    yDesc = tensorDescriptor(y)
    requestedAlgoCount = 1 + cudnnConvolutionFwdAlgo_max
    returnedAlgoCount = Cint[0]
    perfResults = Array(cudnnConvolutionFwdAlgoPerf_t, requestedAlgoCount)
    @test cudnnFindConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,requestedAlgoCount,returnedAlgoCount,perfResults) == nothing # 39/131
    for i=1:returnedAlgoCount[1]
        println((:cudnnFindConvolutionForwardAlgorithm, i, perfResults[i]))
    end
    
    workSpaceSizeInBytes = 1000000
    workSpace = CudaArray(Cuchar, workSpaceSizeInBytes)
    @test cudnnFindConvolutionForwardAlgorithmEx(handle,xDesc,x,wDesc,w,convDesc,yDesc,y,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes) == nothing # 40/131
    for i=1:returnedAlgoCount[1]
        println((:cudnnFindConvolutionForwardAlgorithmEx, i, perfResults[i]))
    end

    for i=0:cudnnConvolutionFwdPreference_max
        preference = cudnnConvolutionFwdPreference_t(i)
        memoryLimitInBytes = Csize_t(10^10)
        algoP = cudnnConvolutionFwdAlgo_t[0]
        @test cudnnGetConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,preference,memoryLimitInBytes,algoP) == nothing # 41/131
        println((:cudnnGetConvolutionForwardAlgorithm, preference, algoP[1]))
    end
    for i=0:cudnnConvolutionFwdAlgo_max
        algo = cudnnConvolutionFwdAlgo_t(i)
        sizeInBytes = Csize_t[0]
        @test cudnnGetConvolutionForwardWorkspaceSize(handle,xDesc,wDesc,convDesc,yDesc,algo,sizeInBytes) == nothing # 42/131
        println((:cudnnGetConvolutionForwardWorkspaceSize, algo, sizeInBytes[1]))
        workSpaceSizeInBytes = sizeInBytes[1]
        workSpace = CudaArray(UInt8, workSpaceSizeInBytes)
        @test cudnnConvolutionForward(handle,alpha,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,yDesc,y) == nothing # 43/131
    end

    # cudnnConvolutionBackwardBias has nothing to do with convolution, it is just broadcast addition.
    # The Julia size of the bias array should be (1,1,...,C,1)
    dy = randn!(similar(y))
    dyDesc = tensorDescriptor(dy)
    bsize = ntuple(i->(i==2 ? ysize[i] : 1), length(ysize))
    db = similar(y, reverse(bsize))
    dbDesc = tensorDescriptor(db)
    @test cudnnConvolutionBackwardBias(handle,alpha,dyDesc,dy,beta,dbDesc,db) == nothing # 44/131

    dw = similar(w)
    dwDesc = tensorDescriptor(dw)
    requestedAlgoCount = Cint(1 + cudnnConvolutionBwdFilterAlgo_max)
    perfResults = Array(cudnnConvolutionBwdFilterAlgoPerf_t, requestedAlgoCount)
    @test cudnnFindConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,requestedAlgoCount,returnedAlgoCount,perfResults) == nothing # 45/131
    for i=1:returnedAlgoCount[1]
        println((:cudnnFindConvolutionBackwardFilterAlgorithm, i, perfResults[i]))
    end

    workSpaceSizeInBytes = 1000000
    workSpace = CudaArray(Cuchar, workSpaceSizeInBytes)
    @test cudnnFindConvolutionBackwardFilterAlgorithmEx(handle,xDesc,x,dyDesc,y,convDesc,dwDesc,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes) == nothing # 46/131
    for i=1:returnedAlgoCount[1]
        println((:cudnnFindConvolutionBackwardFilterAlgorithmEx, i, perfResults[i]))
    end

    for i=0:cudnnConvolutionBwdFilterPreference_max
        preference = cudnnConvolutionBwdFilterPreference_t(i)
        memoryLimitInBytes = Csize_t(10^10)
        algoP = cudnnConvolutionBwdFilterAlgo_t[0]
        @test cudnnGetConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,preference,memoryLimitInBytes,algoP) == nothing # 47/131
        println((:cudnnGetConvolutionBackwardFilterAlgorithm, preference, algoP[1]))
    end

    for i=0:cudnnConvolutionBwdFilterAlgo_max
        algo = cudnnConvolutionBwdFilterAlgo_t(i)
        sizeInBytes = Csize_t[0]
        @test cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,xDesc,dyDesc,convDesc,dwDesc,algo,sizeInBytes) == nothing # 48/131
        println((:cudnnGetConvolutionBackwardFilterWorkspaceSize, algo, sizeInBytes[1]))
        workSpaceSizeInBytes = sizeInBytes[1]
        workSpace = CudaArray(UInt8, workSpaceSizeInBytes)
        @test cudnnConvolutionBackwardFilter(handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw) == nothing # 49/131
    end

    dx = similar(x)
    dxDesc = tensorDescriptor(dx)
    requestedAlgoCount = Cint(1 + cudnnConvolutionBwdDataAlgo_max)
    perfResults = Array(cudnnConvolutionBwdDataAlgoPerf_t, requestedAlgoCount)
    @test cudnnFindConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,requestedAlgoCount,returnedAlgoCount,perfResults) == nothing # 50/131
    for i=1:returnedAlgoCount[1]
        println((:cudnnFindConvolutionBackwardDataAlgorithm, i, perfResults[i]))
    end

    workSpaceSizeInBytes = 1000000
    workSpace = CudaArray(Cuchar, workSpaceSizeInBytes)
    @test cudnnFindConvolutionBackwardDataAlgorithmEx(handle,wDesc,w,dyDesc,dy,convDesc,dxDesc,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes) == nothing # 51/131
    for i=1:returnedAlgoCount[1]
        println((:cudnnFindConvolutionBackwardDataAlgorithmEx, i, perfResults[i]))
    end

    for i=0:cudnnConvolutionBwdDataPreference_max
        preference = cudnnConvolutionBwdDataPreference_t(i)
        memoryLimitInBytes = Csize_t(10^10)
        algoP = cudnnConvolutionBwdDataAlgo_t[0]
        @test cudnnGetConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,preference,memoryLimitInBytes,algoP) == nothing # 52/131
        println((:cudnnGetConvolutionBackwardDataAlgorithm, preference, algoP[1]))
    end

    for i=0:cudnnConvolutionBwdDataAlgo_max
        algo = cudnnConvolutionBwdDataAlgo_t(i)
        sizeInBytes = Csize_t[0]
        @test cudnnGetConvolutionBackwardDataWorkspaceSize(handle,wDesc,dyDesc,convDesc,dxDesc,algo,sizeInBytes) == nothing # 53/131
        println((:cudnnGetConvolutionBackwardDataWorkspaceSize, algo, sizeInBytes[1]))
        workSpaceSizeInBytes = sizeInBytes[1]
        workSpace = CudaArray(UInt8, workSpaceSizeInBytes)
        @test cudnnConvolutionBackwardData(handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx) == nothing # 54/131
    end

    warn("cudnnIm2Col is not in the v4,v5 documentation?")
    colBuffer = CudaArray(CuChar, sizeof(x))
    @test cudnnIm2Col(handle,xDesc,x,wDesc,convDesc,colBuffer) == nothing # 55/131
end

@test cudnnDestroyConvolutionDescriptor(convDesc) == nothing # 38/131

for i=CUDNN_DIM_MIN:CUDNN_DIM_MAX
    xsize = rndsize(i)
    x,y,dy,dx = [ CudaArray(juliaDataType(dataType),reverse(xsize)) for j=1:4 ]
    xDesc = yDesc = dyDesc = dxDesc = tensorDescriptor(x)
    for algo=0:cudnnSoftmaxAlgorithm_max
        for mode=0:cudnnSoftmaxMode_max
            @test cudnnSoftmaxForward(handle,algo,mode,alpha,xDesc,x,beta,yDesc,y) == nothing # 56/131
            @test cudnnSoftmaxBackward(handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx) == nothing # 57/131
        end
    end
end

error(:ok)

@show cudnnCreatePoolingDescriptor(poolingDesc) # 58/131
@show cudnnSetPooling2dDescriptor(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride) # 59/131
@show cudnnGetPooling2dDescriptor(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride) # 60/131
@show cudnnSetPoolingNdDescriptor(poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA) # 61/131
@show cudnnGetPoolingNdDescriptor(poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA) # 62/131
@show cudnnGetPoolingNdForwardOutputDim(poolingDesc,inputTensorDesc,nbDims,outputTensorDimA) # 63/131
@show cudnnGetPooling2dForwardOutputDim(poolingDesc,inputTensorDesc,n,c,h,w) # 64/131
@show cudnnDestroyPoolingDescriptor(poolingDesc) # 65/131

@show cudnnPoolingForward(handle,poolingDesc,alpha,xDesc,x,beta,yDesc,y) # 66/131
@show cudnnPoolingBackward(handle,poolingDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx) # 67/131

@show cudnnCreateActivationDescriptor(activationDesc) # 68/131
@show cudnnSetActivationDescriptor(activationDesc,mode,reluNanOpt,reluCeiling) # 69/131
@show cudnnGetActivationDescriptor(activationDesc,mode,reluNanOpt,reluCeiling) # 70/131
@show cudnnDestroyActivationDescriptor(activationDesc) # 71/131

@show cudnnActivationForward(handle,activationDesc,alpha,xDesc,x,beta,yDesc,y) # 72/131
@show cudnnActivationBackward(handle,activationDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx) # 73/131

@show cudnnCreateLRNDescriptor(normDesc) # 74/131
@show cudnnSetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK) # 75/131
@show cudnnGetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK) # 76/131
@show cudnnDestroyLRNDescriptor(lrnDesc) # 77/131

@show cudnnLRNCrossChannelForward(handle,normDesc,lrnMode,alpha,xDesc,x,beta,yDesc,y) # 78/131
@show cudnnLRNCrossChannelBackward(handle,normDesc,lrnMode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx) # 79/131

@show cudnnDivisiveNormalizationForward(handle,normDesc,mode,alpha,xDesc,x,means,temp,temp2,beta,yDesc,y) # 80/131
@show cudnnDivisiveNormalizationBackward(handle,normDesc,mode,alpha,xDesc,x,means,dy,temp,temp2,beta,dXdMeansDesc,dx,dMeans) # 81/131

@show cudnnDeriveBNTensorDescriptor(derivedBnDesc,xDesc,mode) # 82/131
@show cudnnBatchNormalizationForwardTraining(handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,resultRunningMean,resultRunningVariance,epsilon,resultSaveMean,resultSaveInvVariance) # 83/131
@show cudnnBatchNormalizationForwardInference(handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,estimatedMean,estimatedVariance,epsilon) # 84/131
@show cudnnBatchNormalizationBackward(handle,mode,alphaDataDiff,betaDataDiff,alphaParamDiff,betaParamDiff,xDesc,x,dyDesc,dy,dxDesc,dx,dBnScaleBiasDesc,bnScale,dBnScaleResult,dBnBiasResult,epsilon,savedMean,savedInvVariance) # 85/131

@show cudnnCreateSpatialTransformerDescriptor(stDesc) # 86/131
@show cudnnSetSpatialTransformerNdDescriptor(stDesc,samplerType,dataType,nbDims,dimA) # 87/131
@show cudnnDestroySpatialTransformerDescriptor(stDesc) # 88/131

@show cudnnSpatialTfGridGeneratorForward(handle,stDesc,theta,grid) # 89/131
@show cudnnSpatialTfGridGeneratorBackward(handle,stDesc,dgrid,dtheta) # 90/131

@show cudnnSpatialTfSamplerForward(handle,stDesc,alpha,xDesc,x,grid,beta,yDesc,y) # 91/131
@show cudnnSpatialTfSamplerBackward(handle,stDesc,alpha,xDesc,x,beta,dxDesc,dx,alphaDgrid,dyDesc,dy,grid,betaDgrid,dgrid) # 92/131

@show cudnnCreateDropoutDescriptor(dropoutDesc) # 93/131
@show cudnnDestroyDropoutDescriptor(dropoutDesc) # 94/131
@show cudnnDropoutGetStatesSize(handle,sizeInBytes) # 95/131
@show cudnnDropoutGetReserveSpaceSize(xdesc,sizeInBytes) # 96/131
@show cudnnSetDropoutDescriptor(dropoutDesc,handle,dropout,states,stateSizeInBytes,seed) # 97/131

@show cudnnDropoutForward(handle,dropoutDesc,xdesc,x,ydesc,y,reserveSpace,reserveSpaceSizeInBytes) # 98/131
@show cudnnDropoutBackward(handle,dropoutDesc,dydesc,dy,dxdesc,dx,reserveSpace,reserveSpaceSizeInBytes) # 99/131

@show cudnnCreateRNNDescriptor(rnnDesc) # 100/131
@show cudnnDestroyRNNDescriptor(rnnDesc) # 101/131
@show cudnnSetRNNDescriptor(rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,dataType) # 102/131
@show cudnnGetRNNWorkspaceSize(handle,rnnDesc,seqLength,xDesc,sizeInBytes) # 103/131
@show cudnnGetRNNTrainingReserveSize(handle,rnnDesc,seqLength,xDesc,sizeInBytes) # 104/131
@show cudnnGetRNNParamsSize(handle,rnnDesc,xDesc,sizeInBytes,dataType) # 105/131
@show cudnnGetRNNLinLayerMatrixParams(handle,rnnDesc,layer,xDesc,wDesc,w,linLayerID,linLayerMatDesc,linLayerMat) # 106/131
@show cudnnGetRNNLinLayerBiasParams(handle,rnnDesc,layer,xDesc,wDesc,w,linLayerID,linLayerBiasDesc,linLayerBias) # 107/131

@show cudnnRNNForwardInference(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,workspace,workSpaceSizeInBytes) # 108/131
@show cudnnRNNForwardTraining(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes) # 109/131
@show cudnnRNNBackwardData(handle,rnnDesc,seqLength,yDesc,y,dyDesc,dy,dhyDesc,dhy,dcyDesc,dcy,wDesc,w,hxDesc,hx,cxDesc,cx,dxDesc,dx,dhxDesc,dhx,dcxDesc,dcx,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes) # 110/131
@show cudnnRNNBackwardWeights(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,yDesc,y,workspace,workSpaceSizeInBytes,dwDesc,dw,reserveSpace,reserveSpaceSizeInBytes) # 111/131

@show cudnnSetFilter4dDescriptor_v3(filterDesc,dataType,k,c,h,w) # 112/131
@show cudnnSetFilter4dDescriptor_v4(filterDesc,dataType,format,k,c,h,w) # 113/131
@show cudnnGetFilter4dDescriptor_v3(filterDesc,dataType,k,c,h,w) # 114/131
@show cudnnGetFilter4dDescriptor_v4(filterDesc,dataType,format,k,c,h,w) # 115/131
@show cudnnSetFilterNdDescriptor_v3(filterDesc,dataType,nbDims,filterDimA) # 116/131
@show cudnnSetFilterNdDescriptor_v4(filterDesc,dataType,format,nbDims,filterDimA) # 117/131
@show cudnnGetFilterNdDescriptor_v3(filterDesc,nbDimsRequested,dataType,nbDims,filterDimA) # 118/131
@show cudnnGetFilterNdDescriptor_v4(filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA) # 119/131
@show cudnnSetPooling2dDescriptor_v3(poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride) # 120/131
@show cudnnSetPooling2dDescriptor_v4(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride) # 121/131
@show cudnnGetPooling2dDescriptor_v3(poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride) # 122/131
@show cudnnGetPooling2dDescriptor_v4(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride) # 123/131
@show cudnnSetPoolingNdDescriptor_v3(poolingDesc,mode,nbDims,windowDimA,paddingA,strideA) # 124/131
@show cudnnSetPoolingNdDescriptor_v4(poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA) # 125/131
@show cudnnGetPoolingNdDescriptor_v3(poolingDesc,nbDimsRequested,mode,nbDims,windowDimA,paddingA,strideA) # 126/131
@show cudnnGetPoolingNdDescriptor_v4(poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA) # 127/131
@show cudnnActivationForward_v3(handle,mode,alpha,xDesc,x,beta,yDesc,y) # 128/131
@show cudnnActivationForward_v4(handle,activationDesc,alpha,xDesc,x,beta,yDesc,y) # 129/131
@show cudnnActivationBackward_v3(handle,mode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx) # 130/131
@show cudnnActivationBackward_v4(handle,activationDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx) # 131/131

@test cudnnDestroy(handle) == nothing # 4/131
