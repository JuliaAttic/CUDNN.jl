using CUDNN,Base.Test,CUDArt

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

for nbDims in 3:8
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
for n=3:8
    xsize = rndsize(n)
    X = rand(Float32, xsize); x = CudaArray(X); xDesc = tensorDescriptor(x)
    Y = rand(Float32, xsize); y = CudaArray(Y); yDesc = tensorDescriptor(y)
    @test cudnnTransformTensor(handle,alpha,xDesc,x,beta,yDesc,y) == nothing # 14/131
    @test isapprox(to_host(y), alpha[1]*X + beta[1]*Y)
end

# The cudnnAddTensor documentation is misleading.  There are only 4
# patterns cudnnAddTensor works for: nchw, 1chw, 11hw, 1c11

for n=4:5
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
            for n in 4:5        # Only ndims 4:5 supported
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

for n=4:5
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

for nbDims in 3:8
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

for arrayLength = 1:6
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

    tsize = rndsize(arrayLength+2)
    fsize = ntuple(arrayLength+2) do i
        (i == 1 ? rand(1:5) :
         i == 2 ? tsize[i] :
         rand(1:tsize[i]+2*padA[i-2]))
    end
    tDesc = tensorDescriptor(CudaArray(juliaDataType(dataType),reverse(tsize)))
    fDesc = filterDescriptor(CudaArray(juliaDataType(dataType),reverse(fsize)))
    nbDims = arrayLength+2
    dimA = Array(Cint,nbDims)
    # @show tsize, fsize, padA, strideA
    @test cudnnGetConvolutionNdForwardOutputDim(convDesc,tDesc,fDesc,nbDims,dimA) == nothing # 37/131
    # @show dimA
    @test dimA[1] == tsize[1]
    @test dimA[2] == fsize[1]
    for i=1:arrayLength; @test dimA[i+2]==1+div(tsize[i+2]+2*padA[i]-fsize[i+2],strideA[i]); end
end

@test cudnnDestroyConvolutionDescriptor(convDesc) == nothing # 38/131

error(:ok)

@show cudnnFindConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,requestedAlgoCount,returnedAlgoCount,perfResults) # 39/131
@show cudnnFindConvolutionForwardAlgorithmEx(handle,xDesc,x,wDesc,w,convDesc,yDesc,y,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes) # 40/131
@show cudnnGetConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,preference,memoryLimitInBytes,algo) # 41/131
@show cudnnGetConvolutionForwardWorkspaceSize(handle,xDesc,wDesc,convDesc,yDesc,algo,sizeInBytes) # 42/131

@show cudnnConvolutionForward(handle,alpha,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,yDesc,y) # 43/131

@show cudnnConvolutionBackwardBias(handle,alpha,dyDesc,dy,beta,dbDesc,db) # 44/131

@show cudnnFindConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,requestedAlgoCount,returnedAlgoCount,perfResults) # 45/131
@show cudnnFindConvolutionBackwardFilterAlgorithmEx(handle,xDesc,x,dyDesc,y,convDesc,dwDesc,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes) # 46/131
@show cudnnGetConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,preference,memoryLimitInBytes,algo) # 47/131
@show cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,xDesc,dyDesc,convDesc,gradDesc,algo,sizeInBytes) # 48/131
@show cudnnConvolutionBackwardFilter(handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw) # 49/131

@show cudnnFindConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,requestedAlgoCount,returnedAlgoCount,perfResults) # 50/131
@show cudnnFindConvolutionBackwardDataAlgorithmEx(handle,wDesc,w,dyDesc,dy,convDesc,dxDesc,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes) # 51/131
@show cudnnGetConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,preference,memoryLimitInBytes,algo) # 52/131
@show cudnnGetConvolutionBackwardDataWorkspaceSize(handle,wDesc,dyDesc,convDesc,dxDesc,algo,sizeInBytes) # 53/131
@show cudnnConvolutionBackwardData(handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx) # 54/131

@show cudnnIm2Col(handle,xDesc,x,wDesc,convDesc,colBuffer) # 55/131

@show cudnnSoftmaxForward(handle,algo,mode,alpha,xDesc,x,beta,yDesc,y) # 56/131
@show cudnnSoftmaxBackward(handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx) # 57/131

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
