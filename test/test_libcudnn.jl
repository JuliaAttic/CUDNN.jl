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
for (x,a) in [ (n,nP), (c,cP), (h,hP), (w,wP), (nStride,nStrideP), (cStride,cStrideP), (hStride,hStrideP), (wStride,wStrideP)]
    @test x == a[1]
end

for nbDims = 3:8
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

for n=3:8
    xsize = rndsize(n)
    alpha = rand(Float32, 1)
    beta = rand(Float32, 1)
    X = rand(Float32, xsize); x = CudaArray(X); xDesc = tensorDescriptor(x)
    Y = rand(Float32, xsize); y = CudaArray(Y); yDesc = tensorDescriptor(y)
    @test cudnnTransformTensor(handle,alpha,xDesc,x,beta,yDesc,y) == nothing # 14/131
    @test isapprox(to_host(y), alpha[1]*X + beta[1]*Y)
end

# The documentation is misleading.  There are only 4 patterns cudnnAddTensor works for:
# nchw, 1chw, 11hw, 1c11

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

@show cudnnCreateOpTensorDescriptor(opTensorDesc) # 16/131
@show cudnnSetOpTensorDescriptor(opTensorDesc,opTensorOp,opTensorCompType,opTensorNanOpt) # 17/131
@show cudnnGetOpTensorDescriptor(opTensorDesc,opTensorOp,opTensorCompType,opTensorNanOpt) # 18/131
@show cudnnDestroyOpTensorDescriptor(opTensorDesc) # 19/131
@show cudnnOpTensor(handle,opTensorDesc,alpha1,aDesc,A,alpha2,bDesc,B,beta,cDesc,C) # 20/131

@show cudnnSetTensor(handle,yDesc,y,valuePtr) # 21/131
@show cudnnScaleTensor(handle,yDesc,y,alpha) # 22/131
@show cudnnCreateFilterDescriptor(filterDesc) # 23/131
@show cudnnSetFilter4dDescriptor(filterDesc,dataType,format,k,c,h,w) # 24/131
@show cudnnGetFilter4dDescriptor(filterDesc,dataType,format,k,c,h,w) # 25/131
@show cudnnSetFilterNdDescriptor(filterDesc,dataType,format,nbDims,filterDimA) # 26/131
@show cudnnGetFilterNdDescriptor(filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA) # 27/131
@show cudnnDestroyFilterDescriptor(filterDesc) # 28/131
@show cudnnCreateConvolutionDescriptor(convDesc) # 29/131
@show cudnnSetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode) # 30/131
@show cudnnSetConvolution2dDescriptor_v5(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode,dataType) # 31/131
@show cudnnGetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode) # 32/131
@show cudnnGetConvolution2dDescriptor_v5(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode,dataType) # 33/131
@show cudnnGetConvolution2dForwardOutputDim(convDesc,inputTensorDesc,filterDesc,n,c,h,w) # 34/131
@show cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType) # 35/131
@show cudnnGetConvolutionNdDescriptor(convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType) # 36/131
@show cudnnGetConvolutionNdForwardOutputDim(convDesc,inputTensorDesc,filterDesc,nbDims,tensorOuputDimA) # 37/131
@show cudnnDestroyConvolutionDescriptor(convDesc) # 38/131
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
