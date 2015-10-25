# Julia wrapper for header: /usr/usc/cuDNN/7.0.58/include/cudnn.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cudnnGetVersion()
    ccall((:cudnnGetVersion,libcudnn),Csize_t,())
end

function cudnnGetErrorString(status)
    bytestring(ccall((:cudnnGetErrorString,libcudnn),Ptr{UInt8},(cudnnStatus_t,),status))
end

function cudnnCreate(handle)
    cudnnCheck(ccall((:cudnnCreate,libcudnn),cudnnStatus_t,(Ptr{cudnnHandle_t},),handle))
end

function cudnnDestroy(handle)
    cudnnCheck(ccall((:cudnnDestroy,libcudnn),cudnnStatus_t,(cudnnHandle_t,),handle))
end

function cudnnSetStream(handle,streamId)
    cudnnCheck(ccall((:cudnnSetStream,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudaStream_t),handle,streamId))
end

function cudnnGetStream(handle,streamId)
    cudnnCheck(ccall((:cudnnGetStream,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{cudaStream_t}),handle,streamId))
end

function cudnnCreateTensorDescriptor(tensorDesc)
    cudnnCheck(ccall((:cudnnCreateTensorDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnTensorDescriptor_t},),tensorDesc))
end

function cudnnSetTensor4dDescriptor(tensorDesc,format,dataType,n,c,h,w)
    cudnnCheck(ccall((:cudnnSetTensor4dDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnTensorFormat_t,cudnnDataType_t,Cint,Cint,Cint,Cint),tensorDesc,format,dataType,n,c,h,w))
end

function cudnnSetTensor4dDescriptorEx(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
    cudnnCheck(ccall((:cudnnSetTensor4dDescriptorEx,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride))
end

function cudnnGetTensor4dDescriptor(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
    cudnnCheck(ccall((:cudnnGetTensor4dDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride))
end

function cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA)
    cudnnCheck(ccall((:cudnnSetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,nbDims,dimA,strideA))
end

function cudnnGetTensorNdDescriptor(tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA)
    cudnnCheck(ccall((:cudnnGetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA))
end

function cudnnDestroyTensorDescriptor(tensorDesc)
    cudnnCheck(ccall((:cudnnDestroyTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,),tensorDesc))
end

function cudnnTransformTensor(handle,alpha,srcDesc,srcData,beta,destDesc,destData)
    cudnnCheck(ccall((:cudnnTransformTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,beta,destDesc,destData))
end

if CUDNN_VERSION >= 3000

    function cudnnAddTensor(handle,alpha,biasDesc,biasData,beta,srcDestDesc,srcDestData)
        cudnnCheck(ccall((:cudnnAddTensor_v3,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,biasDesc,biasData,beta,srcDestDesc,srcDestData))
    end

else # if CUDNN_VERSION >= 3000

    function cudnnAddTensor(handle,mode,alpha,biasDesc,biasData,beta,srcDestDesc,srcDestData)
        cudnnCheck(ccall((:cudnnAddTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnAddMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,biasDesc,biasData,beta,srcDestDesc,srcDestData))
    end

end  # if CUDNN_VERSION >= 3000

function cudnnSetTensor(handle,srcDestDesc,srcDestData,value)
    cudnnCheck(ccall((:cudnnSetTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,srcDestDesc,srcDestData,value))
end

function cudnnScaleTensor(handle,srcDestDesc,srcDestData,alpha)
    cudnnCheck(ccall((:cudnnScaleTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,srcDestDesc,srcDestData,alpha))
end

function cudnnCreateFilterDescriptor(filterDesc)
    cudnnCheck(ccall((:cudnnCreateFilterDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnFilterDescriptor_t},),filterDesc))
end

function cudnnSetFilter4dDescriptor(filterDesc,dataType,k,c,h,w)
    cudnnCheck(ccall((:cudnnSetFilter4dDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,Cint,Cint,Cint,Cint),filterDesc,dataType,k,c,h,w))
end

function cudnnGetFilter4dDescriptor(filterDesc,dataType,k,c,h,w)
    cudnnCheck(ccall((:cudnnGetFilter4dDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),filterDesc,dataType,k,c,h,w))
end

function cudnnSetFilterNdDescriptor(filterDesc,dataType,nbDims,filterDimA)
    cudnnCheck(ccall((:cudnnSetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint}),filterDesc,dataType,nbDims,filterDimA))
end

function cudnnGetFilterNdDescriptor(filterDesc,nbDimsRequested,dataType,nbDims,filterDimA)
    cudnnCheck(ccall((:cudnnGetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,nbDims,filterDimA))
end

function cudnnDestroyFilterDescriptor(filterDesc)
    cudnnCheck(ccall((:cudnnDestroyFilterDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,),filterDesc))
end

function cudnnCreateConvolutionDescriptor(convDesc)
    cudnnCheck(ccall((:cudnnCreateConvolutionDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnConvolutionDescriptor_t},),convDesc))
end

function cudnnSetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
    cudnnCheck(ccall((:cudnnSetConvolution2dDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint,cudnnConvolutionMode_t),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode))
end

function cudnnGetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
    cudnnCheck(ccall((:cudnnGetConvolution2dDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t}),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode))
end

function cudnnGetConvolution2dForwardOutputDim(convDesc,inputTensorDesc,filterDesc,n,c,h,w)
    cudnnCheck(ccall((:cudnnGetConvolution2dForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,n,c,h,w))
end

if CUDNN_VERSION >= 3000

    function cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType)
        cudnnCheck(ccall((:cudnnSetConvolutionNdDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},cudnnConvolutionMode_t,cudnnDataType_t),convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType))
    end

    function cudnnGetConvolutionNdDescriptor(convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType)
        cudnnCheck(ccall((:cudnnGetConvolutionNdDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t},Ptr{cudnnDataType_t}),convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType))
    end

else # if CUDNN_VERSION >= 3000

    function cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode)
        cudnnCheck(ccall((:cudnnSetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},cudnnConvolutionMode_t),convDesc,arrayLength,padA,filterStrideA,upscaleA,mode))
    end

    function cudnnGetConvolutionNdDescriptor(convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode)
        cudnnCheck(ccall((:cudnnGetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t}),convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode))
    end

end # if CUDNN_VERSION >= 3000

function cudnnGetConvolutionNdForwardOutputDim(convDesc,inputTensorDesc,filterDesc,nbDims,tensorOuputDimA)
    cudnnCheck(ccall((:cudnnGetConvolutionNdForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Cint,Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,nbDims,tensorOuputDimA))
end

function cudnnDestroyConvolutionDescriptor(convDesc)
    cudnnCheck(ccall((:cudnnDestroyConvolutionDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,),convDesc))
end

function cudnnFindConvolutionForwardAlgorithm(handle,srcDesc,filterDesc,convDesc,destDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
    cudnnCheck(ccall((:cudnnFindConvolutionForwardAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionFwdAlgoPerf_t}),handle,srcDesc,filterDesc,convDesc,destDesc,requestedAlgoCount,returnedAlgoCount,perfResults))
end

function cudnnGetConvolutionForwardAlgorithm(handle,srcDesc,filterDesc,convDesc,destDesc,preference,memoryLimitInbytes,algo)
    cudnnCheck(ccall((:cudnnGetConvolutionForwardAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionFwdPreference_t,Csize_t,Ptr{cudnnConvolutionFwdAlgo_t}),handle,srcDesc,filterDesc,convDesc,destDesc,preference,memoryLimitInbytes,algo))
end

function cudnnGetConvolutionForwardWorkspaceSize(handle,srcDesc,filterDesc,convDesc,destDesc,algo,sizeInBytes)
    cudnnCheck(ccall((:cudnnGetConvolutionForwardWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Csize_t}),handle,srcDesc,filterDesc,convDesc,destDesc,algo,sizeInBytes))
end

function cudnnConvolutionForward(handle,alpha,srcDesc,srcData,filterDesc,filterData,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,destDesc,destData)
    cudnnCheck(ccall((:cudnnConvolutionForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Void},Csize_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,filterDesc,filterData,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,destDesc,destData))
end

function cudnnConvolutionBackwardBias(handle,alpha,srcDesc,srcData,beta,destDesc,destData)
    cudnnCheck(ccall((:cudnnConvolutionBackwardBias,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,beta,destDesc,destData))
end

function cudnnFindConvolutionBackwardFilterAlgorithm(handle,srcDesc,diffDesc,convDesc,gradDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
    cudnnCheck(ccall((:cudnnFindConvolutionBackwardFilterAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}),handle,srcDesc,diffDesc,convDesc,gradDesc,requestedAlgoCount,returnedAlgoCount,perfResults))
end

function cudnnGetConvolutionBackwardFilterAlgorithm(handle,srcDesc,diffDesc,convDesc,gradDesc,preference,memoryLimitInbytes,algo)
    cudnnCheck(ccall((:cudnnGetConvolutionBackwardFilterAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionBwdFilterPreference_t,Csize_t,Ptr{cudnnConvolutionBwdFilterAlgo_t}),handle,srcDesc,diffDesc,convDesc,gradDesc,preference,memoryLimitInbytes,algo))
end

function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,srcDesc,diffDesc,convDesc,gradDesc,algo,sizeInBytes)
    cudnnCheck(ccall((:cudnnGetConvolutionBackwardFilterWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionBwdFilterAlgo_t,Ptr{Csize_t}),handle,srcDesc,diffDesc,convDesc,gradDesc,algo,sizeInBytes))
end

function cudnnFindConvolutionBackwardDataAlgorithm(handle,filterDesc,diffDesc,convDesc,gradDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
    cudnnCheck(ccall((:cudnnFindConvolutionBackwardDataAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionBwdDataAlgoPerf_t}),handle,filterDesc,diffDesc,convDesc,gradDesc,requestedAlgoCount,returnedAlgoCount,perfResults))
end

function cudnnGetConvolutionBackwardDataAlgorithm(handle,filterDesc,diffDesc,convDesc,gradDesc,preference,memoryLimitInbytes,algo)
    cudnnCheck(ccall((:cudnnGetConvolutionBackwardDataAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionBwdDataPreference_t,Csize_t,Ptr{cudnnConvolutionBwdDataAlgo_t}),handle,filterDesc,diffDesc,convDesc,gradDesc,preference,memoryLimitInbytes,algo))
end

function cudnnGetConvolutionBackwardDataWorkspaceSize(handle,filterDesc,diffDesc,convDesc,gradDesc,algo,sizeInBytes)
    cudnnCheck(ccall((:cudnnGetConvolutionBackwardDataWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionBwdDataAlgo_t,Ptr{Csize_t}),handle,filterDesc,diffDesc,convDesc,gradDesc,algo,sizeInBytes))
end

if CUDNN_VERSION >= 3000

    function cudnnConvolutionBackwardFilter(handle,alpha,srcDesc,srcData,diffDesc,diffData,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,gradDesc,gradData)
        cudnnCheck(ccall((:cudnnConvolutionBackwardFilter_v3,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionBwdFilterAlgo_t,Ptr{Void},Csize_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,diffDesc,diffData,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,gradDesc,gradData))
    end

    function cudnnConvolutionBackwardData(handle,alpha,filterDesc,filterData,diffDesc,diffData,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,gradDesc,gradData)
        cudnnCheck(ccall((:cudnnConvolutionBackwardData_v3,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionBwdDataAlgo_t,Ptr{Void},Csize_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,filterDesc,filterData,diffDesc,diffData,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,gradDesc,gradData))
    end

else # if CUDNN_VERSION >= 3000

    function cudnnConvolutionBackwardFilter(handle,alpha,srcDesc,srcData,diffDesc,diffData,convDesc,beta,gradDesc,gradData)
        cudnnCheck(ccall((:cudnnConvolutionBackwardFilter,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,diffDesc,diffData,convDesc,beta,gradDesc,gradData))
    end

    function cudnnConvolutionBackwardData(handle,alpha,filterDesc,filterData,diffDesc,diffData,convDesc,beta,gradDesc,gradData)
        cudnnCheck(ccall((:cudnnConvolutionBackwardData,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,filterDesc,filterData,diffDesc,diffData,convDesc,beta,gradDesc,gradData))
    end

end # if CUDNN_VERSION >= 3000


function cudnnIm2Col(handle,srcDesc,srcData,filterDesc,convDesc,colBuffer)
    cudnnCheck(ccall((:cudnnIm2Col,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,Ptr{Void}),handle,srcDesc,srcData,filterDesc,convDesc,colBuffer))
end

function cudnnSoftmaxForward(handle,algorithm,mode,alpha,srcDesc,srcData,beta,destDesc,destData)
    cudnnCheck(ccall((:cudnnSoftmaxForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algorithm,mode,alpha,srcDesc,srcData,beta,destDesc,destData))
end

function cudnnSoftmaxBackward(handle,algorithm,mode,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,beta,destDiffDesc,destDiffData)
    cudnnCheck(ccall((:cudnnSoftmaxBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algorithm,mode,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,beta,destDiffDesc,destDiffData))
end

function cudnnCreatePoolingDescriptor(poolingDesc)
    cudnnCheck(ccall((:cudnnCreatePoolingDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnPoolingDescriptor_t},),poolingDesc))
end

function cudnnSetPooling2dDescriptor(poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    cudnnCheck(ccall((:cudnnSetPooling2dDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,Cint,Cint,Cint,Cint,Cint,Cint),poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride))
end

function cudnnGetPooling2dDescriptor(poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    cudnnCheck(ccall((:cudnnGetPooling2dDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Ptr{cudnnPoolingMode_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride))
end

function cudnnSetPoolingNdDescriptor(poolingDesc,mode,nbDims,windowDimA,paddingA,strideA)
    cudnnCheck(ccall((:cudnnSetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,nbDims,windowDimA,paddingA,strideA))
end

function cudnnGetPoolingNdDescriptor(poolingDesc,nbDimsRequested,mode,nbDims,windowDimA,paddingA,strideA)
    cudnnCheck(ccall((:cudnnGetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Cint,Ptr{cudnnPoolingMode_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,nbDimsRequested,mode,nbDims,windowDimA,paddingA,strideA))
end

function cudnnGetPoolingNdForwardOutputDim(poolingDesc,inputTensorDesc,nbDims,outputTensorDimA)
    cudnnCheck(ccall((:cudnnGetPoolingNdForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint}),poolingDesc,inputTensorDesc,nbDims,outputTensorDimA))
end

function cudnnGetPooling2dForwardOutputDim(poolingDesc,inputTensorDesc,outN,outC,outH,outW)
    cudnnCheck(ccall((:cudnnGetPooling2dForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,inputTensorDesc,outN,outC,outH,outW))
end

function cudnnDestroyPoolingDescriptor(poolingDesc)
    cudnnCheck(ccall((:cudnnDestroyPoolingDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,),poolingDesc))
end

function cudnnPoolingForward(handle,poolingDesc,alpha,srcDesc,srcData,beta,destDesc,destData)
    cudnnCheck(ccall((:cudnnPoolingForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,srcDesc,srcData,beta,destDesc,destData))
end

function cudnnPoolingBackward(handle,poolingDesc,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData)
    cudnnCheck(ccall((:cudnnPoolingBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData))
end

function cudnnActivationForward(handle,mode,alpha,srcDesc,srcData,beta,destDesc,destData)
    cudnnCheck(ccall((:cudnnActivationForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,srcDesc,srcData,beta,destDesc,destData))
end

function cudnnActivationBackward(handle,mode,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData)
    cudnnCheck(ccall((:cudnnActivationBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData))
end

function cudnnCreateLRNDescriptor(normDesc)
    cudnnCheck(ccall((:cudnnCreateLRNDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnLRNDescriptor_t},),normDesc))
end

function cudnnSetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
    cudnnCheck(ccall((:cudnnSetLRNDescriptor,libcudnn),cudnnStatus_t,(cudnnLRNDescriptor_t,UInt32,Cdouble,Cdouble,Cdouble),normDesc,lrnN,lrnAlpha,lrnBeta,lrnK))
end

function cudnnGetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
    cudnnCheck(ccall((:cudnnGetLRNDescriptor,libcudnn),cudnnStatus_t,(cudnnLRNDescriptor_t,Ptr{UInt32},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),normDesc,lrnN,lrnAlpha,lrnBeta,lrnK))
end

function cudnnDestroyLRNDescriptor(lrnDesc)
    cudnnCheck(ccall((:cudnnDestroyLRNDescriptor,libcudnn),cudnnStatus_t,(cudnnLRNDescriptor_t,),lrnDesc))
end

function cudnnLRNCrossChannelForward(handle,normDesc,lrnMode,alpha,srcDesc,srcData,beta,destDesc,destData)
    cudnnCheck(ccall((:cudnnLRNCrossChannelForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnLRNMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,lrnMode,alpha,srcDesc,srcData,beta,destDesc,destData))
end

function cudnnLRNCrossChannelBackward(handle,normDesc,lrnMode,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData)
    cudnnCheck(ccall((:cudnnLRNCrossChannelBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnLRNMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,lrnMode,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData))
end

function cudnnDivisiveNormalizationForward(handle,normDesc,mode,alpha,srcDesc,srcData,srcMeansData,tempData,tempData2,beta,destDesc,destData)
    cudnnCheck(ccall((:cudnnDivisiveNormalizationForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnDivNormMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,mode,alpha,srcDesc,srcData,srcMeansData,tempData,tempData2,beta,destDesc,destData))
end

function cudnnDivisiveNormalizationBackward(handle,normDesc,mode,alpha,srcDesc,srcData,srcMeansData,srcDiffData,tempData,tempData2,betaData,destDataDesc,destDataDiff,destMeansDiff)
    cudnnCheck(ccall((:cudnnDivisiveNormalizationBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnDivNormMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,normDesc,mode,alpha,srcDesc,srcData,srcMeansData,srcDiffData,tempData,tempData2,betaData,destDataDesc,destDataDiff,destMeansDiff))
end
