# Julia wrapper for header: /mnt/ai/home/dyuret/src/cudnn/4.0.4/include/cudnn.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cudnnGetVersion()
    ccall((:cudnnGetVersion,libcudnn),Cint,())
end

function cudnnGetErrorString(status)
    ccall((:cudnnGetErrorString,libcudnn),Ptr{UInt8},(cudnnStatus_t,),status)
end

function cudnnCreate(handle)
    ccall((:cudnnCreate,libcudnn),cudnnStatus_t,(Ptr{cudnnHandle_t},),handle)
end

function cudnnDestroy(handle)
    ccall((:cudnnDestroy,libcudnn),cudnnStatus_t,(cudnnHandle_t,),handle)
end

function cudnnSetStream(handle,streamId)
    ccall((:cudnnSetStream,libcudnn),cudnnStatus_t,(cudnnHandle_t,Cint),handle,streamId)
end

function cudnnGetStream(handle,streamId)
    ccall((:cudnnGetStream,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Cint}),handle,streamId)
end

function cudnnCreateTensorDescriptor(tensorDesc)
    ccall((:cudnnCreateTensorDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnTensorDescriptor_t},),tensorDesc)
end

function cudnnSetTensor4dDescriptor(tensorDesc,format,dataType,n,c,h,w)
    ccall((:cudnnSetTensor4dDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnTensorFormat_t,cudnnDataType_t,Cint,Cint,Cint,Cint),tensorDesc,format,dataType,n,c,h,w)
end

function cudnnSetTensor4dDescriptorEx(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
    ccall((:cudnnSetTensor4dDescriptorEx,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
end

function cudnnGetTensor4dDescriptor(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
    ccall((:cudnnGetTensor4dDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
end

function cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA)
    ccall((:cudnnSetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,nbDims,dimA,strideA)
end

function cudnnGetTensorNdDescriptor(tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA)
    ccall((:cudnnGetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA)
end

function cudnnDestroyTensorDescriptor(tensorDesc)
    ccall((:cudnnDestroyTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,),tensorDesc)
end

function cudnnTransformTensor(handle,alpha,xDesc,x,beta,yDesc,y)
    ccall((:cudnnTransformTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,beta,yDesc,y)
end

function cudnnAddTensor(handle,alpha,bDesc,b,beta,yDesc,y)
    ccall((:cudnnAddTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,bDesc,b,beta,yDesc,y)
end

function cudnnAddTensor_v3(handle,alpha,bDesc,b,beta,yDesc,y)
    ccall((:cudnnAddTensor_v3,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,bDesc,b,beta,yDesc,y)
end

function cudnnSetTensor(handle,yDesc,y,valuePtr)
    ccall((:cudnnSetTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,yDesc,y,valuePtr)
end

function cudnnScaleTensor(handle,yDesc,y,alpha)
    ccall((:cudnnScaleTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,yDesc,y,alpha)
end

function cudnnCreateFilterDescriptor(filterDesc)
    ccall((:cudnnCreateFilterDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnFilterDescriptor_t},),filterDesc)
end

function cudnnSetFilter4dDescriptor(filterDesc,dataType,k,c,h,w)
    ccall((:cudnnSetFilter4dDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,Cint,Cint,Cint,Cint),filterDesc,dataType,k,c,h,w)
end

function cudnnSetFilter4dDescriptor_v4(filterDesc,dataType,format,k,c,h,w)
    ccall((:cudnnSetFilter4dDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,cudnnTensorFormat_t,Cint,Cint,Cint,Cint),filterDesc,dataType,format,k,c,h,w)
end

function cudnnGetFilter4dDescriptor(filterDesc,dataType,k,c,h,w)
    ccall((:cudnnGetFilter4dDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),filterDesc,dataType,k,c,h,w)
end

function cudnnGetFilter4dDescriptor_v4(filterDesc,dataType,format,k,c,h,w)
    ccall((:cudnnGetFilter4dDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Ptr{cudnnDataType_t},Ptr{cudnnTensorFormat_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),filterDesc,dataType,format,k,c,h,w)
end

function cudnnSetFilterNdDescriptor(filterDesc,dataType,nbDims,filterDimA)
    ccall((:cudnnSetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint}),filterDesc,dataType,nbDims,filterDimA)
end

function cudnnSetFilterNdDescriptor_v4(filterDesc,dataType,format,nbDims,filterDimA)
    ccall((:cudnnSetFilterNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,cudnnTensorFormat_t,Cint,Ptr{Cint}),filterDesc,dataType,format,nbDims,filterDimA)
end

function cudnnGetFilterNdDescriptor(filterDesc,nbDimsRequested,dataType,nbDims,filterDimA)
    ccall((:cudnnGetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,nbDims,filterDimA)
end

function cudnnGetFilterNdDescriptor_v4(filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA)
    ccall((:cudnnGetFilterNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{cudnnTensorFormat_t},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA)
end

function cudnnDestroyFilterDescriptor(filterDesc)
    ccall((:cudnnDestroyFilterDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,),filterDesc)
end

function cudnnCreateConvolutionDescriptor(convDesc)
    ccall((:cudnnCreateConvolutionDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnConvolutionDescriptor_t},),convDesc)
end

function cudnnSetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
    ccall((:cudnnSetConvolution2dDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint,cudnnConvolutionMode_t),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
end

function cudnnGetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
    ccall((:cudnnGetConvolution2dDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t}),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
end

function cudnnGetConvolution2dForwardOutputDim(convDesc,inputTensorDesc,filterDesc,n,c,h,w)
    ccall((:cudnnGetConvolution2dForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,n,c,h,w)
end

function cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType)
    ccall((:cudnnSetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},cudnnConvolutionMode_t,cudnnDataType_t),convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType)
end

function cudnnGetConvolutionNdDescriptor(convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType)
    ccall((:cudnnGetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t},Ptr{cudnnDataType_t}),convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType)
end

function cudnnSetConvolutionNdDescriptor_v3(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType)
    ccall((:cudnnSetConvolutionNdDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},cudnnConvolutionMode_t,cudnnDataType_t),convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType)
end

function cudnnGetConvolutionNdDescriptor_v3(convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType)
    ccall((:cudnnGetConvolutionNdDescriptor_v3,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t},Ptr{cudnnDataType_t}),convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode,dataType)
end

function cudnnGetConvolutionNdForwardOutputDim(convDesc,inputTensorDesc,filterDesc,nbDims,tensorOuputDimA)
    ccall((:cudnnGetConvolutionNdForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Cint,Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,nbDims,tensorOuputDimA)
end

function cudnnDestroyConvolutionDescriptor(convDesc)
    ccall((:cudnnDestroyConvolutionDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,),convDesc)
end

function cudnnFindConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
    ccall((:cudnnFindConvolutionForwardAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionFwdAlgoPerf_t}),handle,xDesc,wDesc,convDesc,yDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
end

function cudnnGetConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,preference,memoryLimitInBytes,algo)
    ccall((:cudnnGetConvolutionForwardAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionFwdPreference_t,Cint,Ptr{cudnnConvolutionFwdAlgo_t}),handle,xDesc,wDesc,convDesc,yDesc,preference,memoryLimitInBytes,algo)
end

function cudnnGetConvolutionForwardWorkspaceSize(handle,xDesc,wDesc,convDesc,yDesc,algo,sizeInBytes)
    ccall((:cudnnGetConvolutionForwardWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Cint}),handle,xDesc,wDesc,convDesc,yDesc,algo,sizeInBytes)
end

function cudnnConvolutionForward(handle,alpha,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,yDesc,y)
    ccall((:cudnnConvolutionForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Void},Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,yDesc,y)
end

function cudnnConvolutionBackwardBias(handle,alpha,dyDesc,dy,beta,dbDesc,db)
    ccall((:cudnnConvolutionBackwardBias,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,dyDesc,dy,beta,dbDesc,db)
end

function cudnnFindConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
    ccall((:cudnnFindConvolutionBackwardFilterAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}),handle,xDesc,dyDesc,convDesc,dwDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
end

function cudnnGetConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,preference,memoryLimitInBytes,algo)
    ccall((:cudnnGetConvolutionBackwardFilterAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionBwdFilterPreference_t,Cint,Ptr{cudnnConvolutionBwdFilterAlgo_t}),handle,xDesc,dyDesc,convDesc,dwDesc,preference,memoryLimitInBytes,algo)
end

function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,xDesc,dyDesc,convDesc,gradDesc,algo,sizeInBytes)
    ccall((:cudnnGetConvolutionBackwardFilterWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionBwdFilterAlgo_t,Ptr{Cint}),handle,xDesc,dyDesc,convDesc,gradDesc,algo,sizeInBytes)
end

function cudnnConvolutionBackwardFilter(handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw)
    ccall((:cudnnConvolutionBackwardFilter,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionBwdFilterAlgo_t,Ptr{Void},Cint,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw)
end

function cudnnConvolutionBackwardFilter_v3(handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw)
    ccall((:cudnnConvolutionBackwardFilter_v3,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionBwdFilterAlgo_t,Ptr{Void},Cint,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw)
end

function cudnnFindConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
    ccall((:cudnnFindConvolutionBackwardDataAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionBwdDataAlgoPerf_t}),handle,wDesc,dyDesc,convDesc,dxDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
end

function cudnnGetConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,preference,memoryLimitInBytes,algo)
    ccall((:cudnnGetConvolutionBackwardDataAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionBwdDataPreference_t,Cint,Ptr{cudnnConvolutionBwdDataAlgo_t}),handle,wDesc,dyDesc,convDesc,dxDesc,preference,memoryLimitInBytes,algo)
end

function cudnnGetConvolutionBackwardDataWorkspaceSize(handle,wDesc,dyDesc,convDesc,dxDesc,algo,sizeInBytes)
    ccall((:cudnnGetConvolutionBackwardDataWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionBwdDataAlgo_t,Ptr{Cint}),handle,wDesc,dyDesc,convDesc,dxDesc,algo,sizeInBytes)
end

function cudnnConvolutionBackwardData(handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx)
    ccall((:cudnnConvolutionBackwardData,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionBwdDataAlgo_t,Ptr{Void},Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx)
end

function cudnnConvolutionBackwardData_v3(handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx)
    ccall((:cudnnConvolutionBackwardData_v3,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionBwdDataAlgo_t,Ptr{Void},Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx)
end

function cudnnIm2Col(handle,xDesc,x,wDesc,convDesc,colBuffer)
    ccall((:cudnnIm2Col,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,Ptr{Void}),handle,xDesc,x,wDesc,convDesc,colBuffer)
end

function cudnnSoftmaxForward(handle,algo,mode,alpha,xDesc,x,beta,yDesc,y)
    ccall((:cudnnSoftmaxForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algo,mode,alpha,xDesc,x,beta,yDesc,y)
end

function cudnnSoftmaxBackward(handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx)
    ccall((:cudnnSoftmaxBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx)
end

function cudnnCreatePoolingDescriptor(poolingDesc)
    ccall((:cudnnCreatePoolingDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnPoolingDescriptor_t},),poolingDesc)
end

function cudnnSetPooling2dDescriptor(poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    ccall((:cudnnSetPooling2dDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,Cint,Cint,Cint,Cint,Cint,Cint),poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
end

function cudnnSetPooling2dDescriptor_v4(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    ccall((:cudnnSetPooling2dDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,cudnnNanPropagation_t,Cint,Cint,Cint,Cint,Cint,Cint),poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
end

function cudnnGetPooling2dDescriptor(poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    ccall((:cudnnGetPooling2dDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Ptr{cudnnPoolingMode_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
end

function cudnnGetPooling2dDescriptor_v4(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
    ccall((:cudnnGetPooling2dDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Ptr{cudnnPoolingMode_t},Ptr{cudnnNanPropagation_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
end

function cudnnSetPoolingNdDescriptor(poolingDesc,mode,nbDims,windowDimA,paddingA,strideA)
    ccall((:cudnnSetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,nbDims,windowDimA,paddingA,strideA)
end

function cudnnSetPoolingNdDescriptor_v4(poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
    ccall((:cudnnSetPoolingNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,cudnnNanPropagation_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
end

function cudnnGetPoolingNdDescriptor(poolingDesc,nbDimsRequested,mode,nbDims,windowDimA,paddingA,strideA)
    ccall((:cudnnGetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Cint,Ptr{cudnnPoolingMode_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,nbDimsRequested,mode,nbDims,windowDimA,paddingA,strideA)
end

function cudnnGetPoolingNdDescriptor_v4(poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
    ccall((:cudnnGetPoolingNdDescriptor_v4,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Cint,Ptr{cudnnPoolingMode_t},Ptr{cudnnNanPropagation_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA)
end

function cudnnGetPoolingNdForwardOutputDim(poolingDesc,inputTensorDesc,nbDims,outputTensorDimA)
    ccall((:cudnnGetPoolingNdForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint}),poolingDesc,inputTensorDesc,nbDims,outputTensorDimA)
end

function cudnnGetPooling2dForwardOutputDim(poolingDesc,inputTensorDesc,n,c,h,w)
    ccall((:cudnnGetPooling2dForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,inputTensorDesc,n,c,h,w)
end

function cudnnDestroyPoolingDescriptor(poolingDesc)
    ccall((:cudnnDestroyPoolingDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,),poolingDesc)
end

function cudnnPoolingForward(handle,poolingDesc,alpha,xDesc,x,beta,yDesc,y)
    ccall((:cudnnPoolingForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,xDesc,x,beta,yDesc,y)
end

function cudnnPoolingBackward(handle,poolingDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    ccall((:cudnnPoolingBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
end

function cudnnCreateActivationDescriptor(activationDesc)
    ccall((:cudnnCreateActivationDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnActivationDescriptor_t},),activationDesc)
end

function cudnnSetActivationDescriptor(activationDesc,mode,reluNanOpt,reluCeiling)
    ccall((:cudnnSetActivationDescriptor,libcudnn),cudnnStatus_t,(cudnnActivationDescriptor_t,cudnnActivationMode_t,cudnnNanPropagation_t,Cdouble),activationDesc,mode,reluNanOpt,reluCeiling)
end

function cudnnGetActivationDescriptor(activationDesc,mode,reluNanOpt,reluCeiling)
    ccall((:cudnnGetActivationDescriptor,libcudnn),cudnnStatus_t,(cudnnActivationDescriptor_t,Ptr{cudnnActivationMode_t},Ptr{cudnnNanPropagation_t},Ptr{Cdouble}),activationDesc,mode,reluNanOpt,reluCeiling)
end

function cudnnDestroyActivationDescriptor(activationDesc)
    ccall((:cudnnDestroyActivationDescriptor,libcudnn),cudnnStatus_t,(cudnnActivationDescriptor_t,),activationDesc)
end

function cudnnActivationForward(handle,mode,alpha,xDesc,x,beta,yDesc,y)
    ccall((:cudnnActivationForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,xDesc,x,beta,yDesc,y)
end

function cudnnActivationForward_v4(handle,activationDesc,alpha,xDesc,x,beta,yDesc,y)
    ccall((:cudnnActivationForward_v4,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,activationDesc,alpha,xDesc,x,beta,yDesc,y)
end

function cudnnActivationBackward(handle,mode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    ccall((:cudnnActivationBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
end

function cudnnActivationBackward_v4(handle,activationDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    ccall((:cudnnActivationBackward_v4,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,activationDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
end

function cudnnCreateLRNDescriptor(normDesc)
    ccall((:cudnnCreateLRNDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnLRNDescriptor_t},),normDesc)
end

function cudnnSetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
    ccall((:cudnnSetLRNDescriptor,libcudnn),cudnnStatus_t,(cudnnLRNDescriptor_t,UInt32,Cdouble,Cdouble,Cdouble),normDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
end

function cudnnGetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
    ccall((:cudnnGetLRNDescriptor,libcudnn),cudnnStatus_t,(cudnnLRNDescriptor_t,Ptr{UInt32},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),normDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
end

function cudnnDestroyLRNDescriptor(lrnDesc)
    ccall((:cudnnDestroyLRNDescriptor,libcudnn),cudnnStatus_t,(cudnnLRNDescriptor_t,),lrnDesc)
end

function cudnnLRNCrossChannelForward(handle,normDesc,lrnMode,alpha,xDesc,x,beta,yDesc,y)
    ccall((:cudnnLRNCrossChannelForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnLRNMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,lrnMode,alpha,xDesc,x,beta,yDesc,y)
end

function cudnnLRNCrossChannelBackward(handle,normDesc,lrnMode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
    ccall((:cudnnLRNCrossChannelBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnLRNMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,lrnMode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx)
end

function cudnnDivisiveNormalizationForward(handle,normDesc,mode,alpha,xDesc,x,means,temp,temp2,beta,yDesc,y)
    ccall((:cudnnDivisiveNormalizationForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnDivNormMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,mode,alpha,xDesc,x,means,temp,temp2,beta,yDesc,y)
end

function cudnnDivisiveNormalizationBackward(handle,normDesc,mode,alpha,xDesc,x,means,dy,temp,temp2,beta,dXdMeansDesc,dx,dMeans)
    ccall((:cudnnDivisiveNormalizationBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnLRNDescriptor_t,cudnnDivNormMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,normDesc,mode,alpha,xDesc,x,means,dy,temp,temp2,beta,dXdMeansDesc,dx,dMeans)
end

function cudnnDeriveBNTensorDescriptor(derivedBnDesc,xDesc,mode)
    ccall((:cudnnDeriveBNTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnBatchNormMode_t),derivedBnDesc,xDesc,mode)
end

function cudnnBatchNormalizationForwardTraining(handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,resultRunningMean,resultRunningInvVariance,epsilon,resultSaveMean,resultSaveInvVariance)
    ccall((:cudnnBatchNormalizationForwardTraining,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnBatchNormMode_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Cdouble,Ptr{Void},Ptr{Void},Cdouble,Ptr{Void},Ptr{Void}),handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,resultRunningMean,resultRunningInvVariance,epsilon,resultSaveMean,resultSaveInvVariance)
end

function cudnnBatchNormalizationForwardInference(handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,estimatedMean,estimatedInvVariance,epsilon)
    ccall((:cudnnBatchNormalizationForwardInference,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnBatchNormMode_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Cdouble),handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,estimatedMean,estimatedInvVariance,epsilon)
end

function cudnnBatchNormalizationBackward(handle,mode,alpha,beta,xDesc,x,dyDesc,dy,dxDesc,dx,dBnScaleBiasDesc,bnScale,dBnScaleResult,dBnBiasResult,epsilon,savedMean,savedInvVariance)
    ccall((:cudnnBatchNormalizationBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnBatchNormMode_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Cdouble,Ptr{Void},Ptr{Void}),handle,mode,alpha,beta,xDesc,x,dyDesc,dy,dxDesc,dx,dBnScaleBiasDesc,bnScale,dBnScaleResult,dBnBiasResult,epsilon,savedMean,savedInvVariance)
end

function cudnnSetConvolutionNdDescriptor_v2(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode)
    ccall((:cudnnSetConvolutionNdDescriptor_v2,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},cudnnConvolutionMode_t),convDesc,arrayLength,padA,filterStrideA,upscaleA,mode)
end

function cudnnGetConvolutionNdDescriptor_v2(convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode)
    ccall((:cudnnGetConvolutionNdDescriptor_v2,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t}),convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode)
end

function cudnnAddTensor_v2(handle,mode,alpha,bDesc,b,beta,yDesc,y)
    ccall((:cudnnAddTensor_v2,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnAddMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,bDesc,b,beta,yDesc,y)
end

function cudnnConvolutionBackwardFilter_v2(handle,alpha,xDesc,x,dyDesc,dy,convDesc,beta,dxDesc,dx)
    ccall((:cudnnConvolutionBackwardFilter_v2,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,dyDesc,dy,convDesc,beta,dxDesc,dx)
end

function cudnnConvolutionBackwardData_v2(handle,alpha,xDesc,x,dyDesc,dy,convDesc,beta,dxDesc,dx)
    ccall((:cudnnConvolutionBackwardData_v2,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,dyDesc,dy,convDesc,beta,dxDesc,dx)
end


### To make things work on v2:

function cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode)
    ccall((:cudnnSetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},cudnnConvolutionMode_t),convDesc,arrayLength,padA,filterStrideA,upscaleA,mode)
end

function cudnnGetConvolutionNdDescriptor(convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode)
    ccall((:cudnnGetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t}),convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode)
end

function cudnnAddTensor(handle,mode,alpha,bDesc,b,beta,yDesc,y)
    ccall((:cudnnAddTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnAddMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,bDesc,b,beta,yDesc,y)
end

function cudnnConvolutionBackwardFilter(handle,alpha,xDesc,x,dyDesc,dy,convDesc,beta,dxDesc,dx)
    ccall((:cudnnConvolutionBackwardFilter,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,dyDesc,dy,convDesc,beta,dxDesc,dx)
end

function cudnnConvolutionBackwardData(handle,alpha,xDesc,x,dyDesc,dy,convDesc,beta,dxDesc,dx)
    ccall((:cudnnConvolutionBackwardData,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,xDesc,x,dyDesc,dy,convDesc,beta,dxDesc,dx)
end
