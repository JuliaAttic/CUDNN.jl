# Julia wrapper for header: /share/apps/cuDNN/cudnn-6.5-linux-x64-v2/cudnn.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cudnnGetVersion()
    ccall((:cudnnGetVersion,libcudnn),Csize_t,())
end

function cudnnGetErrorString(status::cudnnStatus_t)
    ccall((:cudnnGetErrorString,libcudnn),Ptr{Uint8},(cudnnStatus_t,),status)
end

function cudnnCreate(handle::Ptr{cudnnHandle_t})
    ccall((:cudnnCreate,libcudnn),cudnnStatus_t,(Ptr{cudnnHandle_t},),handle)
end

function cudnnDestroy(handle::cudnnHandle_t)
    ccall((:cudnnDestroy,libcudnn),cudnnStatus_t,(cudnnHandle_t,),handle)
end

# function cudnnSetStream(handle::cudnnHandle_t,streamId::cudaStream_t)
#     ccall((:cudnnSetStream,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudaStream_t),handle,streamId)
# end

# function cudnnGetStream(handle::cudnnHandle_t,streamId::Ptr{cudaStream_t})
#     ccall((:cudnnGetStream,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{cudaStream_t}),handle,streamId)
# end

function cudnnCreateTensorDescriptor(tensorDesc::Ptr{cudnnTensorDescriptor_t})
    ccall((:cudnnCreateTensorDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnTensorDescriptor_t},),tensorDesc)
end

function cudnnSetTensor4dDescriptor(tensorDesc::cudnnTensorDescriptor_t,format::cudnnTensorFormat_t,dataType::cudnnDataType_t,n::Cint,c::Cint,h::Cint,w::Cint)
    ccall((:cudnnSetTensor4dDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnTensorFormat_t,cudnnDataType_t,Cint,Cint,Cint,Cint),tensorDesc,format,dataType,n,c,h,w)
end

function cudnnSetTensor4dDescriptorEx(tensorDesc::cudnnTensorDescriptor_t,dataType::cudnnDataType_t,n::Cint,c::Cint,h::Cint,w::Cint,nStride::Cint,cStride::Cint,hStride::Cint,wStride::Cint)
    ccall((:cudnnSetTensor4dDescriptorEx,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
end

function cudnnGetTensor4dDescriptor(tensorDesc::cudnnTensorDescriptor_t,dataType::Ptr{cudnnDataType_t},n::Ptr{Cint},c::Ptr{Cint},h::Ptr{Cint},w::Ptr{Cint},nStride::Ptr{Cint},cStride::Ptr{Cint},hStride::Ptr{Cint},wStride::Ptr{Cint})
    ccall((:cudnnGetTensor4dDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
end

function cudnnSetTensorNdDescriptor(tensorDesc::cudnnTensorDescriptor_t,dataType::cudnnDataType_t,nbDims::Cint,dimA::Ptr{Cint},strideA::Ptr{Cint})
    ccall((:cudnnSetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,nbDims,dimA,strideA)
end

function cudnnGetTensorNdDescriptor(tensorDesc::cudnnTensorDescriptor_t,nbDimsRequested::Cint,dataType::Ptr{cudnnDataType_t},nbDims::Ptr{Cint},dimA::Ptr{Cint},strideA::Ptr{Cint})
    ccall((:cudnnGetTensorNdDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,nbDimsRequested,dataType,nbDims,dimA,strideA)
end

function cudnnDestroyTensorDescriptor(tensorDesc::cudnnTensorDescriptor_t)
    ccall((:cudnnDestroyTensorDescriptor,libcudnn),cudnnStatus_t,(cudnnTensorDescriptor_t,),tensorDesc)
end

function cudnnTransformTensor(handle::cudnnHandle_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},beta::Ptr{Void},destDesc::cudnnTensorDescriptor_t,destData::Ptr{Void})
    ccall((:cudnnTransformTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,beta,destDesc,destData)
end

function cudnnAddTensor(handle::cudnnHandle_t,mode::cudnnAddMode_t,alpha::Ptr{Void},biasDesc::cudnnTensorDescriptor_t,biasData::Ptr{Void},beta::Ptr{Void},srcDestDesc::cudnnTensorDescriptor_t,srcDestData::Ptr{Void})
    ccall((:cudnnAddTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnAddMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,biasDesc,biasData,beta,srcDestDesc,srcDestData)
end

function cudnnSetTensor(handle::cudnnHandle_t,srcDestDesc::cudnnTensorDescriptor_t,srcDestData::Ptr{Void},value::Ptr{Void})
    ccall((:cudnnSetTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,srcDestDesc,srcDestData,value)
end

function cudnnScaleTensor(handle::cudnnHandle_t,srcDestDesc::cudnnTensorDescriptor_t,srcDestData::Ptr{Void},alpha::Ptr{Void})
    ccall((:cudnnScaleTensor,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,srcDestDesc,srcDestData,alpha)
end

function cudnnCreateFilterDescriptor(filterDesc::Ptr{cudnnFilterDescriptor_t})
    ccall((:cudnnCreateFilterDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnFilterDescriptor_t},),filterDesc)
end

function cudnnSetFilter4dDescriptor(filterDesc::cudnnFilterDescriptor_t,dataType::cudnnDataType_t,k::Cint,c::Cint,h::Cint,w::Cint)
    ccall((:cudnnSetFilter4dDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,Cint,Cint,Cint,Cint),filterDesc,dataType,k,c,h,w)
end

function cudnnGetFilter4dDescriptor(filterDesc::cudnnFilterDescriptor_t,dataType::Ptr{cudnnDataType_t},k::Ptr{Cint},c::Ptr{Cint},h::Ptr{Cint},w::Ptr{Cint})
    ccall((:cudnnGetFilter4dDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),filterDesc,dataType,k,c,h,w)
end

function cudnnSetFilterNdDescriptor(filterDesc::cudnnFilterDescriptor_t,dataType::cudnnDataType_t,nbDims::Cint,filterDimA::Ptr{Cint})
    ccall((:cudnnSetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,cudnnDataType_t,Cint,Ptr{Cint}),filterDesc,dataType,nbDims,filterDimA)
end

function cudnnGetFilterNdDescriptor(filterDesc::cudnnFilterDescriptor_t,nbDimsRequested::Cint,dataType::Ptr{cudnnDataType_t},nbDims::Ptr{Cint},filterDimA::Ptr{Cint})
    ccall((:cudnnGetFilterNdDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,Cint,Ptr{cudnnDataType_t},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,nbDims,filterDimA)
end

function cudnnDestroyFilterDescriptor(filterDesc::cudnnFilterDescriptor_t)
    ccall((:cudnnDestroyFilterDescriptor,libcudnn),cudnnStatus_t,(cudnnFilterDescriptor_t,),filterDesc)
end

function cudnnCreateConvolutionDescriptor(convDesc::Ptr{cudnnConvolutionDescriptor_t})
    ccall((:cudnnCreateConvolutionDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnConvolutionDescriptor_t},),convDesc)
end

function cudnnSetConvolution2dDescriptor(convDesc::cudnnConvolutionDescriptor_t,pad_h::Cint,pad_w::Cint,u::Cint,v::Cint,upscalex::Cint,upscaley::Cint,mode::cudnnConvolutionMode_t)
    ccall((:cudnnSetConvolution2dDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint,cudnnConvolutionMode_t),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
end

function cudnnGetConvolution2dDescriptor(convDesc::cudnnConvolutionDescriptor_t,pad_h::Ptr{Cint},pad_w::Ptr{Cint},u::Ptr{Cint},v::Ptr{Cint},upscalex::Ptr{Cint},upscaley::Ptr{Cint},mode::Ptr{cudnnConvolutionMode_t})
    ccall((:cudnnGetConvolution2dDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t}),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
end

function cudnnGetConvolution2dForwardOutputDim(convDesc::cudnnConvolutionDescriptor_t,inputTensorDesc::cudnnTensorDescriptor_t,filterDesc::cudnnFilterDescriptor_t,n::Ptr{Cint},c::Ptr{Cint},h::Ptr{Cint},w::Ptr{Cint})
    ccall((:cudnnGetConvolution2dForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,n,c,h,w)
end

function cudnnSetConvolutionNdDescriptor(convDesc::cudnnConvolutionDescriptor_t,arrayLength::Cint,padA::Ptr{Cint},filterStrideA::Ptr{Cint},upscaleA::Ptr{Cint},mode::cudnnConvolutionMode_t)
    ccall((:cudnnSetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},cudnnConvolutionMode_t),convDesc,arrayLength,padA,filterStrideA,upscaleA,mode)
end

function cudnnGetConvolutionNdDescriptor(convDesc::cudnnConvolutionDescriptor_t,arrayLengthRequested::Cint,arrayLength::Ptr{Cint},padA::Ptr{Cint},strideA::Ptr{Cint},upscaleA::Ptr{Cint},mode::Ptr{cudnnConvolutionMode_t})
    ccall((:cudnnGetConvolutionNdDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{cudnnConvolutionMode_t}),convDesc,arrayLengthRequested,arrayLength,padA,strideA,upscaleA,mode)
end

function cudnnGetConvolutionNdForwardOutputDim(convDesc::cudnnConvolutionDescriptor_t,inputTensorDesc::cudnnTensorDescriptor_t,filterDesc::cudnnFilterDescriptor_t,nbDims::Cint,tensorOuputDimA::Ptr{Cint})
    ccall((:cudnnGetConvolutionNdForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Cint,Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,nbDims,tensorOuputDimA)
end

function cudnnDestroyConvolutionDescriptor(convDesc::cudnnConvolutionDescriptor_t)
    ccall((:cudnnDestroyConvolutionDescriptor,libcudnn),cudnnStatus_t,(cudnnConvolutionDescriptor_t,),convDesc)
end

function cudnnGetConvolutionForwardAlgorithm(handle::cudnnHandle_t,srcDesc::cudnnTensorDescriptor_t,filterDesc::cudnnFilterDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,destDesc::cudnnTensorDescriptor_t,preference::cudnnConvolutionFwdPreference_t,memoryLimitInbytes::Csize_t,algo::Ptr{cudnnConvolutionFwdAlgo_t})
    ccall((:cudnnGetConvolutionForwardAlgorithm,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionFwdPreference_t,Csize_t,Ptr{cudnnConvolutionFwdAlgo_t}),handle,srcDesc,filterDesc,convDesc,destDesc,preference,memoryLimitInbytes,algo)
end

function cudnnGetConvolutionForwardWorkspaceSize(handle::cudnnHandle_t,srcDesc::cudnnTensorDescriptor_t,filterDesc::cudnnFilterDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,destDesc::cudnnTensorDescriptor_t,algo::cudnnConvolutionFwdAlgo_t,sizeInBytes::Ptr{Csize_t})
    ccall((:cudnnGetConvolutionForwardWorkspaceSize,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Csize_t}),handle,srcDesc,filterDesc,convDesc,destDesc,algo,sizeInBytes)
end

function cudnnConvolutionForward(handle::cudnnHandle_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},filterDesc::cudnnFilterDescriptor_t,filterData::Ptr{Void},convDesc::cudnnConvolutionDescriptor_t,algo::cudnnConvolutionFwdAlgo_t,workSpace::Ptr{Void},workSpaceSizeInBytes::Csize_t,beta::Ptr{Void},destDesc::cudnnTensorDescriptor_t,destData::Ptr{Void})
    ccall((:cudnnConvolutionForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,cudnnConvolutionFwdAlgo_t,Ptr{Void},Csize_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,filterDesc,filterData,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,destDesc,destData)
end

function cudnnConvolutionBackwardBias(handle::cudnnHandle_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},beta::Ptr{Void},destDesc::cudnnTensorDescriptor_t,destData::Ptr{Void})
    ccall((:cudnnConvolutionBackwardBias,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,beta,destDesc,destData)
end

function cudnnConvolutionBackwardFilter(handle::cudnnHandle_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},diffDesc::cudnnTensorDescriptor_t,diffData::Ptr{Void},convDesc::cudnnConvolutionDescriptor_t,beta::Ptr{Void},gradDesc::cudnnFilterDescriptor_t,gradData::Ptr{Void})
    ccall((:cudnnConvolutionBackwardFilter,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,diffDesc,diffData,convDesc,beta,gradDesc,gradData)
end

function cudnnConvolutionBackwardData(handle::cudnnHandle_t,alpha::Ptr{Void},filterDesc::cudnnFilterDescriptor_t,filterData::Ptr{Void},diffDesc::cudnnTensorDescriptor_t,diffData::Ptr{Void},convDesc::cudnnConvolutionDescriptor_t,beta::Ptr{Void},gradDesc::cudnnTensorDescriptor_t,gradData::Ptr{Void})
    ccall((:cudnnConvolutionBackwardData,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,filterDesc,filterData,diffDesc,diffData,convDesc,beta,gradDesc,gradData)
end

function cudnnIm2Col(handle::cudnnHandle_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},filterDesc::cudnnFilterDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,colBuffer::Ptr{Void})
    ccall((:cudnnIm2Col,libcudnn),cudnnStatus_t,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,filterDesc,convDesc,colBuffer)
end

function cudnnSoftmaxForward(handle::cudnnHandle_t,algorithm::cudnnSoftmaxAlgorithm_t,mode::cudnnSoftmaxMode_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},beta::Ptr{Void},destDesc::cudnnTensorDescriptor_t,destData::Ptr{Void})
    ccall((:cudnnSoftmaxForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algorithm,mode,alpha,srcDesc,srcData,beta,destDesc,destData)
end

function cudnnSoftmaxBackward(handle::cudnnHandle_t,algorithm::cudnnSoftmaxAlgorithm_t,mode::cudnnSoftmaxMode_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},srcDiffDesc::cudnnTensorDescriptor_t,srcDiffData::Ptr{Void},beta::Ptr{Void},destDiffDesc::cudnnTensorDescriptor_t,destDiffData::Ptr{Void})
    ccall((:cudnnSoftmaxBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnSoftmaxAlgorithm_t,cudnnSoftmaxMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algorithm,mode,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,beta,destDiffDesc,destDiffData)
end

function cudnnCreatePoolingDescriptor(poolingDesc::Ptr{cudnnPoolingDescriptor_t})
    ccall((:cudnnCreatePoolingDescriptor,libcudnn),cudnnStatus_t,(Ptr{cudnnPoolingDescriptor_t},),poolingDesc)
end

function cudnnSetPooling2dDescriptor(poolingDesc::cudnnPoolingDescriptor_t,mode::cudnnPoolingMode_t,windowHeight::Cint,windowWidth::Cint,verticalPadding::Cint,horizontalPadding::Cint,verticalStride::Cint,horizontalStride::Cint)
    ccall((:cudnnSetPooling2dDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,Cint,Cint,Cint,Cint,Cint,Cint),poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
end

function cudnnGetPooling2dDescriptor(poolingDesc::cudnnPoolingDescriptor_t,mode::Ptr{cudnnPoolingMode_t},windowHeight::Ptr{Cint},windowWidth::Ptr{Cint},verticalPadding::Ptr{Cint},horizontalPadding::Ptr{Cint},verticalStride::Ptr{Cint},horizontalStride::Ptr{Cint})
    ccall((:cudnnGetPooling2dDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Ptr{cudnnPoolingMode_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
end

function cudnnSetPoolingNdDescriptor(poolingDesc::cudnnPoolingDescriptor_t,mode::cudnnPoolingMode_t,nbDims::Cint,windowDimA::Ptr{Cint},paddingA::Ptr{Cint},strideA::Ptr{Cint})
    ccall((:cudnnSetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnPoolingMode_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,nbDims,windowDimA,paddingA,strideA)
end

function cudnnGetPoolingNdDescriptor(poolingDesc::cudnnPoolingDescriptor_t,nbDimsRequested::Cint,mode::Ptr{cudnnPoolingMode_t},nbDims::Ptr{Cint},windowDimA::Ptr{Cint},paddingA::Ptr{Cint},strideA::Ptr{Cint})
    ccall((:cudnnGetPoolingNdDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,Cint,Ptr{cudnnPoolingMode_t},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,nbDimsRequested,mode,nbDims,windowDimA,paddingA,strideA)
end

function cudnnGetPoolingNdForwardOutputDim(poolingDesc::cudnnPoolingDescriptor_t,inputTensorDesc::cudnnTensorDescriptor_t,nbDims::Cint,outputTensorDimA::Ptr{Cint})
    ccall((:cudnnGetPoolingNdForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint}),poolingDesc,inputTensorDesc,nbDims,outputTensorDimA)
end

function cudnnGetPooling2dForwardOutputDim(poolingDesc::cudnnPoolingDescriptor_t,inputTensorDesc::cudnnTensorDescriptor_t,outN::Ptr{Cint},outC::Ptr{Cint},outH::Ptr{Cint},outW::Ptr{Cint})
    ccall((:cudnnGetPooling2dForwardOutputDim,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,inputTensorDesc,outN,outC,outH,outW)
end

function cudnnDestroyPoolingDescriptor(poolingDesc::cudnnPoolingDescriptor_t)
    ccall((:cudnnDestroyPoolingDescriptor,libcudnn),cudnnStatus_t,(cudnnPoolingDescriptor_t,),poolingDesc)
end

function cudnnPoolingForward(handle::cudnnHandle_t,poolingDesc::cudnnPoolingDescriptor_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},beta::Ptr{Void},destDesc::cudnnTensorDescriptor_t,destData::Ptr{Void})
    ccall((:cudnnPoolingForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,srcDesc,srcData,beta,destDesc,destData)
end

function cudnnPoolingBackward(handle::cudnnHandle_t,poolingDesc::cudnnPoolingDescriptor_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},srcDiffDesc::cudnnTensorDescriptor_t,srcDiffData::Ptr{Void},destDesc::cudnnTensorDescriptor_t,destData::Ptr{Void},beta::Ptr{Void},destDiffDesc::cudnnTensorDescriptor_t,destDiffData::Ptr{Void})
    ccall((:cudnnPoolingBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData)
end

function cudnnActivationForward(handle::cudnnHandle_t,mode::cudnnActivationMode_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},beta::Ptr{Void},destDesc::cudnnTensorDescriptor_t,destData::Ptr{Void})
    ccall((:cudnnActivationForward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,srcDesc,srcData,beta,destDesc,destData)
end

function cudnnActivationBackward(handle::cudnnHandle_t,mode::cudnnActivationMode_t,alpha::Ptr{Void},srcDesc::cudnnTensorDescriptor_t,srcData::Ptr{Void},srcDiffDesc::cudnnTensorDescriptor_t,srcDiffData::Ptr{Void},destDesc::cudnnTensorDescriptor_t,destData::Ptr{Void},beta::Ptr{Void},destDiffDesc::cudnnTensorDescriptor_t,destDiffData::Ptr{Void})
    ccall((:cudnnActivationBackward,libcudnn),cudnnStatus_t,(cudnnHandle_t,cudnnActivationMode_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,beta,destDiffDesc,destDiffData)
end
