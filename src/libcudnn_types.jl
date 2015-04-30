
const CUDNN_VERSION = 2000

type cudnnContext
end

typealias cudnnHandle_t Ptr{cudnnContext}

# begin enum cudnnStatus_t
typealias cudnnStatus_t Uint32
const CUDNN_STATUS_SUCCESS = (uint32)(0)
const CUDNN_STATUS_NOT_INITIALIZED = (uint32)(1)
const CUDNN_STATUS_ALLOC_FAILED = (uint32)(2)
const CUDNN_STATUS_BAD_PARAM = (uint32)(3)
const CUDNN_STATUS_INTERNAL_ERROR = (uint32)(4)
const CUDNN_STATUS_INVALID_VALUE = (uint32)(5)
const CUDNN_STATUS_ARCH_MISMATCH = (uint32)(6)
const CUDNN_STATUS_MAPPING_ERROR = (uint32)(7)
const CUDNN_STATUS_EXECUTION_FAILED = (uint32)(8)
const CUDNN_STATUS_NOT_SUPPORTED = (uint32)(9)
const CUDNN_STATUS_LICENSE_ERROR = (uint32)(10)
# end enum cudnnStatus_t

type cudnnTensorStruct
end

typealias cudnnTensorDescriptor_t Ptr{cudnnTensorStruct}

type cudnnConvolutionStruct
end

typealias cudnnConvolutionDescriptor_t Ptr{cudnnConvolutionStruct}

type cudnnPoolingStruct
end

typealias cudnnPoolingDescriptor_t Ptr{cudnnPoolingStruct}

type cudnnFilterStruct
end

typealias cudnnFilterDescriptor_t Ptr{cudnnFilterStruct}

# begin enum cudnnDataType_t
typealias cudnnDataType_t Uint32
const CUDNN_DATA_FLOAT = (uint32)(0)
const CUDNN_DATA_DOUBLE = (uint32)(1)
# end enum cudnnDataType_t

# begin enum cudnnTensorFormat_t
typealias cudnnTensorFormat_t Uint32
const CUDNN_TENSOR_NCHW = (uint32)(0)
const CUDNN_TENSOR_NHWC = (uint32)(1)
# end enum cudnnTensorFormat_t

# begin enum cudnnAddMode_t
typealias cudnnAddMode_t Uint32
const CUDNN_ADD_IMAGE = (uint32)(0)
const CUDNN_ADD_SAME_HW = (uint32)(0)
const CUDNN_ADD_FEATURE_MAP = (uint32)(1)
const CUDNN_ADD_SAME_CHW = (uint32)(1)
const CUDNN_ADD_SAME_C = (uint32)(2)
const CUDNN_ADD_FULL_TENSOR = (uint32)(3)
# end enum cudnnAddMode_t

# begin enum cudnnConvolutionMode_t
typealias cudnnConvolutionMode_t Uint32
const CUDNN_CONVOLUTION = (uint32)(0)
const CUDNN_CROSS_CORRELATION = (uint32)(1)
# end enum cudnnConvolutionMode_t

# begin enum cudnnConvolutionFwdPreference_t
typealias cudnnConvolutionFwdPreference_t Uint32
const CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = (uint32)(0)
const CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = (uint32)(1)
const CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = (uint32)(2)
# end enum cudnnConvolutionFwdPreference_t

# begin enum cudnnConvolutionFwdAlgo_t
typealias cudnnConvolutionFwdAlgo_t Uint32
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = (uint32)(0)
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = (uint32)(1)
const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = (uint32)(2)
const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = (uint32)(3)
# end enum cudnnConvolutionFwdAlgo_t

# begin enum cudnnSoftmaxAlgorithm_t
typealias cudnnSoftmaxAlgorithm_t Uint32
const CUDNN_SOFTMAX_FAST = (uint32)(0)
const CUDNN_SOFTMAX_ACCURATE = (uint32)(1)
# end enum cudnnSoftmaxAlgorithm_t

# begin enum cudnnSoftmaxMode_t
typealias cudnnSoftmaxMode_t Uint32
const CUDNN_SOFTMAX_MODE_INSTANCE = (uint32)(0)
const CUDNN_SOFTMAX_MODE_CHANNEL = (uint32)(1)
# end enum cudnnSoftmaxMode_t

# begin enum cudnnPoolingMode_t
typealias cudnnPoolingMode_t Uint32
const CUDNN_POOLING_MAX = (uint32)(0)
const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = (uint32)(1)
const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = (uint32)(2)
# end enum cudnnPoolingMode_t

# begin enum cudnnActivationMode_t
typealias cudnnActivationMode_t Uint32
const CUDNN_ACTIVATION_SIGMOID = (uint32)(0)
const CUDNN_ACTIVATION_RELU = (uint32)(1)
const CUDNN_ACTIVATION_TANH = (uint32)(2)
# end enum cudnnActivationMode_t
