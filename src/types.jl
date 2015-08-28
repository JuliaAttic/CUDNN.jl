
const CUDNN_VERSION = 2000

type cudnnContext
end

typealias cudnnHandle_t Ptr{cudnnContext}

# begin enum cudnnStatus_t
typealias cudnnStatus_t UInt32
const CUDNN_STATUS_SUCCESS = (UInt32)(0)
const CUDNN_STATUS_NOT_INITIALIZED = (UInt32)(1)
const CUDNN_STATUS_ALLOC_FAILED = (UInt32)(2)
const CUDNN_STATUS_BAD_PARAM = (UInt32)(3)
const CUDNN_STATUS_INTERNAL_ERROR = (UInt32)(4)
const CUDNN_STATUS_INVALID_VALUE = (UInt32)(5)
const CUDNN_STATUS_ARCH_MISMATCH = (UInt32)(6)
const CUDNN_STATUS_MAPPING_ERROR = (UInt32)(7)
const CUDNN_STATUS_EXECUTION_FAILED = (UInt32)(8)
const CUDNN_STATUS_NOT_SUPPORTED = (UInt32)(9)
const CUDNN_STATUS_LICENSE_ERROR = (UInt32)(10)
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
typealias cudnnDataType_t UInt32
const CUDNN_DATA_FLOAT = (UInt32)(0)
const CUDNN_DATA_DOUBLE = (UInt32)(1)
# end enum cudnnDataType_t

# begin enum cudnnTensorFormat_t
typealias cudnnTensorFormat_t UInt32
const CUDNN_TENSOR_NCHW = (UInt32)(0)
const CUDNN_TENSOR_NHWC = (UInt32)(1)
# end enum cudnnTensorFormat_t

# begin enum cudnnAddMode_t
typealias cudnnAddMode_t UInt32
const CUDNN_ADD_IMAGE = (UInt32)(0)
const CUDNN_ADD_SAME_HW = (UInt32)(0)
const CUDNN_ADD_FEATURE_MAP = (UInt32)(1)
const CUDNN_ADD_SAME_CHW = (UInt32)(1)
const CUDNN_ADD_SAME_C = (UInt32)(2)
const CUDNN_ADD_FULL_TENSOR = (UInt32)(3)
# end enum cudnnAddMode_t

# begin enum cudnnConvolutionMode_t
typealias cudnnConvolutionMode_t UInt32
const CUDNN_CONVOLUTION = (UInt32)(0)
const CUDNN_CROSS_CORRELATION = (UInt32)(1)
# end enum cudnnConvolutionMode_t

# begin enum cudnnConvolutionFwdPreference_t
typealias cudnnConvolutionFwdPreference_t UInt32
const CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum cudnnConvolutionFwdPreference_t

# begin enum cudnnConvolutionFwdAlgo_t
typealias cudnnConvolutionFwdAlgo_t UInt32
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = (UInt32)(0)
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = (UInt32)(1)
const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = (UInt32)(2)
const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = (UInt32)(3)
# end enum cudnnConvolutionFwdAlgo_t

# begin enum cudnnSoftmaxAlgorithm_t
typealias cudnnSoftmaxAlgorithm_t UInt32
const CUDNN_SOFTMAX_FAST = (UInt32)(0)
const CUDNN_SOFTMAX_ACCURATE = (UInt32)(1)
# end enum cudnnSoftmaxAlgorithm_t

# begin enum cudnnSoftmaxMode_t
typealias cudnnSoftmaxMode_t UInt32
const CUDNN_SOFTMAX_MODE_INSTANCE = (UInt32)(0)
const CUDNN_SOFTMAX_MODE_CHANNEL = (UInt32)(1)
# end enum cudnnSoftmaxMode_t

# begin enum cudnnPoolingMode_t
typealias cudnnPoolingMode_t UInt32
const CUDNN_POOLING_MAX = (UInt32)(0)
const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = (UInt32)(1)
const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = (UInt32)(2)
# end enum cudnnPoolingMode_t

# begin enum cudnnActivationMode_t
typealias cudnnActivationMode_t UInt32
const CUDNN_ACTIVATION_SIGMOID = (UInt32)(0)
const CUDNN_ACTIVATION_RELU = (UInt32)(1)
const CUDNN_ACTIVATION_TANH = (UInt32)(2)
# end enum cudnnActivationMode_t
