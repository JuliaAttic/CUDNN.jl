# Automatically generated using Clang.jl wrap_c, version 0.0.0

using Compat

# const CUDNN_MAJOR = 4
# const CUDNN_MINOR = 0
# const CUDNN_PATCHLEVEL = 4
# const CUDNN_VERSION = CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL
const CUDNN_DIM_MAX = 8
const CUDNN_LRN_MIN_N = 1
const CUDNN_LRN_MAX_N = 16
const CUDNN_LRN_MIN_K = 1.0e-5
const CUDNN_LRN_MIN_BETA = 0.01
const CUDNN_BN_MIN_EPSILON = 1.0e-5

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

type cudnnLRNStruct
end

typealias cudnnLRNDescriptor_t Ptr{cudnnLRNStruct}

type cudnnActivationStruct
end

typealias cudnnActivationDescriptor_t Ptr{cudnnActivationStruct}

# begin enum cudnnDataType_t
typealias cudnnDataType_t UInt32
const CUDNN_DATA_FLOAT = (UInt32)(0)
const CUDNN_DATA_DOUBLE = (UInt32)(1)
const CUDNN_DATA_HALF = (UInt32)(2)
# end enum cudnnDataType_t

# begin enum cudnnNanPropagation_t
typealias cudnnNanPropagation_t UInt32
const CUDNN_NOT_PROPAGATE_NAN = (UInt32)(0)
const CUDNN_PROPAGATE_NAN = (UInt32)(1)
# end enum cudnnNanPropagation_t

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
const CUDNN_CONVOLUTION_FWD_ALGO_FFT = (UInt32)(4)
const CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = (UInt32)(5)
# end enum cudnnConvolutionFwdAlgo_t

type cudnnConvolutionFwdAlgoPerf_t
    algo::cudnnConvolutionFwdAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Cint
end

# begin enum cudnnConvolutionBwdFilterPreference_t
typealias cudnnConvolutionBwdFilterPreference_t UInt32
const CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum cudnnConvolutionBwdFilterPreference_t

# begin enum cudnnConvolutionBwdFilterAlgo_t
typealias cudnnConvolutionBwdFilterAlgo_t UInt32
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = (UInt32)(2)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = (UInt32)(3)
# end enum cudnnConvolutionBwdFilterAlgo_t

type cudnnConvolutionBwdFilterAlgoPerf_t
    algo::cudnnConvolutionBwdFilterAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Cint
end

# begin enum cudnnConvolutionBwdDataPreference_t
typealias cudnnConvolutionBwdDataPreference_t UInt32
const CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum cudnnConvolutionBwdDataPreference_t

# begin enum cudnnConvolutionBwdDataAlgo_t
typealias cudnnConvolutionBwdDataAlgo_t UInt32
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = (UInt32)(2)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = (UInt32)(3)
# end enum cudnnConvolutionBwdDataAlgo_t

type cudnnConvolutionBwdDataAlgoPerf_t
    algo::cudnnConvolutionBwdDataAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Cint
end

# begin enum cudnnSoftmaxAlgorithm_t
typealias cudnnSoftmaxAlgorithm_t UInt32
const CUDNN_SOFTMAX_FAST = (UInt32)(0)
const CUDNN_SOFTMAX_ACCURATE = (UInt32)(1)
const CUDNN_SOFTMAX_LOG = (UInt32)(2)
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
const CUDNN_ACTIVATION_CLIPPED_RELU = (UInt32)(3)
# end enum cudnnActivationMode_t

# begin enum cudnnLRNMode_t
typealias cudnnLRNMode_t UInt32
const CUDNN_LRN_CROSS_CHANNEL_DIM1 = (UInt32)(0)
# end enum cudnnLRNMode_t

# begin enum cudnnDivNormMode_t
typealias cudnnDivNormMode_t UInt32
const CUDNN_DIVNORM_PRECOMPUTED_MEANS = (UInt32)(0)
# end enum cudnnDivNormMode_t

# begin enum cudnnBatchNormMode_t
typealias cudnnBatchNormMode_t UInt32
const CUDNN_BATCHNORM_PER_ACTIVATION = (UInt32)(0)
const CUDNN_BATCHNORM_SPATIAL = (UInt32)(1)
# end enum cudnnBatchNormMode_t
