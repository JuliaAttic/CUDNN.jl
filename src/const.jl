

## softmax

# begin enum cudnnSoftmaxAlgorithm_t
const cudnnSoftmaxAlgorithm_t = UInt32
const CUDNN_SOFTMAX_FAST = (UInt32)(0)
const CUDNN_SOFTMAX_ACCURATE = (UInt32)(1)
const CUDNN_SOFTMAX_LOG = (UInt32)(2)
# end enum cudnnSoftmaxAlgorithm_t

# begin enum cudnnSoftmaxMode_t
const cudnnSoftmaxMode_t = UInt32
const CUDNN_SOFTMAX_MODE_INSTANCE = (UInt32)(0)
const CUDNN_SOFTMAX_MODE_CHANNEL = (UInt32)(1)
