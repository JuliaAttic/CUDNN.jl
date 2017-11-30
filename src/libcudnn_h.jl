# Automatically generated using Clang.jl wrap_c, version 0.0.0

using Compat

const unix = 1
const linux = 1
const CUDNN_MAJOR = 5
const CUDNN_MINOR = 1
const CUDNN_PATCHLEVEL = 10

# Skipping MacroDefinition: CUDNN_VERSION ( CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL ) #

const MB_LEN_MAX = 16

# Skipping MacroDefinition: LLONG_MIN ( - LLONG_MAX - 1 ) #

# const LLONG_MAX = __LONG_LONG_MAX__

# Skipping MacroDefinition: ULLONG_MAX ( LLONG_MAX * 2ULL + 1 ) #

const NR_OPEN = 1024
const NGROUPS_MAX = 65536
const ARG_MAX = 131072
const LINK_MAX = 127
const MAX_CANON = 255
const MAX_INPUT = 255
const NAME_MAX = 255
const PATH_MAX = 4096
const PIPE_BUF = 4096
const XATTR_NAME_MAX = 255
const XATTR_SIZE_MAX = 65536
const XATTR_LIST_MAX = 65536
const RTSIG_MAX = 32
const PTHREAD_KEYS_MAX = 1024
# const PTHREAD_DESTRUCTOR_ITERATIONS = _POSIX_THREAD_DESTRUCTOR_ITERATIONS
const AIO_PRIO_DELTA_MAX = 20
const PTHREAD_STACK_MIN = 16384
const DELAYTIMER_MAX = 2147483647
const TTY_NAME_MAX = 32
const LOGIN_NAME_MAX = 256
const HOST_NAME_MAX = 64
const MQ_PRIO_MAX = 32768
const SEM_VALUE_MAX = 2147483647
# const SSIZE_MAX = LONG_MAX
# const BC_BASE_MAX = _POSIX2_BC_BASE_MAX
# const BC_DIM_MAX = _POSIX2_BC_DIM_MAX
# const BC_SCALE_MAX = _POSIX2_BC_SCALE_MAX
# const BC_STRING_MAX = _POSIX2_BC_STRING_MAX
const COLL_WEIGHTS_MAX = 255
# const EXPR_NEST_MAX = _POSIX2_EXPR_NEST_MAX
#  const LINE_MAX = _POSIX2_LINE_MAX
const CHARCLASS_NAME_MAX = 2048

# Skipping MacroDefinition: RE_DUP_MAX ( 0x7fff ) #

const cudaHostAllocDefault = 0x00
const cudaHostAllocPortable = 0x01
const cudaHostAllocMapped = 0x02
const cudaHostAllocWriteCombined = 0x04
const cudaHostRegisterDefault = 0x00
const cudaHostRegisterPortable = 0x01
const cudaHostRegisterMapped = 0x02
const cudaHostRegisterIoMemory = 0x04
const cudaPeerAccessDefault = 0x00
const cudaStreamDefault = 0x00
const cudaStreamNonBlocking = 0x01

# Skipping MacroDefinition: cudaStreamLegacy ( ( cudaStream_t ) 0x1 ) /**
# * Per-thread stream handle
# *
# * Stream handle that can be passed as a cudaStream_t to use an implicit stream
# * with per-thread synchronization behavior.
# *
# * See details of the \link_sync_behavior
# */
# Skipping MacroDefinition: cudaStreamPerThread ( ( cudaStream_t ) 0x2 ) #

const cudaEventDefault = 0x00
const cudaEventBlockingSync = 0x01
const cudaEventDisableTiming = 0x02
const cudaEventInterprocess = 0x04
const cudaDeviceScheduleAuto = 0x00
const cudaDeviceScheduleSpin = 0x01
const cudaDeviceScheduleYield = 0x02
const cudaDeviceScheduleBlockingSync = 0x04
const cudaDeviceBlockingSync = 0x04
const cudaDeviceScheduleMask = 0x07
const cudaDeviceMapHost = 0x08
const cudaDeviceLmemResizeToMax = 0x10
const cudaDeviceMask = Float32(0x01)
const cudaArrayDefault = 0x00
const cudaArrayLayered = 0x01
const cudaArraySurfaceLoadStore = 0x02
const cudaArrayCubemap = 0x04
const cudaArrayTextureGather = 0x08
const cudaIpcMemLazyEnablePeerAccess = 0x01
const cudaMemAttachGlobal = 0x01
const cudaMemAttachHost = 0x02
const cudaMemAttachSingle = 0x04
const cudaOccupancyDefault = 0x00
const cudaOccupancyDisableCachingOverride = 0x01

# Skipping MacroDefinition: cudaCpuDeviceId ( ( int ) - 1 ) /**< Device id that represents the CPU */
# Skipping MacroDefinition: cudaInvalidDeviceId ( ( int ) - 2 ) /**< Device id that represents an invalid device */
# Skipping MacroDefinition: cudaDevicePropDontCare { { '\0' } , /* char   name[256];               */ 0 , /* size_t totalGlobalMem;          */ 0 , /* size_t sharedMemPerBlock;       */ 0 , /* int    regsPerBlock;            */ 0 , /* int    warpSize;                */ 0 , /* size_t memPitch;                */ 0 , /* int    maxThreadsPerBlock;      */ { 0 , 0 , 0 } , /* int    maxThreadsDim[3];        */ { 0 , 0 , 0 } , /* int    maxGridSize[3];          */ 0 , /* int    clockRate;               */ 0 , /* size_t totalConstMem;           */ - 1 , /* int    major;                   */ - 1 , /* int    minor;                   */ 0 , /* size_t textureAlignment;        */ 0 , /* size_t texturePitchAlignment    */ - 1 , /* int    deviceOverlap;           */ 0 , /* int    multiProcessorCount;     */ 0 , /* int    kernelExecTimeoutEnabled */ 0 , /* int    integrated               */ 0 , /* int    canMapHostMemory         */ 0 , /* int    computeMode              */ 0 , /* int    maxTexture1D             */ 0 , /* int    maxTexture1DMipmap       */ 0 , /* int    maxTexture1DLinear       */ { 0 , 0 } , /* int    maxTexture2D[2]          */ { 0 , 0 } , /* int    maxTexture2DMipmap[2]    */ { 0 , 0 , 0 } , /* int    maxTexture2DLinear[3]    */ { 0 , 0 } , /* int    maxTexture2DGather[2]    */ { 0 , 0 , 0 } , /* int    maxTexture3D[3]          */ { 0 , 0 , 0 } , /* int    maxTexture3DAlt[3]       */ 0 , /* int    maxTextureCubemap        */ { 0 , 0 } , /* int    maxTexture1DLayered[2]   */ { 0 , 0 , 0 } , /* int    maxTexture2DLayered[3]   */ { 0 , 0 } , /* int    maxTextureCubemapLayered[2] */ 0 , /* int    maxSurface1D             */ { 0 , 0 } , /* int    maxSurface2D[2]          */ { 0 , 0 , 0 } , /* int    maxSurface3D[3]          */ { 0 , 0 } , /* int    maxSurface1DLayered[2]   */ { 0 , 0 , 0 } , /* int    maxSurface2DLayered[3]   */ 0 , /* int    maxSurfaceCubemap        */ { 0 , 0 } , /* int    maxSurfaceCubemapLayered[2] */ 0 , /* size_t surfaceAlignment         */ 0 , /* int    concurrentKernels        */ 0 , /* int    ECCEnabled               */ 0 , /* int    pciBusID                 */ 0 , /* int    pciDeviceID              */ 0 , /* int    pciDomainID              */ 0 , /* int    tccDriver                */ 0 , /* int    asyncEngineCount         */ 0 , /* int    unifiedAddressing        */ 0 , /* int    memoryClockRate          */ 0 , /* int    memoryBusWidth           */ 0 , /* int    l2CacheSize              */ 0 , /* int    maxThreadsPerMultiProcessor */ 0 , /* int    streamPrioritiesSupported */ 0 , /* int    globalL1CacheSupported   */ 0 , /* int    localL1CacheSupported    */ 0 , /* size_t sharedMemPerMultiprocessor; */ 0 , /* int    regsPerMultiprocessor;   */ 0 , /* int    managedMemory            */ 0 , /* int    isMultiGpuBoard          */ 0 , /* int    multiGpuBoardGroupID     */ 0 , /* int    hostNativeAtomicSupported */ 0 , /* int    singleToDoublePrecisionPerfRatio */ 0 , /* int    pageableMemoryAccess     */ 0 , /* int    concurrentManagedAccess  */ } /**< Empty device properties */

const CUDA_IPC_HANDLE_SIZE = 64
const cudaSurfaceType1D = 0x01
const cudaSurfaceType2D = 0x02
const cudaSurfaceType3D = 0x03
const cudaSurfaceTypeCubemap = 0x0c
const cudaSurfaceType1DLayered = 0xf1
const cudaSurfaceType2DLayered = 0xf2
const cudaSurfaceTypeCubemapLayered = 0xfc
const cudaTextureType1D = 0x01
const cudaTextureType2D = 0x02
const cudaTextureType3D = 0x03
const cudaTextureTypeCubemap = 0x0c
const cudaTextureType1DLayered = 0xf1
const cudaTextureType2DLayered = 0xf2
const cudaTextureTypeCubemapLayered = 0xfc
const CUDART_VERSION = 8000
# const CUDART_DEVICE = __device__
const CUDNN_DIM_MAX = 8
const CUDNN_LRN_MIN_N = 1
const CUDNN_LRN_MAX_N = 16
const CUDNN_LRN_MIN_K = 1.0e-5
const CUDNN_LRN_MIN_BETA = 0.01
const CUDNN_BN_MIN_EPSILON = 1.0e-5

# begin enum cudaError
const cudaError = UInt32
const cudaSuccess = (UInt32)(0)
const cudaErrorMissingConfiguration = (UInt32)(1)
const cudaErrorMemoryAllocation = (UInt32)(2)
const cudaErrorInitializationError = (UInt32)(3)
const cudaErrorLaunchFailure = (UInt32)(4)
const cudaErrorPriorLaunchFailure = (UInt32)(5)
const cudaErrorLaunchTimeout = (UInt32)(6)
const cudaErrorLaunchOutOfResources = (UInt32)(7)
const cudaErrorInvalidDeviceFunction = (UInt32)(8)
const cudaErrorInvalidConfiguration = (UInt32)(9)
const cudaErrorInvalidDevice = (UInt32)(10)
const cudaErrorInvalidValue = (UInt32)(11)
const cudaErrorInvalidPitchValue = (UInt32)(12)
const cudaErrorInvalidSymbol = (UInt32)(13)
const cudaErrorMapBufferObjectFailed = (UInt32)(14)
const cudaErrorUnmapBufferObjectFailed = (UInt32)(15)
const cudaErrorInvalidHostPointer = (UInt32)(16)
const cudaErrorInvalidDevicePointer = (UInt32)(17)
const cudaErrorInvalidTexture = (UInt32)(18)
const cudaErrorInvalidTextureBinding = (UInt32)(19)
const cudaErrorInvalidChannelDescriptor = (UInt32)(20)
const cudaErrorInvalidMemcpyDirection = (UInt32)(21)
const cudaErrorAddressOfConstant = (UInt32)(22)
const cudaErrorTextureFetchFailed = (UInt32)(23)
const cudaErrorTextureNotBound = (UInt32)(24)
const cudaErrorSynchronizationError = (UInt32)(25)
const cudaErrorInvalidFilterSetting = (UInt32)(26)
const cudaErrorInvalidNormSetting = (UInt32)(27)
const cudaErrorMixedDeviceExecution = (UInt32)(28)
const cudaErrorCudartUnloading = (UInt32)(29)
const cudaErrorUnknown = (UInt32)(30)
const cudaErrorNotYetImplemented = (UInt32)(31)
const cudaErrorMemoryValueTooLarge = (UInt32)(32)
const cudaErrorInvalidResourceHandle = (UInt32)(33)
const cudaErrorNotReady = (UInt32)(34)
const cudaErrorInsufficientDriver = (UInt32)(35)
const cudaErrorSetOnActiveProcess = (UInt32)(36)
const cudaErrorInvalidSurface = (UInt32)(37)
const cudaErrorNoDevice = (UInt32)(38)
const cudaErrorECCUncorrectable = (UInt32)(39)
const cudaErrorSharedObjectSymbolNotFound = (UInt32)(40)
const cudaErrorSharedObjectInitFailed = (UInt32)(41)
const cudaErrorUnsupportedLimit = (UInt32)(42)
const cudaErrorDuplicateVariableName = (UInt32)(43)
const cudaErrorDuplicateTextureName = (UInt32)(44)
const cudaErrorDuplicateSurfaceName = (UInt32)(45)
const cudaErrorDevicesUnavailable = (UInt32)(46)
const cudaErrorInvalidKernelImage = (UInt32)(47)
const cudaErrorNoKernelImageForDevice = (UInt32)(48)
const cudaErrorIncompatibleDriverContext = (UInt32)(49)
const cudaErrorPeerAccessAlreadyEnabled = (UInt32)(50)
const cudaErrorPeerAccessNotEnabled = (UInt32)(51)
const cudaErrorDeviceAlreadyInUse = (UInt32)(54)
const cudaErrorProfilerDisabled = (UInt32)(55)
const cudaErrorProfilerNotInitialized = (UInt32)(56)
const cudaErrorProfilerAlreadyStarted = (UInt32)(57)
const cudaErrorProfilerAlreadyStopped = (UInt32)(58)
const cudaErrorAssert = (UInt32)(59)
const cudaErrorTooManyPeers = (UInt32)(60)
const cudaErrorHostMemoryAlreadyRegistered = (UInt32)(61)
const cudaErrorHostMemoryNotRegistered = (UInt32)(62)
const cudaErrorOperatingSystem = (UInt32)(63)
const cudaErrorPeerAccessUnsupported = (UInt32)(64)
const cudaErrorLaunchMaxDepthExceeded = (UInt32)(65)
const cudaErrorLaunchFileScopedTex = (UInt32)(66)
const cudaErrorLaunchFileScopedSurf = (UInt32)(67)
const cudaErrorSyncDepthExceeded = (UInt32)(68)
const cudaErrorLaunchPendingCountExceeded = (UInt32)(69)
const cudaErrorNotPermitted = (UInt32)(70)
const cudaErrorNotSupported = (UInt32)(71)
const cudaErrorHardwareStackError = (UInt32)(72)
const cudaErrorIllegalInstruction = (UInt32)(73)
const cudaErrorMisalignedAddress = (UInt32)(74)
const cudaErrorInvalidAddressSpace = (UInt32)(75)
const cudaErrorInvalidPc = (UInt32)(76)
const cudaErrorIllegalAddress = (UInt32)(77)
const cudaErrorInvalidPtx = (UInt32)(78)
const cudaErrorInvalidGraphicsContext = (UInt32)(79)
const cudaErrorNvlinkUncorrectable = (UInt32)(80)
const cudaErrorStartupFailure = (UInt32)(127)
const cudaErrorApiFailureBase = (UInt32)(10000)
# end enum cudaError

# begin enum cudaChannelFormatKind
const cudaChannelFormatKind = UInt32
const cudaChannelFormatKindSigned = (UInt32)(0)
const cudaChannelFormatKindUnsigned = (UInt32)(1)
const cudaChannelFormatKindFloat = (UInt32)(2)
const cudaChannelFormatKindNone = (UInt32)(3)
# end enum cudaChannelFormatKind

mutable struct cudaChannelFormatDesc
    x::Cint
    y::Cint
    z::Cint
    w::Cint
    f::cudaChannelFormatKind
end

mutable struct cudaArray
end

const cudaArray_t = Ptr{cudaArray}
const cudaArray_const_t = Ptr{cudaArray}

mutable struct cudaMipmappedArray
end

const cudaMipmappedArray_t = Ptr{cudaMipmappedArray}
const cudaMipmappedArray_const_t = Ptr{cudaMipmappedArray}

# begin enum cudaMemoryType
const cudaMemoryType = UInt32
const cudaMemoryTypeHost = (UInt32)(1)
const cudaMemoryTypeDevice = (UInt32)(2)
# end enum cudaMemoryType

# begin enum cudaMemcpyKind
const cudaMemcpyKind = UInt32
const cudaMemcpyHostToHost = (UInt32)(0)
const cudaMemcpyHostToDevice = (UInt32)(1)
const cudaMemcpyDeviceToHost = (UInt32)(2)
const cudaMemcpyDeviceToDevice = (UInt32)(3)
const cudaMemcpyDefault = (UInt32)(4)
# end enum cudaMemcpyKind

mutable struct cudaPitchedPtr
    ptr::Ptr{Void}
    pitch::Cint
    xsize::Cint
    ysize::Cint
end

mutable struct cudaExtent
    width::Cint
    height::Cint
    depth::Cint
end

mutable struct cudaPos
    x::Cint
    y::Cint
    z::Cint
end

mutable struct cudaMemcpy3DParms
    srcArray::cudaArray_t
    srcPos::cudaPos
    srcPtr::cudaPitchedPtr
    dstArray::cudaArray_t
    dstPos::cudaPos
    dstPtr::cudaPitchedPtr
    extent::cudaExtent
    kind::cudaMemcpyKind
end

mutable struct cudaMemcpy3DPeerParms
    srcArray::cudaArray_t
    srcPos::cudaPos
    srcPtr::cudaPitchedPtr
    srcDevice::Cint
    dstArray::cudaArray_t
    dstPos::cudaPos
    dstPtr::cudaPitchedPtr
    dstDevice::Cint
    extent::cudaExtent
end

mutable struct cudaGraphicsResource
end

# begin enum cudaGraphicsRegisterFlags
const cudaGraphicsRegisterFlags = UInt32
const cudaGraphicsRegisterFlagsNone = (UInt32)(0)
const cudaGraphicsRegisterFlagsReadOnly = (UInt32)(1)
const cudaGraphicsRegisterFlagsWriteDiscard = (UInt32)(2)
const cudaGraphicsRegisterFlagsSurfaceLoadStore = (UInt32)(4)
const cudaGraphicsRegisterFlagsTextureGather = (UInt32)(8)
# end enum cudaGraphicsRegisterFlags

# begin enum cudaGraphicsMapFlags
const cudaGraphicsMapFlags = UInt32
const cudaGraphicsMapFlagsNone = (UInt32)(0)
const cudaGraphicsMapFlagsReadOnly = (UInt32)(1)
const cudaGraphicsMapFlagsWriteDiscard = (UInt32)(2)
# end enum cudaGraphicsMapFlags

# begin enum cudaGraphicsCubeFace
const cudaGraphicsCubeFace = UInt32
const cudaGraphicsCubeFacePositiveX = (UInt32)(0)
const cudaGraphicsCubeFaceNegativeX = (UInt32)(1)
const cudaGraphicsCubeFacePositiveY = (UInt32)(2)
const cudaGraphicsCubeFaceNegativeY = (UInt32)(3)
const cudaGraphicsCubeFacePositiveZ = (UInt32)(4)
const cudaGraphicsCubeFaceNegativeZ = (UInt32)(5)
# end enum cudaGraphicsCubeFace

# begin enum cudaResourceType
const cudaResourceType = UInt32
const cudaResourceTypeArray = (UInt32)(0)
const cudaResourceTypeMipmappedArray = (UInt32)(1)
const cudaResourceTypeLinear = (UInt32)(2)
const cudaResourceTypePitch2D = (UInt32)(3)
# end enum cudaResourceType

# begin enum cudaResourceViewFormat
const cudaResourceViewFormat = UInt32
const cudaResViewFormatNone = (UInt32)(0)
const cudaResViewFormatUnsignedChar1 = (UInt32)(1)
const cudaResViewFormatUnsignedChar2 = (UInt32)(2)
const cudaResViewFormatUnsignedChar4 = (UInt32)(3)
const cudaResViewFormatSignedChar1 = (UInt32)(4)
const cudaResViewFormatSignedChar2 = (UInt32)(5)
const cudaResViewFormatSignedChar4 = (UInt32)(6)
const cudaResViewFormatUnsignedShort1 = (UInt32)(7)
const cudaResViewFormatUnsignedShort2 = (UInt32)(8)
const cudaResViewFormatUnsignedShort4 = (UInt32)(9)
const cudaResViewFormatSignedShort1 = (UInt32)(10)
const cudaResViewFormatSignedShort2 = (UInt32)(11)
const cudaResViewFormatSignedShort4 = (UInt32)(12)
const cudaResViewFormatUnsignedInt1 = (UInt32)(13)
const cudaResViewFormatUnsignedInt2 = (UInt32)(14)
const cudaResViewFormatUnsignedInt4 = (UInt32)(15)
const cudaResViewFormatSignedInt1 = (UInt32)(16)
const cudaResViewFormatSignedInt2 = (UInt32)(17)
const cudaResViewFormatSignedInt4 = (UInt32)(18)
const cudaResViewFormatHalf1 = (UInt32)(19)
const cudaResViewFormatHalf2 = (UInt32)(20)
const cudaResViewFormatHalf4 = (UInt32)(21)
const cudaResViewFormatFloat1 = (UInt32)(22)
const cudaResViewFormatFloat2 = (UInt32)(23)
const cudaResViewFormatFloat4 = (UInt32)(24)
const cudaResViewFormatUnsignedBlockCompressed1 = (UInt32)(25)
const cudaResViewFormatUnsignedBlockCompressed2 = (UInt32)(26)
const cudaResViewFormatUnsignedBlockCompressed3 = (UInt32)(27)
const cudaResViewFormatUnsignedBlockCompressed4 = (UInt32)(28)
const cudaResViewFormatSignedBlockCompressed4 = (UInt32)(29)
const cudaResViewFormatUnsignedBlockCompressed5 = (UInt32)(30)
const cudaResViewFormatSignedBlockCompressed5 = (UInt32)(31)
const cudaResViewFormatUnsignedBlockCompressed6H = (UInt32)(32)
const cudaResViewFormatSignedBlockCompressed6H = (UInt32)(33)
const cudaResViewFormatUnsignedBlockCompressed7 = (UInt32)(34)
# end enum cudaResourceViewFormat

mutable struct cudaResourceDesc
    resType::cudaResourceType
    res::Void
end

mutable struct cudaResourceViewDesc
    format::cudaResourceViewFormat
    width::Cint
    height::Cint
    depth::Cint
    firstMipmapLevel::UInt32
    lastMipmapLevel::UInt32
    firstLayer::UInt32
    lastLayer::UInt32
end

mutable struct cudaPointerAttributes
    memoryType::cudaMemoryType
    device::Cint
    devicePointer::Ptr{Void}
    hostPointer::Ptr{Void}
    isManaged::Cint
end

mutable struct cudaFuncAttributes
    sharedSizeBytes::Cint
    constSizeBytes::Cint
    localSizeBytes::Cint
    maxThreadsPerBlock::Cint
    numRegs::Cint
    ptxVersion::Cint
    binaryVersion::Cint
    cacheModeCA::Cint
end

# begin enum cudaFuncCache
const cudaFuncCache = UInt32
const cudaFuncCachePreferNone = (UInt32)(0)
const cudaFuncCachePreferShared = (UInt32)(1)
const cudaFuncCachePreferL1 = (UInt32)(2)
const cudaFuncCachePreferEqual = (UInt32)(3)
# end enum cudaFuncCache

# begin enum cudaSharedMemConfig
const cudaSharedMemConfig = UInt32
const cudaSharedMemBankSizeDefault = (UInt32)(0)
const cudaSharedMemBankSizeFourByte = (UInt32)(1)
const cudaSharedMemBankSizeEightByte = (UInt32)(2)
# end enum cudaSharedMemConfig

# begin enum cudaComputeMode
const cudaComputeMode = UInt32
const cudaComputeModeDefault = (UInt32)(0)
const cudaComputeModeExclusive = (UInt32)(1)
const cudaComputeModeProhibited = (UInt32)(2)
const cudaComputeModeExclusiveProcess = (UInt32)(3)
# end enum cudaComputeMode

# begin enum cudaLimit
const cudaLimit = UInt32
const cudaLimitStackSize = (UInt32)(0)
const cudaLimitPrintfFifoSize = (UInt32)(1)
const cudaLimitMallocHeapSize = (UInt32)(2)
const cudaLimitDevRuntimeSyncDepth = (UInt32)(3)
const cudaLimitDevRuntimePendingLaunchCount = (UInt32)(4)
# end enum cudaLimit

# begin enum cudaMemoryAdvise
const cudaMemoryAdvise = UInt32
const cudaMemAdviseSetReadMostly = (UInt32)(1)
const cudaMemAdviseUnsetReadMostly = (UInt32)(2)
const cudaMemAdviseSetPreferredLocation = (UInt32)(3)
const cudaMemAdviseUnsetPreferredLocation = (UInt32)(4)
const cudaMemAdviseSetAccessedBy = (UInt32)(5)
const cudaMemAdviseUnsetAccessedBy = (UInt32)(6)
# end enum cudaMemoryAdvise

# begin enum cudaMemRangeAttribute
const cudaMemRangeAttribute = UInt32
const cudaMemRangeAttributeReadMostly = (UInt32)(1)
const cudaMemRangeAttributePreferredLocation = (UInt32)(2)
const cudaMemRangeAttributeAccessedBy = (UInt32)(3)
const cudaMemRangeAttributeLastPrefetchLocation = (UInt32)(4)
# end enum cudaMemRangeAttribute

# begin enum cudaOutputMode
const cudaOutputMode = UInt32
const cudaKeyValuePair = (UInt32)(0)
const cudaCSV = (UInt32)(1)
# end enum cudaOutputMode

# begin enum cudaDeviceAttr
const cudaDeviceAttr = UInt32
const cudaDevAttrMaxThreadsPerBlock = (UInt32)(1)
const cudaDevAttrMaxBlockDimX = (UInt32)(2)
const cudaDevAttrMaxBlockDimY = (UInt32)(3)
const cudaDevAttrMaxBlockDimZ = (UInt32)(4)
const cudaDevAttrMaxGridDimX = (UInt32)(5)
const cudaDevAttrMaxGridDimY = (UInt32)(6)
const cudaDevAttrMaxGridDimZ = (UInt32)(7)
const cudaDevAttrMaxSharedMemoryPerBlock = (UInt32)(8)
const cudaDevAttrTotalConstantMemory = (UInt32)(9)
const cudaDevAttrWarpSize = (UInt32)(10)
const cudaDevAttrMaxPitch = (UInt32)(11)
const cudaDevAttrMaxRegistersPerBlock = (UInt32)(12)
const cudaDevAttrClockRate = (UInt32)(13)
const cudaDevAttrTextureAlignment = (UInt32)(14)
const cudaDevAttrGpuOverlap = (UInt32)(15)
const cudaDevAttrMultiProcessorCount = (UInt32)(16)
const cudaDevAttrKernelExecTimeout = (UInt32)(17)
const cudaDevAttrIntegrated = (UInt32)(18)
const cudaDevAttrCanMapHostMemory = (UInt32)(19)
const cudaDevAttrComputeMode = (UInt32)(20)
const cudaDevAttrMaxTexture1DWidth = (UInt32)(21)
const cudaDevAttrMaxTexture2DWidth = (UInt32)(22)
const cudaDevAttrMaxTexture2DHeight = (UInt32)(23)
const cudaDevAttrMaxTexture3DWidth = (UInt32)(24)
const cudaDevAttrMaxTexture3DHeight = (UInt32)(25)
const cudaDevAttrMaxTexture3DDepth = (UInt32)(26)
const cudaDevAttrMaxTexture2DLayeredWidth = (UInt32)(27)
const cudaDevAttrMaxTexture2DLayeredHeight = (UInt32)(28)
const cudaDevAttrMaxTexture2DLayeredLayers = (UInt32)(29)
const cudaDevAttrSurfaceAlignment = (UInt32)(30)
const cudaDevAttrConcurrentKernels = (UInt32)(31)
const cudaDevAttrEccEnabled = (UInt32)(32)
const cudaDevAttrPciBusId = (UInt32)(33)
const cudaDevAttrPciDeviceId = (UInt32)(34)
const cudaDevAttrTccDriver = (UInt32)(35)
const cudaDevAttrMemoryClockRate = (UInt32)(36)
const cudaDevAttrGlobalMemoryBusWidth = (UInt32)(37)
const cudaDevAttrL2CacheSize = (UInt32)(38)
const cudaDevAttrMaxThreadsPerMultiProcessor = (UInt32)(39)
const cudaDevAttrAsyncEngineCount = (UInt32)(40)
const cudaDevAttrUnifiedAddressing = (UInt32)(41)
const cudaDevAttrMaxTexture1DLayeredWidth = (UInt32)(42)
const cudaDevAttrMaxTexture1DLayeredLayers = (UInt32)(43)
const cudaDevAttrMaxTexture2DGatherWidth = (UInt32)(45)
const cudaDevAttrMaxTexture2DGatherHeight = (UInt32)(46)
const cudaDevAttrMaxTexture3DWidthAlt = (UInt32)(47)
const cudaDevAttrMaxTexture3DHeightAlt = (UInt32)(48)
const cudaDevAttrMaxTexture3DDepthAlt = (UInt32)(49)
const cudaDevAttrPciDomainId = (UInt32)(50)
const cudaDevAttrTexturePitchAlignment = (UInt32)(51)
const cudaDevAttrMaxTextureCubemapWidth = (UInt32)(52)
const cudaDevAttrMaxTextureCubemapLayeredWidth = (UInt32)(53)
const cudaDevAttrMaxTextureCubemapLayeredLayers = (UInt32)(54)
const cudaDevAttrMaxSurface1DWidth = (UInt32)(55)
const cudaDevAttrMaxSurface2DWidth = (UInt32)(56)
const cudaDevAttrMaxSurface2DHeight = (UInt32)(57)
const cudaDevAttrMaxSurface3DWidth = (UInt32)(58)
const cudaDevAttrMaxSurface3DHeight = (UInt32)(59)
const cudaDevAttrMaxSurface3DDepth = (UInt32)(60)
const cudaDevAttrMaxSurface1DLayeredWidth = (UInt32)(61)
const cudaDevAttrMaxSurface1DLayeredLayers = (UInt32)(62)
const cudaDevAttrMaxSurface2DLayeredWidth = (UInt32)(63)
const cudaDevAttrMaxSurface2DLayeredHeight = (UInt32)(64)
const cudaDevAttrMaxSurface2DLayeredLayers = (UInt32)(65)
const cudaDevAttrMaxSurfaceCubemapWidth = (UInt32)(66)
const cudaDevAttrMaxSurfaceCubemapLayeredWidth = (UInt32)(67)
const cudaDevAttrMaxSurfaceCubemapLayeredLayers = (UInt32)(68)
const cudaDevAttrMaxTexture1DLinearWidth = (UInt32)(69)
const cudaDevAttrMaxTexture2DLinearWidth = (UInt32)(70)
const cudaDevAttrMaxTexture2DLinearHeight = (UInt32)(71)
const cudaDevAttrMaxTexture2DLinearPitch = (UInt32)(72)
const cudaDevAttrMaxTexture2DMipmappedWidth = (UInt32)(73)
const cudaDevAttrMaxTexture2DMipmappedHeight = (UInt32)(74)
const cudaDevAttrComputeCapabilityMajor = (UInt32)(75)
const cudaDevAttrComputeCapabilityMinor = (UInt32)(76)
const cudaDevAttrMaxTexture1DMipmappedWidth = (UInt32)(77)
const cudaDevAttrStreamPrioritiesSupported = (UInt32)(78)
const cudaDevAttrGlobalL1CacheSupported = (UInt32)(79)
const cudaDevAttrLocalL1CacheSupported = (UInt32)(80)
const cudaDevAttrMaxSharedMemoryPerMultiprocessor = (UInt32)(81)
const cudaDevAttrMaxRegistersPerMultiprocessor = (UInt32)(82)
const cudaDevAttrManagedMemory = (UInt32)(83)
const cudaDevAttrIsMultiGpuBoard = (UInt32)(84)
const cudaDevAttrMultiGpuBoardGroupID = (UInt32)(85)
const cudaDevAttrHostNativeAtomicSupported = (UInt32)(86)
const cudaDevAttrSingleToDoublePrecisionPerfRatio = (UInt32)(87)
const cudaDevAttrPageableMemoryAccess = (UInt32)(88)
const cudaDevAttrConcurrentManagedAccess = (UInt32)(89)
const cudaDevAttrComputePreemptionSupported = (UInt32)(90)
const cudaDevAttrCanUseHostPointerForRegisteredMem = (UInt32)(91)
# end enum cudaDeviceAttr

# begin enum cudaDeviceP2PAttr
const cudaDeviceP2PAttr = UInt32
const cudaDevP2PAttrPerformanceRank = (UInt32)(1)
const cudaDevP2PAttrAccessSupported = (UInt32)(2)
const cudaDevP2PAttrNativeAtomicSupported = (UInt32)(3)
# end enum cudaDeviceP2PAttr

mutable struct cudaDeviceProp
    name::NTuple{256, UInt8}
    totalGlobalMem::Cint
    sharedMemPerBlock::Cint
    regsPerBlock::Cint
    warpSize::Cint
    memPitch::Cint
    maxThreadsPerBlock::Cint
    maxThreadsDim::NTuple{3, Cint}
    maxGridSize::NTuple{3, Cint}
    clockRate::Cint
    totalConstMem::Cint
    major::Cint
    minor::Cint
    textureAlignment::Cint
    texturePitchAlignment::Cint
    deviceOverlap::Cint
    multiProcessorCount::Cint
    kernelExecTimeoutEnabled::Cint
    integrated::Cint
    canMapHostMemory::Cint
    computeMode::Cint
    maxTexture1D::Cint
    maxTexture1DMipmap::Cint
    maxTexture1DLinear::Cint
    maxTexture2D::NTuple{2, Cint}
    maxTexture2DMipmap::NTuple{2, Cint}
    maxTexture2DLinear::NTuple{3, Cint}
    maxTexture2DGather::NTuple{2, Cint}
    maxTexture3D::NTuple{3, Cint}
    maxTexture3DAlt::NTuple{3, Cint}
    maxTextureCubemap::Cint
    maxTexture1DLayered::NTuple{2, Cint}
    maxTexture2DLayered::NTuple{3, Cint}
    maxTextureCubemapLayered::NTuple{2, Cint}
    maxSurface1D::Cint
    maxSurface2D::NTuple{2, Cint}
    maxSurface3D::NTuple{3, Cint}
    maxSurface1DLayered::NTuple{2, Cint}
    maxSurface2DLayered::NTuple{3, Cint}
    maxSurfaceCubemap::Cint
    maxSurfaceCubemapLayered::NTuple{2, Cint}
    surfaceAlignment::Cint
    concurrentKernels::Cint
    ECCEnabled::Cint
    pciBusID::Cint
    pciDeviceID::Cint
    pciDomainID::Cint
    tccDriver::Cint
    asyncEngineCount::Cint
    unifiedAddressing::Cint
    memoryClockRate::Cint
    memoryBusWidth::Cint
    l2CacheSize::Cint
    maxThreadsPerMultiProcessor::Cint
    streamPrioritiesSupported::Cint
    globalL1CacheSupported::Cint
    localL1CacheSupported::Cint
    sharedMemPerMultiprocessor::Cint
    regsPerMultiprocessor::Cint
    managedMemory::Cint
    isMultiGpuBoard::Cint
    multiGpuBoardGroupID::Cint
    hostNativeAtomicSupported::Cint
    singleToDoublePrecisionPerfRatio::Cint
    pageableMemoryAccess::Cint
    concurrentManagedAccess::Cint
end

mutable struct cudaIpcEventHandle_st
    reserved::NTuple{64, UInt8}
end

mutable struct cudaIpcEventHandle_t
    reserved::NTuple{64, UInt8}
end

mutable struct cudaIpcMemHandle_st
    reserved::NTuple{64, UInt8}
end

mutable struct cudaIpcMemHandle_t
    reserved::NTuple{64, UInt8}
end

const cudaError_t = cudaError
# $(Expr(:typealias, :cudaError_t, :cudaError))

mutable struct CUstream_st
end

const cudaStream_t = Ptr{CUstream_st}
# $(Expr(:typealias, :cudaStream_t, :(Ptr{CUstream_st})))

mutable struct CUevent_st
end

const cudaEvent_t = Ptr{CUevent_st}
const cudaGraphicsResource_t = Ptr{cudaGraphicsResource}
# $(Expr(:typealias, :cudaEvent_t, :(Ptr{CUevent_st})))
# $(Expr(:typealias, :cudaGraphicsResource_t, :(Ptr{cudaGraphicsResource})))

mutable struct CUuuid_st
end

const cudaUUID_t = Void
const cudaOutputMode_t = cudaOutputMode
# $(Expr(:typealias, :cudaUUID_t, :Void))
# $(Expr(:typealias, :cudaOutputMode_t, :cudaOutputMode))

# begin enum cudaRoundMode
const cudaRoundMode = UInt32
const cudaRoundNearest = (UInt32)(0)
const cudaRoundZero = (UInt32)(1)
const cudaRoundPosInf = (UInt32)(2)
const cudaRoundMinInf = (UInt32)(3)
# end enum cudaRoundMode

# begin enum cudaSurfaceBoundaryMode
const cudaSurfaceBoundaryMode = UInt32
const cudaBoundaryModeZero = (UInt32)(0)
const cudaBoundaryModeClamp = (UInt32)(1)
const cudaBoundaryModeTrap = (UInt32)(2)
# end enum cudaSurfaceBoundaryMode

# begin enum cudaSurfaceFormatMode
const cudaSurfaceFormatMode = UInt32
const cudaFormatModeForced = (UInt32)(0)
const cudaFormatModeAuto = (UInt32)(1)
# end enum cudaSurfaceFormatMode

mutable struct surfaceReference
    channelDesc::cudaChannelFormatDesc
end

const cudaSurfaceObject_t = Culonglong
# $(Expr(:typealias, :cudaSurfaceObject_t, :Culonglong))

# begin enum cudaTextureAddressMode
const cudaTextureAddressMode = UInt32
const cudaAddressModeWrap = (UInt32)(0)
const cudaAddressModeClamp = (UInt32)(1)
const cudaAddressModeMirror = (UInt32)(2)
const cudaAddressModeBorder = (UInt32)(3)
# end enum cudaTextureAddressMode

# begin enum cudaTextureFilterMode
const cudaTextureFilterMode = UInt32
const cudaFilterModePoint = (UInt32)(0)
const cudaFilterModeLinear = (UInt32)(1)
# end enum cudaTextureFilterMode

# begin enum cudaTextureReadMode
const cudaTextureReadMode = UInt32
const cudaReadModeElementType = (UInt32)(0)
const cudaReadModeNormalizedFloat = (UInt32)(1)
# end enum cudaTextureReadMode

mutable struct textureReference
    normalized::Cint
    filterMode::cudaTextureFilterMode
    addressMode::NTuple{3, cudaTextureAddressMode}
    channelDesc::cudaChannelFormatDesc
    sRGB::Cint
    maxAnisotropy::UInt32
    mipmapFilterMode::cudaTextureFilterMode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
    __cudaReserved::NTuple{15, Cint}
end

mutable struct cudaTextureDesc
    addressMode::NTuple{3, cudaTextureAddressMode}
    filterMode::cudaTextureFilterMode
    readMode::cudaTextureReadMode
    sRGB::Cint
    borderColor::NTuple{4, Cfloat}
    normalizedCoords::Cint
    maxAnisotropy::UInt32
    mipmapFilterMode::cudaTextureFilterMode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
end

const cudaTextureObject_t = Culonglong
# $(Expr(:typealias, :cudaTextureObject_t, :Culonglong))

mutable struct char1
    x::UInt8
end

mutable struct uchar1
    x::Cuchar
end

mutable struct char3
    x::UInt8
    y::UInt8
    z::UInt8
end

mutable struct uchar3
    x::Cuchar
    y::Cuchar
    z::Cuchar
end

mutable struct short1
    x::Int16
end

mutable struct ushort1
    x::UInt16
end

mutable struct short3
    x::Int16
    y::Int16
    z::Int16
end

mutable struct ushort3
    x::UInt16
    y::UInt16
    z::UInt16
end

mutable struct int1
    x::Cint
end

mutable struct uint1
    x::UInt32
end

mutable struct int3
    x::Cint
    y::Cint
    z::Cint
end

mutable struct uint3
    x::UInt32
    y::UInt32
    z::UInt32
end

mutable struct long1
    x::Clong
end

mutable struct ulong1
    x::Culong
end

mutable struct long3
    x::Clong
    y::Clong
    z::Clong
end

mutable struct ulong3
    x::Culong
    y::Culong
    z::Culong
end

mutable struct float1
    x::Cfloat
end

mutable struct float3
    x::Cfloat
    y::Cfloat
    z::Cfloat
end

mutable struct longlong1
    x::Clonglong
end

mutable struct ulonglong1
    x::Culonglong
end

mutable struct longlong3
    x::Clonglong
    y::Clonglong
    z::Clonglong
end

mutable struct ulonglong3
    x::Culonglong
    y::Culonglong
    z::Culonglong
end

mutable struct double1
    x::Cdouble
end

mutable struct double3
    x::Cdouble
    y::Cdouble
    z::Cdouble
end

mutable struct dim3
    x::UInt32
    y::UInt32
    z::UInt32
end

# begin enum cudaDataType_t
const cudaDataType_t = UInt32
const CUDA_R_16F = (UInt32)(2)
const CUDA_C_16F = (UInt32)(6)
const CUDA_R_32F = (UInt32)(0)
const CUDA_C_32F = (UInt32)(4)
const CUDA_R_64F = (UInt32)(1)
const CUDA_C_64F = (UInt32)(5)
const CUDA_R_8I = (UInt32)(3)
const CUDA_C_8I = (UInt32)(7)
const CUDA_R_8U = (UInt32)(8)
const CUDA_C_8U = (UInt32)(9)
const CUDA_R_32I = (UInt32)(10)
const CUDA_C_32I = (UInt32)(11)
const CUDA_R_32U = (UInt32)(12)
const CUDA_C_32U = (UInt32)(13)
# end enum cudaDataType_t

# begin enum cudaDataType
const cudaDataType = UInt32
const CUDA_R_16F = (UInt32)(2)
const CUDA_C_16F = (UInt32)(6)
const CUDA_R_32F = (UInt32)(0)
const CUDA_C_32F = (UInt32)(4)
const CUDA_R_64F = (UInt32)(1)
const CUDA_C_64F = (UInt32)(5)
const CUDA_R_8I = (UInt32)(3)
const CUDA_C_8I = (UInt32)(7)
const CUDA_R_8U = (UInt32)(8)
const CUDA_C_8U = (UInt32)(9)
const CUDA_R_32I = (UInt32)(10)
const CUDA_C_32I = (UInt32)(11)
const CUDA_R_32U = (UInt32)(12)
const CUDA_C_32U = (UInt32)(13)
# end enum cudaDataType

# begin enum libraryPropertyType_t
const libraryPropertyType_t = UInt32
const MAJOR_VERSION = (UInt32)(0)
const MINOR_VERSION = (UInt32)(1)
const PATCH_LEVEL = (UInt32)(2)
# end enum libraryPropertyType_t

# begin enum libraryPropertyType
const libraryPropertyType = UInt32
const MAJOR_VERSION = (UInt32)(0)
const MINOR_VERSION = (UInt32)(1)
const PATCH_LEVEL = (UInt32)(2)
# end enum libraryPropertyType

const cudaStreamCallback_t = Ptr{Void}
# $(Expr(:typealias, :cudaStreamCallback_t, :(Ptr{Void})))

mutable struct cudnnContext
end

const cudnnHandle_t = Ptr{cudnnContext}
# $(Expr(:typealias, :cudnnHandle_t, :(Ptr{cudnnContext})))

# begin enum ANONYMOUS_1
const ANONYMOUS_1 = UInt32
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
# end enum ANONYMOUS_1

# begin enum cudnnStatus_t
const cudnnStatus_t = UInt32
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

mutable struct cudnnTensorStruct
end

const cudnnTensorDescriptor_t = Ptr{cudnnTensorStruct}
# $(Expr(:typealias, :cudnnTensorDescriptor_t, :(Ptr{cudnnTensorStruct})))

mutable struct cudnnConvolutionStruct
end

const cudnnConvolutionDescriptor_t = Ptr{cudnnConvolutionStruct}
# $(Expr(:typealias, :cudnnConvolutionDescriptor_t, :(Ptr{cudnnConvolutionStruct})))

mutable struct cudnnPoolingStruct
end

const cudnnPoolingDescriptor_t = Ptr{cudnnPoolingStruct}
# $(Expr(:typealias, :cudnnPoolingDescriptor_t, :(Ptr{cudnnPoolingStruct})))

mutable struct cudnnFilterStruct
end

const cudnnFilterDescriptor_t = Ptr{cudnnFilterStruct}
# $(Expr(:typealias, :cudnnFilterDescriptor_t, :(Ptr{cudnnFilterStruct})))

mutable struct cudnnLRNStruct
end

const cudnnLRNDescriptor_t = Ptr{cudnnLRNStruct}
# $(Expr(:typealias, :cudnnLRNDescriptor_t, :(Ptr{cudnnLRNStruct})))

mutable struct cudnnActivationStruct
end

const cudnnActivationDescriptor_t = Ptr{cudnnActivationStruct}
# $(Expr(:typealias, :cudnnActivationDescriptor_t, :(Ptr{cudnnActivationStruct})))

mutable struct cudnnSpatialTransformerStruct
end

const cudnnSpatialTransformerDescriptor_t = Ptr{cudnnSpatialTransformerStruct}
# $(Expr(:typealias, :cudnnSpatialTransformerDescriptor_t, :(Ptr{cudnnSpatialTransformerStruct})))

mutable struct cudnnOpTensorStruct
end

const cudnnOpTensorDescriptor_t = Ptr{cudnnOpTensorStruct}
# $(Expr(:typealias, :cudnnOpTensorDescriptor_t, :(Ptr{cudnnOpTensorStruct})))

# begin enum ANONYMOUS_2
const ANONYMOUS_2 = UInt32
const CUDNN_DATA_FLOAT = (UInt32)(0)
const CUDNN_DATA_DOUBLE = (UInt32)(1)
const CUDNN_DATA_HALF = (UInt32)(2)
# end enum ANONYMOUS_2

# begin enum cudnnDataType_t
const cudnnDataType_t = UInt32
const CUDNN_DATA_FLOAT = (UInt32)(0)
const CUDNN_DATA_DOUBLE = (UInt32)(1)
const CUDNN_DATA_HALF = (UInt32)(2)
# end enum cudnnDataType_t

# begin enum ANONYMOUS_3
const ANONYMOUS_3 = UInt32
const CUDNN_NOT_PROPAGATE_NAN = (UInt32)(0)
const CUDNN_PROPAGATE_NAN = (UInt32)(1)
# end enum ANONYMOUS_3

# begin enum cudnnNanPropagation_t
const cudnnNanPropagation_t = UInt32
const CUDNN_NOT_PROPAGATE_NAN = (UInt32)(0)
const CUDNN_PROPAGATE_NAN = (UInt32)(1)
# end enum cudnnNanPropagation_t

# begin enum ANONYMOUS_4
const ANONYMOUS_4 = UInt32
const CUDNN_TENSOR_NCHW = (UInt32)(0)
const CUDNN_TENSOR_NHWC = (UInt32)(1)
# end enum ANONYMOUS_4

# begin enum cudnnTensorFormat_t
const cudnnTensorFormat_t = UInt32
const CUDNN_TENSOR_NCHW = (UInt32)(0)
const CUDNN_TENSOR_NHWC = (UInt32)(1)
# end enum cudnnTensorFormat_t

# begin enum ANONYMOUS_5
const ANONYMOUS_5 = UInt32
const CUDNN_OP_TENSOR_ADD = (UInt32)(0)
const CUDNN_OP_TENSOR_MUL = (UInt32)(1)
const CUDNN_OP_TENSOR_MIN = (UInt32)(2)
const CUDNN_OP_TENSOR_MAX = (UInt32)(3)
# end enum ANONYMOUS_5

# begin enum cudnnOpTensorOp_t
const cudnnOpTensorOp_t = UInt32
const CUDNN_OP_TENSOR_ADD = (UInt32)(0)
const CUDNN_OP_TENSOR_MUL = (UInt32)(1)
const CUDNN_OP_TENSOR_MIN = (UInt32)(2)
const CUDNN_OP_TENSOR_MAX = (UInt32)(3)
# end enum cudnnOpTensorOp_t

# begin enum ANONYMOUS_6
const ANONYMOUS_6 = UInt32
const CUDNN_CONVOLUTION = (UInt32)(0)
const CUDNN_CROSS_CORRELATION = (UInt32)(1)
# end enum ANONYMOUS_6

# begin enum cudnnConvolutionMode_t
const cudnnConvolutionMode_t = UInt32
const CUDNN_CONVOLUTION = (UInt32)(0)
const CUDNN_CROSS_CORRELATION = (UInt32)(1)
# end enum cudnnConvolutionMode_t

# begin enum ANONYMOUS_7
const ANONYMOUS_7 = UInt32
const CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum ANONYMOUS_7

# begin enum cudnnConvolutionFwdPreference_t
const cudnnConvolutionFwdPreference_t = UInt32
const CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum cudnnConvolutionFwdPreference_t

# begin enum ANONYMOUS_8
const ANONYMOUS_8 = UInt32
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = (UInt32)(0)
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = (UInt32)(1)
const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = (UInt32)(2)
const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = (UInt32)(3)
const CUDNN_CONVOLUTION_FWD_ALGO_FFT = (UInt32)(4)
const CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = (UInt32)(5)
const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = (UInt32)(6)
const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = (UInt32)(7)
# end enum ANONYMOUS_8

# begin enum cudnnConvolutionFwdAlgo_t
const cudnnConvolutionFwdAlgo_t = UInt32
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = (UInt32)(0)
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = (UInt32)(1)
const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = (UInt32)(2)
const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = (UInt32)(3)
const CUDNN_CONVOLUTION_FWD_ALGO_FFT = (UInt32)(4)
const CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = (UInt32)(5)
const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = (UInt32)(6)
const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = (UInt32)(7)
# end enum cudnnConvolutionFwdAlgo_t

mutable struct cudnnConvolutionFwdAlgoPerf_t
    algo::cudnnConvolutionFwdAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Cint
end

# begin enum ANONYMOUS_9
const ANONYMOUS_9 = UInt32
const CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum ANONYMOUS_9

# begin enum cudnnConvolutionBwdFilterPreference_t
const cudnnConvolutionBwdFilterPreference_t = UInt32
const CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum cudnnConvolutionBwdFilterPreference_t

# begin enum ANONYMOUS_10
const ANONYMOUS_10 = UInt32
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = (UInt32)(2)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = (UInt32)(3)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = (UInt32)(5)
# end enum ANONYMOUS_10

# begin enum cudnnConvolutionBwdFilterAlgo_t
const cudnnConvolutionBwdFilterAlgo_t = UInt32
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = (UInt32)(2)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = (UInt32)(3)
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = (UInt32)(5)
# end enum cudnnConvolutionBwdFilterAlgo_t

mutable struct cudnnConvolutionBwdFilterAlgoPerf_t
    algo::cudnnConvolutionBwdFilterAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Cint
end

# begin enum ANONYMOUS_11
const ANONYMOUS_11 = UInt32
const CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum ANONYMOUS_11

# begin enum cudnnConvolutionBwdDataPreference_t
const cudnnConvolutionBwdDataPreference_t = UInt32
const CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = (UInt32)(2)
# end enum cudnnConvolutionBwdDataPreference_t

# begin enum ANONYMOUS_12
const ANONYMOUS_12 = UInt32
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = (UInt32)(2)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = (UInt32)(3)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = (UInt32)(4)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = (UInt32)(5)
# end enum ANONYMOUS_12

# begin enum cudnnConvolutionBwdDataAlgo_t
const cudnnConvolutionBwdDataAlgo_t = UInt32
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = (UInt32)(0)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = (UInt32)(1)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = (UInt32)(2)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = (UInt32)(3)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = (UInt32)(4)
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = (UInt32)(5)
# end enum cudnnConvolutionBwdDataAlgo_t

mutable struct cudnnConvolutionBwdDataAlgoPerf_t
    algo::cudnnConvolutionBwdDataAlgo_t
    status::cudnnStatus_t
    time::Cfloat
    memory::Cint
end

# begin enum ANONYMOUS_13
const ANONYMOUS_13 = UInt32
const CUDNN_SOFTMAX_FAST = (UInt32)(0)
const CUDNN_SOFTMAX_ACCURATE = (UInt32)(1)
const CUDNN_SOFTMAX_LOG = (UInt32)(2)
# end enum ANONYMOUS_13

# begin enum cudnnSoftmaxAlgorithm_t
const cudnnSoftmaxAlgorithm_t = UInt32
const CUDNN_SOFTMAX_FAST = (UInt32)(0)
const CUDNN_SOFTMAX_ACCURATE = (UInt32)(1)
const CUDNN_SOFTMAX_LOG = (UInt32)(2)
# end enum cudnnSoftmaxAlgorithm_t

# begin enum ANONYMOUS_14
const ANONYMOUS_14 = UInt32
const CUDNN_SOFTMAX_MODE_INSTANCE = (UInt32)(0)
const CUDNN_SOFTMAX_MODE_CHANNEL = (UInt32)(1)
# end enum ANONYMOUS_14

# begin enum cudnnSoftmaxMode_t
const cudnnSoftmaxMode_t = UInt32
const CUDNN_SOFTMAX_MODE_INSTANCE = (UInt32)(0)
const CUDNN_SOFTMAX_MODE_CHANNEL = (UInt32)(1)
# end enum cudnnSoftmaxMode_t

# begin enum ANONYMOUS_15
const ANONYMOUS_15 = UInt32
const CUDNN_POOLING_MAX = (UInt32)(0)
const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = (UInt32)(1)
const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = (UInt32)(2)
# end enum ANONYMOUS_15

# begin enum cudnnPoolingMode_t
const cudnnPoolingMode_t = UInt32
const CUDNN_POOLING_MAX = (UInt32)(0)
const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = (UInt32)(1)
const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = (UInt32)(2)
# end enum cudnnPoolingMode_t

# begin enum ANONYMOUS_16
const ANONYMOUS_16 = UInt32
const CUDNN_ACTIVATION_SIGMOID = (UInt32)(0)
const CUDNN_ACTIVATION_RELU = (UInt32)(1)
const CUDNN_ACTIVATION_TANH = (UInt32)(2)
const CUDNN_ACTIVATION_CLIPPED_RELU = (UInt32)(3)
# end enum ANONYMOUS_16

# begin enum cudnnActivationMode_t
const cudnnActivationMode_t = UInt32
const CUDNN_ACTIVATION_SIGMOID = (UInt32)(0)
const CUDNN_ACTIVATION_RELU = (UInt32)(1)
const CUDNN_ACTIVATION_TANH = (UInt32)(2)
const CUDNN_ACTIVATION_CLIPPED_RELU = (UInt32)(3)
# end enum cudnnActivationMode_t

# begin enum ANONYMOUS_17
const ANONYMOUS_17 = UInt32
const CUDNN_LRN_CROSS_CHANNEL_DIM1 = (UInt32)(0)
# end enum ANONYMOUS_17

# begin enum cudnnLRNMode_t
const cudnnLRNMode_t = UInt32
const CUDNN_LRN_CROSS_CHANNEL_DIM1 = (UInt32)(0)
# end enum cudnnLRNMode_t

# begin enum ANONYMOUS_18
const ANONYMOUS_18 = UInt32
const CUDNN_DIVNORM_PRECOMPUTED_MEANS = (UInt32)(0)
# end enum ANONYMOUS_18

# begin enum cudnnDivNormMode_t
const cudnnDivNormMode_t = UInt32
const CUDNN_DIVNORM_PRECOMPUTED_MEANS = (UInt32)(0)
# end enum cudnnDivNormMode_t

# begin enum ANONYMOUS_19
const ANONYMOUS_19 = UInt32
const CUDNN_BATCHNORM_PER_ACTIVATION = (UInt32)(0)
const CUDNN_BATCHNORM_SPATIAL = (UInt32)(1)
# end enum ANONYMOUS_19

# begin enum cudnnBatchNormMode_t
const cudnnBatchNormMode_t = UInt32
const CUDNN_BATCHNORM_PER_ACTIVATION = (UInt32)(0)
const CUDNN_BATCHNORM_SPATIAL = (UInt32)(1)
# end enum cudnnBatchNormMode_t

# begin enum ANONYMOUS_20
const ANONYMOUS_20 = UInt32
const CUDNN_SAMPLER_BILINEAR = (UInt32)(0)
# end enum ANONYMOUS_20

# begin enum cudnnSamplerType_t
const cudnnSamplerType_t = UInt32
const CUDNN_SAMPLER_BILINEAR = (UInt32)(0)
# end enum cudnnSamplerType_t

mutable struct cudnnDropoutStruct
end

const cudnnDropoutDescriptor_t = Ptr{cudnnDropoutStruct}
# $(Expr(:typealias, :cudnnDropoutDescriptor_t, :(Ptr{cudnnDropoutStruct})))

# begin enum ANONYMOUS_21
const ANONYMOUS_21 = UInt32
const CUDNN_RNN_RELU = (UInt32)(0)
const CUDNN_RNN_TANH = (UInt32)(1)
const CUDNN_LSTM = (UInt32)(2)
const CUDNN_GRU = (UInt32)(3)
# end enum ANONYMOUS_21

# begin enum cudnnRNNMode_t
const cudnnRNNMode_t = UInt32
const CUDNN_RNN_RELU = (UInt32)(0)
const CUDNN_RNN_TANH = (UInt32)(1)
const CUDNN_LSTM = (UInt32)(2)
const CUDNN_GRU = (UInt32)(3)
# end enum cudnnRNNMode_t

# begin enum ANONYMOUS_22
const ANONYMOUS_22 = UInt32
const CUDNN_UNIDIRECTIONAL = (UInt32)(0)
const CUDNN_BIDIRECTIONAL = (UInt32)(1)
# end enum ANONYMOUS_22

# begin enum cudnnDirectionMode_t
const cudnnDirectionMode_t = UInt32
const CUDNN_UNIDIRECTIONAL = (UInt32)(0)
const CUDNN_BIDIRECTIONAL = (UInt32)(1)
# end enum cudnnDirectionMode_t

# begin enum ANONYMOUS_23
const ANONYMOUS_23 = UInt32
const CUDNN_LINEAR_INPUT = (UInt32)(0)
const CUDNN_SKIP_INPUT = (UInt32)(1)
# end enum ANONYMOUS_23

# begin enum cudnnRNNInputMode_t
const cudnnRNNInputMode_t = UInt32
const CUDNN_LINEAR_INPUT = (UInt32)(0)
const CUDNN_SKIP_INPUT = (UInt32)(1)
# end enum cudnnRNNInputMode_t

mutable struct cudnnRNNStruct
end

const cudnnRNNDescriptor_t = Ptr{cudnnRNNStruct}
# $(Expr(:typealias, :cudnnRNNDescriptor_t, :(Ptr{cudnnRNNStruct})))
