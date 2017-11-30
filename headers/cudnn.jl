# Julia wrapper for header: /usr/local/cuda-8.0/targets/x86_64-linux/include/cudnn.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0


function cudaDeviceReset()
    ccall((:cudaDeviceReset, cuda_runtime_api), cudaError_t, ())
end

function cudaDeviceSynchronize()
    ccall((:cudaDeviceSynchronize, cuda_runtime_api), cudaError_t, ())
end

function cudaDeviceSetLimit(limit::cudaLimit, value::Cint)
    ccall((:cudaDeviceSetLimit, cuda_runtime_api), cudaError_t, (cudaLimit, Cint), limit, value)
end

function cudaDeviceGetLimit(pValue, limit::cudaLimit)
    ccall((:cudaDeviceGetLimit, cuda_runtime_api), cudaError_t, (Ptr{Cint}, cudaLimit), pValue, limit)
end

function cudaDeviceGetCacheConfig(pCacheConfig)
    ccall((:cudaDeviceGetCacheConfig, cuda_runtime_api), cudaError_t, (Ptr{cudaFuncCache},), pCacheConfig)
end

function cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority)
    ccall((:cudaDeviceGetStreamPriorityRange, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Ptr{Cint}), leastPriority, greatestPriority)
end

function cudaDeviceSetCacheConfig(cacheConfig::cudaFuncCache)
    ccall((:cudaDeviceSetCacheConfig, cuda_runtime_api), cudaError_t, (cudaFuncCache,), cacheConfig)
end

function cudaDeviceGetSharedMemConfig(pConfig)
    ccall((:cudaDeviceGetSharedMemConfig, cuda_runtime_api), cudaError_t, (Ptr{cudaSharedMemConfig},), pConfig)
end

function cudaDeviceSetSharedMemConfig(config::cudaSharedMemConfig)
    ccall((:cudaDeviceSetSharedMemConfig, cuda_runtime_api), cudaError_t, (cudaSharedMemConfig,), config)
end

function cudaDeviceGetByPCIBusId(device, pciBusId)
    ccall((:cudaDeviceGetByPCIBusId, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Cstring), device, pciBusId)
end

function cudaDeviceGetPCIBusId(pciBusId, len::Cint, device::Cint)
    ccall((:cudaDeviceGetPCIBusId, cuda_runtime_api), cudaError_t, (Cstring, Cint, Cint), pciBusId, len, device)
end

function cudaIpcGetEventHandle(handle, event::cudaEvent_t)
    ccall((:cudaIpcGetEventHandle, cuda_runtime_api), cudaError_t, (Ptr{cudaIpcEventHandle_t}, cudaEvent_t), handle, event)
end

function cudaIpcOpenEventHandle(event, handle::cudaIpcEventHandle_t)
    ccall((:cudaIpcOpenEventHandle, cuda_runtime_api), cudaError_t, (Ptr{cudaEvent_t}, cudaIpcEventHandle_t), event, handle)
end

function cudaIpcGetMemHandle(handle, devPtr)
    ccall((:cudaIpcGetMemHandle, cuda_runtime_api), cudaError_t, (Ptr{cudaIpcMemHandle_t}, Ptr{Void}), handle, devPtr)
end

function cudaIpcOpenMemHandle(devPtr, handle::cudaIpcMemHandle_t, flags::UInt32)
    ccall((:cudaIpcOpenMemHandle, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, cudaIpcMemHandle_t, UInt32), devPtr, handle, flags)
end

function cudaIpcCloseMemHandle(devPtr)
    ccall((:cudaIpcCloseMemHandle, cuda_runtime_api), cudaError_t, (Ptr{Void},), devPtr)
end

function cudaThreadExit()
    ccall((:cudaThreadExit, cuda_runtime_api), cudaError_t, ())
end

function cudaThreadSynchronize()
    ccall((:cudaThreadSynchronize, cuda_runtime_api), cudaError_t, ())
end

function cudaThreadSetLimit(limit::cudaLimit, value::Cint)
    ccall((:cudaThreadSetLimit, cuda_runtime_api), cudaError_t, (cudaLimit, Cint), limit, value)
end

function cudaThreadGetLimit(pValue, limit::cudaLimit)
    ccall((:cudaThreadGetLimit, cuda_runtime_api), cudaError_t, (Ptr{Cint}, cudaLimit), pValue, limit)
end

function cudaThreadGetCacheConfig(pCacheConfig)
    ccall((:cudaThreadGetCacheConfig, cuda_runtime_api), cudaError_t, (Ptr{cudaFuncCache},), pCacheConfig)
end

function cudaThreadSetCacheConfig(cacheConfig::cudaFuncCache)
    ccall((:cudaThreadSetCacheConfig, cuda_runtime_api), cudaError_t, (cudaFuncCache,), cacheConfig)
end

function cudaGetLastError()
    ccall((:cudaGetLastError, cuda_runtime_api), cudaError_t, ())
end

function cudaPeekAtLastError()
    ccall((:cudaPeekAtLastError, cuda_runtime_api), cudaError_t, ())
end

function cudaGetErrorName(error::cudaError_t)
    ccall((:cudaGetErrorName, cuda_runtime_api), Cstring, (cudaError_t,), error)
end

function cudaGetErrorString(error::cudaError_t)
    ccall((:cudaGetErrorString, cuda_runtime_api), Cstring, (cudaError_t,), error)
end

function cudaGetDeviceCount(count)
    ccall((:cudaGetDeviceCount, cuda_runtime_api), cudaError_t, (Ptr{Cint},), count)
end

function cudaGetDeviceProperties(prop, device::Cint)
    ccall((:cudaGetDeviceProperties, cuda_runtime_api), cudaError_t, (Ptr{cudaDeviceProp}, Cint), prop, device)
end

function cudaDeviceGetAttribute(value, attr::cudaDeviceAttr, device::Cint)
    ccall((:cudaDeviceGetAttribute, cuda_runtime_api), cudaError_t, (Ptr{Cint}, cudaDeviceAttr, Cint), value, attr, device)
end

function cudaDeviceGetP2PAttribute(value, attr::cudaDeviceP2PAttr, srcDevice::Cint, dstDevice::Cint)
    ccall((:cudaDeviceGetP2PAttribute, cuda_runtime_api), cudaError_t, (Ptr{Cint}, cudaDeviceP2PAttr, Cint, Cint), value, attr, srcDevice, dstDevice)
end

function cudaChooseDevice(device, prop)
    ccall((:cudaChooseDevice, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Ptr{cudaDeviceProp}), device, prop)
end

function cudaSetDevice(device::Cint)
    ccall((:cudaSetDevice, cuda_runtime_api), cudaError_t, (Cint,), device)
end

function cudaGetDevice(device)
    ccall((:cudaGetDevice, cuda_runtime_api), cudaError_t, (Ptr{Cint},), device)
end

function cudaSetValidDevices(device_arr, len::Cint)
    ccall((:cudaSetValidDevices, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Cint), device_arr, len)
end

function cudaSetDeviceFlags(flags::UInt32)
    ccall((:cudaSetDeviceFlags, cuda_runtime_api), cudaError_t, (UInt32,), flags)
end

function cudaGetDeviceFlags(flags)
    ccall((:cudaGetDeviceFlags, cuda_runtime_api), cudaError_t, (Ptr{UInt32},), flags)
end

function cudaStreamCreate(pStream)
    ccall((:cudaStreamCreate, cuda_runtime_api), cudaError_t, (Ptr{cudaStream_t},), pStream)
end

function cudaStreamCreateWithFlags(pStream, flags::UInt32)
    ccall((:cudaStreamCreateWithFlags, cuda_runtime_api), cudaError_t, (Ptr{cudaStream_t}, UInt32), pStream, flags)
end

function cudaStreamCreateWithPriority(pStream, flags::UInt32, priority::Cint)
    ccall((:cudaStreamCreateWithPriority, cuda_runtime_api), cudaError_t, (Ptr{cudaStream_t}, UInt32, Cint), pStream, flags, priority)
end

function cudaStreamGetPriority(hStream::cudaStream_t, priority)
    ccall((:cudaStreamGetPriority, cuda_runtime_api), cudaError_t, (cudaStream_t, Ptr{Cint}), hStream, priority)
end

function cudaStreamGetFlags(hStream::cudaStream_t, flags)
    ccall((:cudaStreamGetFlags, cuda_runtime_api), cudaError_t, (cudaStream_t, Ptr{UInt32}), hStream, flags)
end

function cudaStreamDestroy(stream::cudaStream_t)
    ccall((:cudaStreamDestroy, cuda_runtime_api), cudaError_t, (cudaStream_t,), stream)
end

function cudaStreamWaitEvent(stream::cudaStream_t, event::cudaEvent_t, flags::UInt32)
    ccall((:cudaStreamWaitEvent, cuda_runtime_api), cudaError_t, (cudaStream_t, cudaEvent_t, UInt32), stream, event, flags)
end

function cudaStreamAddCallback(stream::cudaStream_t, callback::cudaStreamCallback_t, userData, flags::UInt32)
    ccall((:cudaStreamAddCallback, cuda_runtime_api), cudaError_t, (cudaStream_t, cudaStreamCallback_t, Ptr{Void}, UInt32), stream, callback, userData, flags)
end

function cudaStreamSynchronize(stream::cudaStream_t)
    ccall((:cudaStreamSynchronize, cuda_runtime_api), cudaError_t, (cudaStream_t,), stream)
end

function cudaStreamQuery(stream::cudaStream_t)
    ccall((:cudaStreamQuery, cuda_runtime_api), cudaError_t, (cudaStream_t,), stream)
end

function cudaStreamAttachMemAsync(stream::cudaStream_t, devPtr, length::Cint, flags::UInt32)
    ccall((:cudaStreamAttachMemAsync, cuda_runtime_api), cudaError_t, (cudaStream_t, Ptr{Void}, Cint, UInt32), stream, devPtr, length, flags)
end

function cudaEventCreate(event)
    ccall((:cudaEventCreate, cuda_runtime_api), cudaError_t, (Ptr{cudaEvent_t},), event)
end

function cudaEventCreateWithFlags(event, flags::UInt32)
    ccall((:cudaEventCreateWithFlags, cuda_runtime_api), cudaError_t, (Ptr{cudaEvent_t}, UInt32), event, flags)
end

function cudaEventRecord(event::cudaEvent_t, stream::cudaStream_t)
    ccall((:cudaEventRecord, cuda_runtime_api), cudaError_t, (cudaEvent_t, cudaStream_t), event, stream)
end

function cudaEventQuery(event::cudaEvent_t)
    ccall((:cudaEventQuery, cuda_runtime_api), cudaError_t, (cudaEvent_t,), event)
end

function cudaEventSynchronize(event::cudaEvent_t)
    ccall((:cudaEventSynchronize, cuda_runtime_api), cudaError_t, (cudaEvent_t,), event)
end

function cudaEventDestroy(event::cudaEvent_t)
    ccall((:cudaEventDestroy, cuda_runtime_api), cudaError_t, (cudaEvent_t,), event)
end

function cudaEventElapsedTime(ms, start::cudaEvent_t, _end::cudaEvent_t)
    ccall((:cudaEventElapsedTime, cuda_runtime_api), cudaError_t, (Ptr{Cfloat}, cudaEvent_t, cudaEvent_t), ms, start, _end)
end

function cudaLaunchKernel(func, gridDim::dim3, blockDim::dim3, args, sharedMem::Cint, stream::cudaStream_t)
    ccall((:cudaLaunchKernel, cuda_runtime_api), cudaError_t, (Ptr{Void}, dim3, dim3, Ptr{Ptr{Void}}, Cint, cudaStream_t), func, gridDim, blockDim, args, sharedMem, stream)
end

function cudaFuncSetCacheConfig(func, cacheConfig::cudaFuncCache)
    ccall((:cudaFuncSetCacheConfig, cuda_runtime_api), cudaError_t, (Ptr{Void}, cudaFuncCache), func, cacheConfig)
end

function cudaFuncSetSharedMemConfig(func, config::cudaSharedMemConfig)
    ccall((:cudaFuncSetSharedMemConfig, cuda_runtime_api), cudaError_t, (Ptr{Void}, cudaSharedMemConfig), func, config)
end

function cudaFuncGetAttributes(attr, func)
    ccall((:cudaFuncGetAttributes, cuda_runtime_api), cudaError_t, (Ptr{cudaFuncAttributes}, Ptr{Void}), attr, func)
end

function cudaSetDoubleForDevice(d)
    ccall((:cudaSetDoubleForDevice, cuda_runtime_api), cudaError_t, (Ptr{Cdouble},), d)
end

function cudaSetDoubleForHost(d)
    ccall((:cudaSetDoubleForHost, cuda_runtime_api), cudaError_t, (Ptr{Cdouble},), d)
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize::Cint, dynamicSMemSize::Cint)
    ccall((:cudaOccupancyMaxActiveBlocksPerMultiprocessor, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Ptr{Void}, Cint, Cint), numBlocks, func, blockSize, dynamicSMemSize)
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize::Cint, dynamicSMemSize::Cint, flags::UInt32)
    ccall((:cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Ptr{Void}, Cint, Cint, UInt32), numBlocks, func, blockSize, dynamicSMemSize, flags)
end

function cudaConfigureCall(gridDim::dim3, blockDim::dim3, sharedMem::Cint, stream::cudaStream_t)
    ccall((:cudaConfigureCall, cuda_runtime_api), cudaError_t, (dim3, dim3, Cint, cudaStream_t), gridDim, blockDim, sharedMem, stream)
end

function cudaSetupArgument(arg, size::Cint, offset::Cint)
    ccall((:cudaSetupArgument, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Cint), arg, size, offset)
end

function cudaLaunch(func)
    ccall((:cudaLaunch, cuda_runtime_api), cudaError_t, (Ptr{Void},), func)
end

function cudaMallocManaged(devPtr, size::Cint, flags::UInt32)
    ccall((:cudaMallocManaged, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Cint, UInt32), devPtr, size, flags)
end

function cudaMalloc(devPtr, size::Cint)
    ccall((:cudaMalloc, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Cint), devPtr, size)
end

function cudaMallocHost(ptr, size::Cint)
    ccall((:cudaMallocHost, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Cint), ptr, size)
end

function cudaMallocPitch(devPtr, pitch, width::Cint, height::Cint)
    ccall((:cudaMallocPitch, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Ptr{Cint}, Cint, Cint), devPtr, pitch, width, height)
end

function cudaMallocArray(array, desc, width::Cint, height::Cint, flags::UInt32)
    ccall((:cudaMallocArray, cuda_runtime_api), cudaError_t, (Ptr{cudaArray_t}, Ptr{cudaChannelFormatDesc}, Cint, Cint, UInt32), array, desc, width, height, flags)
end

function cudaFree(devPtr)
    ccall((:cudaFree, cuda_runtime_api), cudaError_t, (Ptr{Void},), devPtr)
end

function cudaFreeHost(ptr)
    ccall((:cudaFreeHost, cuda_runtime_api), cudaError_t, (Ptr{Void},), ptr)
end

function cudaFreeArray(array::cudaArray_t)
    ccall((:cudaFreeArray, cuda_runtime_api), cudaError_t, (cudaArray_t,), array)
end

function cudaFreeMipmappedArray(mipmappedArray::cudaMipmappedArray_t)
    ccall((:cudaFreeMipmappedArray, cuda_runtime_api), cudaError_t, (cudaMipmappedArray_t,), mipmappedArray)
end

function cudaHostAlloc(pHost, size::Cint, flags::UInt32)
    ccall((:cudaHostAlloc, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Cint, UInt32), pHost, size, flags)
end

function cudaHostRegister(ptr, size::Cint, flags::UInt32)
    ccall((:cudaHostRegister, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, UInt32), ptr, size, flags)
end

function cudaHostUnregister(ptr)
    ccall((:cudaHostUnregister, cuda_runtime_api), cudaError_t, (Ptr{Void},), ptr)
end

function cudaHostGetDevicePointer(pDevice, pHost, flags::UInt32)
    ccall((:cudaHostGetDevicePointer, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Ptr{Void}, UInt32), pDevice, pHost, flags)
end

function cudaHostGetFlags(pFlags, pHost)
    ccall((:cudaHostGetFlags, cuda_runtime_api), cudaError_t, (Ptr{UInt32}, Ptr{Void}), pFlags, pHost)
end

function cudaMalloc3D(pitchedDevPtr, extent::cudaExtent)
    ccall((:cudaMalloc3D, cuda_runtime_api), cudaError_t, (Ptr{cudaPitchedPtr}, cudaExtent), pitchedDevPtr, extent)
end

function cudaMalloc3DArray(array, desc, extent::cudaExtent, flags::UInt32)
    ccall((:cudaMalloc3DArray, cuda_runtime_api), cudaError_t, (Ptr{cudaArray_t}, Ptr{cudaChannelFormatDesc}, cudaExtent, UInt32), array, desc, extent, flags)
end

function cudaMallocMipmappedArray(mipmappedArray, desc, extent::cudaExtent, numLevels::UInt32, flags::UInt32)
    ccall((:cudaMallocMipmappedArray, cuda_runtime_api), cudaError_t, (Ptr{cudaMipmappedArray_t}, Ptr{cudaChannelFormatDesc}, cudaExtent, UInt32, UInt32), mipmappedArray, desc, extent, numLevels, flags)
end

function cudaGetMipmappedArrayLevel(levelArray, mipmappedArray::cudaMipmappedArray_const_t, level::UInt32)
    ccall((:cudaGetMipmappedArrayLevel, cuda_runtime_api), cudaError_t, (Ptr{cudaArray_t}, cudaMipmappedArray_const_t, UInt32), levelArray, mipmappedArray, level)
end

function cudaMemcpy3D(p)
    ccall((:cudaMemcpy3D, cuda_runtime_api), cudaError_t, (Ptr{cudaMemcpy3DParms},), p)
end

function cudaMemcpy3DPeer(p)
    ccall((:cudaMemcpy3DPeer, cuda_runtime_api), cudaError_t, (Ptr{cudaMemcpy3DPeerParms},), p)
end

function cudaMemcpy3DAsync(p, stream::cudaStream_t)
    ccall((:cudaMemcpy3DAsync, cuda_runtime_api), cudaError_t, (Ptr{cudaMemcpy3DParms}, cudaStream_t), p, stream)
end

function cudaMemcpy3DPeerAsync(p, stream::cudaStream_t)
    ccall((:cudaMemcpy3DPeerAsync, cuda_runtime_api), cudaError_t, (Ptr{cudaMemcpy3DPeerParms}, cudaStream_t), p, stream)
end

function cudaMemGetInfo(free, total)
    ccall((:cudaMemGetInfo, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Ptr{Cint}), free, total)
end

function cudaArrayGetInfo(desc, extent, flags, array::cudaArray_t)
    ccall((:cudaArrayGetInfo, cuda_runtime_api), cudaError_t, (Ptr{cudaChannelFormatDesc}, Ptr{cudaExtent}, Ptr{UInt32}, cudaArray_t), desc, extent, flags, array)
end

function cudaMemcpy(dst, src, count::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpy, cuda_runtime_api), cudaError_t, (Ptr{Void}, Ptr{Void}, Cint, cudaMemcpyKind), dst, src, count, kind)
end

function cudaMemcpyPeer(dst, dstDevice::Cint, src, srcDevice::Cint, count::Cint)
    ccall((:cudaMemcpyPeer, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Ptr{Void}, Cint, Cint), dst, dstDevice, src, srcDevice, count)
end

function cudaMemcpyToArray(dst::cudaArray_t, wOffset::Cint, hOffset::Cint, src, count::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpyToArray, cuda_runtime_api), cudaError_t, (cudaArray_t, Cint, Cint, Ptr{Void}, Cint, cudaMemcpyKind), dst, wOffset, hOffset, src, count, kind)
end

function cudaMemcpyFromArray(dst, src::cudaArray_const_t, wOffset::Cint, hOffset::Cint, count::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpyFromArray, cuda_runtime_api), cudaError_t, (Ptr{Void}, cudaArray_const_t, Cint, Cint, Cint, cudaMemcpyKind), dst, src, wOffset, hOffset, count, kind)
end

function cudaMemcpyArrayToArray(dst::cudaArray_t, wOffsetDst::Cint, hOffsetDst::Cint, src::cudaArray_const_t, wOffsetSrc::Cint, hOffsetSrc::Cint, count::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpyArrayToArray, cuda_runtime_api), cudaError_t, (cudaArray_t, Cint, Cint, cudaArray_const_t, Cint, Cint, Cint, cudaMemcpyKind), dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)
end

function cudaMemcpy2D(dst, dpitch::Cint, src, spitch::Cint, width::Cint, height::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpy2D, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Ptr{Void}, Cint, Cint, Cint, cudaMemcpyKind), dst, dpitch, src, spitch, width, height, kind)
end

function cudaMemcpy2DToArray(dst::cudaArray_t, wOffset::Cint, hOffset::Cint, src, spitch::Cint, width::Cint, height::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpy2DToArray, cuda_runtime_api), cudaError_t, (cudaArray_t, Cint, Cint, Ptr{Void}, Cint, Cint, Cint, cudaMemcpyKind), dst, wOffset, hOffset, src, spitch, width, height, kind)
end

function cudaMemcpy2DFromArray(dst, dpitch::Cint, src::cudaArray_const_t, wOffset::Cint, hOffset::Cint, width::Cint, height::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpy2DFromArray, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, cudaArray_const_t, Cint, Cint, Cint, Cint, cudaMemcpyKind), dst, dpitch, src, wOffset, hOffset, width, height, kind)
end

function cudaMemcpy2DArrayToArray(dst::cudaArray_t, wOffsetDst::Cint, hOffsetDst::Cint, src::cudaArray_const_t, wOffsetSrc::Cint, hOffsetSrc::Cint, width::Cint, height::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpy2DArrayToArray, cuda_runtime_api), cudaError_t, (cudaArray_t, Cint, Cint, cudaArray_const_t, Cint, Cint, Cint, Cint, cudaMemcpyKind), dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)
end

function cudaMemcpyToSymbol(symbol, src, count::Cint, offset::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpyToSymbol, cuda_runtime_api), cudaError_t, (Ptr{Void}, Ptr{Void}, Cint, Cint, cudaMemcpyKind), symbol, src, count, offset, kind)
end

function cudaMemcpyFromSymbol(dst, symbol, count::Cint, offset::Cint, kind::cudaMemcpyKind)
    ccall((:cudaMemcpyFromSymbol, cuda_runtime_api), cudaError_t, (Ptr{Void}, Ptr{Void}, Cint, Cint, cudaMemcpyKind), dst, symbol, count, offset, kind)
end

function cudaMemcpyAsync(dst, src, count::Cint, kind::cudaMemcpyKind, stream::cudaStream_t)
    ccall((:cudaMemcpyAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, Ptr{Void}, Cint, cudaMemcpyKind, cudaStream_t), dst, src, count, kind, stream)
end

function cudaMemcpyPeerAsync(dst, dstDevice::Cint, src, srcDevice::Cint, count::Cint, stream::cudaStream_t)
    ccall((:cudaMemcpyPeerAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Ptr{Void}, Cint, Cint, cudaStream_t), dst, dstDevice, src, srcDevice, count, stream)
end

function cudaMemcpyToArrayAsync(dst::cudaArray_t, wOffset::Cint, hOffset::Cint, src, count::Cint, kind::cudaMemcpyKind, stream::cudaStream_t)
    ccall((:cudaMemcpyToArrayAsync, cuda_runtime_api), cudaError_t, (cudaArray_t, Cint, Cint, Ptr{Void}, Cint, cudaMemcpyKind, cudaStream_t), dst, wOffset, hOffset, src, count, kind, stream)
end

function cudaMemcpyFromArrayAsync(dst, src::cudaArray_const_t, wOffset::Cint, hOffset::Cint, count::Cint, kind::cudaMemcpyKind, stream::cudaStream_t)
    ccall((:cudaMemcpyFromArrayAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, cudaArray_const_t, Cint, Cint, Cint, cudaMemcpyKind, cudaStream_t), dst, src, wOffset, hOffset, count, kind, stream)
end

function cudaMemcpy2DAsync(dst, dpitch::Cint, src, spitch::Cint, width::Cint, height::Cint, kind::cudaMemcpyKind, stream::cudaStream_t)
    ccall((:cudaMemcpy2DAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Ptr{Void}, Cint, Cint, Cint, cudaMemcpyKind, cudaStream_t), dst, dpitch, src, spitch, width, height, kind, stream)
end

function cudaMemcpy2DToArrayAsync(dst::cudaArray_t, wOffset::Cint, hOffset::Cint, src, spitch::Cint, width::Cint, height::Cint, kind::cudaMemcpyKind, stream::cudaStream_t)
    ccall((:cudaMemcpy2DToArrayAsync, cuda_runtime_api), cudaError_t, (cudaArray_t, Cint, Cint, Ptr{Void}, Cint, Cint, Cint, cudaMemcpyKind, cudaStream_t), dst, wOffset, hOffset, src, spitch, width, height, kind, stream)
end

function cudaMemcpy2DFromArrayAsync(dst, dpitch::Cint, src::cudaArray_const_t, wOffset::Cint, hOffset::Cint, width::Cint, height::Cint, kind::cudaMemcpyKind, stream::cudaStream_t)
    ccall((:cudaMemcpy2DFromArrayAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, cudaArray_const_t, Cint, Cint, Cint, Cint, cudaMemcpyKind, cudaStream_t), dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)
end

function cudaMemcpyToSymbolAsync(symbol, src, count::Cint, offset::Cint, kind::cudaMemcpyKind, stream::cudaStream_t)
    ccall((:cudaMemcpyToSymbolAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, Ptr{Void}, Cint, Cint, cudaMemcpyKind, cudaStream_t), symbol, src, count, offset, kind, stream)
end

function cudaMemcpyFromSymbolAsync(dst, symbol, count::Cint, offset::Cint, kind::cudaMemcpyKind, stream::cudaStream_t)
    ccall((:cudaMemcpyFromSymbolAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, Ptr{Void}, Cint, Cint, cudaMemcpyKind, cudaStream_t), dst, symbol, count, offset, kind, stream)
end

function cudaMemset(devPtr, value::Cint, count::Cint)
    ccall((:cudaMemset, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Cint), devPtr, value, count)
end

function cudaMemset2D(devPtr, pitch::Cint, value::Cint, width::Cint, height::Cint)
    ccall((:cudaMemset2D, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Cint, Cint, Cint), devPtr, pitch, value, width, height)
end

function cudaMemset3D(pitchedDevPtr::cudaPitchedPtr, value::Cint, extent::cudaExtent)
    ccall((:cudaMemset3D, cuda_runtime_api), cudaError_t, (cudaPitchedPtr, Cint, cudaExtent), pitchedDevPtr, value, extent)
end

function cudaMemsetAsync(devPtr, value::Cint, count::Cint, stream::cudaStream_t)
    ccall((:cudaMemsetAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Cint, cudaStream_t), devPtr, value, count, stream)
end

function cudaMemset2DAsync(devPtr, pitch::Cint, value::Cint, width::Cint, height::Cint, stream::cudaStream_t)
    ccall((:cudaMemset2DAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Cint, Cint, Cint, cudaStream_t), devPtr, pitch, value, width, height, stream)
end

function cudaMemset3DAsync(pitchedDevPtr::cudaPitchedPtr, value::Cint, extent::cudaExtent, stream::cudaStream_t)
    ccall((:cudaMemset3DAsync, cuda_runtime_api), cudaError_t, (cudaPitchedPtr, Cint, cudaExtent, cudaStream_t), pitchedDevPtr, value, extent, stream)
end

function cudaGetSymbolAddress(devPtr, symbol)
    ccall((:cudaGetSymbolAddress, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Ptr{Void}), devPtr, symbol)
end

function cudaGetSymbolSize(size, symbol)
    ccall((:cudaGetSymbolSize, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Ptr{Void}), size, symbol)
end

function cudaMemPrefetchAsync(devPtr, count::Cint, dstDevice::Cint, stream::cudaStream_t)
    ccall((:cudaMemPrefetchAsync, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, Cint, cudaStream_t), devPtr, count, dstDevice, stream)
end

function cudaMemAdvise(devPtr, count::Cint, advice::cudaMemoryAdvise, device::Cint)
    ccall((:cudaMemAdvise, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, cudaMemoryAdvise, Cint), devPtr, count, advice, device)
end

function cudaMemRangeGetAttribute(data, dataSize::Cint, attribute::cudaMemRangeAttribute, devPtr, count::Cint)
    ccall((:cudaMemRangeGetAttribute, cuda_runtime_api), cudaError_t, (Ptr{Void}, Cint, cudaMemRangeAttribute, Ptr{Void}, Cint), data, dataSize, attribute, devPtr, count)
end

function cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes::Cint, devPtr, count::Cint)
    ccall((:cudaMemRangeGetAttributes, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Ptr{Cint}, Ptr{cudaMemRangeAttribute}, Cint, Ptr{Void}, Cint), data, dataSizes, attributes, numAttributes, devPtr, count)
end

function cudaPointerGetAttributes(attributes, ptr)
    ccall((:cudaPointerGetAttributes, cuda_runtime_api), cudaError_t, (Ptr{cudaPointerAttributes}, Ptr{Void}), attributes, ptr)
end

function cudaDeviceCanAccessPeer(canAccessPeer, device::Cint, peerDevice::Cint)
    ccall((:cudaDeviceCanAccessPeer, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Cint, Cint), canAccessPeer, device, peerDevice)
end

function cudaDeviceEnablePeerAccess(peerDevice::Cint, flags::UInt32)
    ccall((:cudaDeviceEnablePeerAccess, cuda_runtime_api), cudaError_t, (Cint, UInt32), peerDevice, flags)
end

function cudaDeviceDisablePeerAccess(peerDevice::Cint)
    ccall((:cudaDeviceDisablePeerAccess, cuda_runtime_api), cudaError_t, (Cint,), peerDevice)
end

function cudaGraphicsUnregisterResource(resource::cudaGraphicsResource_t)
    ccall((:cudaGraphicsUnregisterResource, cuda_runtime_api), cudaError_t, (cudaGraphicsResource_t,), resource)
end

function cudaGraphicsResourceSetMapFlags(resource::cudaGraphicsResource_t, flags::UInt32)
    ccall((:cudaGraphicsResourceSetMapFlags, cuda_runtime_api), cudaError_t, (cudaGraphicsResource_t, UInt32), resource, flags)
end

function cudaGraphicsMapResources(count::Cint, resources, stream::cudaStream_t)
    ccall((:cudaGraphicsMapResources, cuda_runtime_api), cudaError_t, (Cint, Ptr{cudaGraphicsResource_t}, cudaStream_t), count, resources, stream)
end

function cudaGraphicsUnmapResources(count::Cint, resources, stream::cudaStream_t)
    ccall((:cudaGraphicsUnmapResources, cuda_runtime_api), cudaError_t, (Cint, Ptr{cudaGraphicsResource_t}, cudaStream_t), count, resources, stream)
end

function cudaGraphicsResourceGetMappedPointer(devPtr, size, resource::cudaGraphicsResource_t)
    ccall((:cudaGraphicsResourceGetMappedPointer, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Ptr{Cint}, cudaGraphicsResource_t), devPtr, size, resource)
end

function cudaGraphicsSubResourceGetMappedArray(array, resource::cudaGraphicsResource_t, arrayIndex::UInt32, mipLevel::UInt32)
    ccall((:cudaGraphicsSubResourceGetMappedArray, cuda_runtime_api), cudaError_t, (Ptr{cudaArray_t}, cudaGraphicsResource_t, UInt32, UInt32), array, resource, arrayIndex, mipLevel)
end

function cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource::cudaGraphicsResource_t)
    ccall((:cudaGraphicsResourceGetMappedMipmappedArray, cuda_runtime_api), cudaError_t, (Ptr{cudaMipmappedArray_t}, cudaGraphicsResource_t), mipmappedArray, resource)
end

function cudaGetChannelDesc(desc, array::cudaArray_const_t)
    ccall((:cudaGetChannelDesc, cuda_runtime_api), cudaError_t, (Ptr{cudaChannelFormatDesc}, cudaArray_const_t), desc, array)
end

function cudaCreateChannelDesc(x::Cint, y::Cint, z::Cint, w::Cint, f::cudaChannelFormatKind)
    ccall((:cudaCreateChannelDesc, cuda_runtime_api), cudaChannelFormatDesc, (Cint, Cint, Cint, Cint, cudaChannelFormatKind), x, y, z, w, f)
end

function cudaBindTexture(offset, texref, devPtr, desc, size::Cint)
    ccall((:cudaBindTexture, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Ptr{textureReference}, Ptr{Void}, Ptr{cudaChannelFormatDesc}, Cint), offset, texref, devPtr, desc, size)
end

function cudaBindTexture2D(offset, texref, devPtr, desc, width::Cint, height::Cint, pitch::Cint)
    ccall((:cudaBindTexture2D, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Ptr{textureReference}, Ptr{Void}, Ptr{cudaChannelFormatDesc}, Cint, Cint, Cint), offset, texref, devPtr, desc, width, height, pitch)
end

function cudaBindTextureToArray(texref, array::cudaArray_const_t, desc)
    ccall((:cudaBindTextureToArray, cuda_runtime_api), cudaError_t, (Ptr{textureReference}, cudaArray_const_t, Ptr{cudaChannelFormatDesc}), texref, array, desc)
end

function cudaBindTextureToMipmappedArray(texref, mipmappedArray::cudaMipmappedArray_const_t, desc)
    ccall((:cudaBindTextureToMipmappedArray, cuda_runtime_api), cudaError_t, (Ptr{textureReference}, cudaMipmappedArray_const_t, Ptr{cudaChannelFormatDesc}), texref, mipmappedArray, desc)
end

function cudaUnbindTexture(texref)
    ccall((:cudaUnbindTexture, cuda_runtime_api), cudaError_t, (Ptr{textureReference},), texref)
end

function cudaGetTextureAlignmentOffset(offset, texref)
    ccall((:cudaGetTextureAlignmentOffset, cuda_runtime_api), cudaError_t, (Ptr{Cint}, Ptr{textureReference}), offset, texref)
end

function cudaGetTextureReference(texref, symbol)
    ccall((:cudaGetTextureReference, cuda_runtime_api), cudaError_t, (Ptr{Ptr{textureReference}}, Ptr{Void}), texref, symbol)
end

function cudaBindSurfaceToArray(surfref, array::cudaArray_const_t, desc)
    ccall((:cudaBindSurfaceToArray, cuda_runtime_api), cudaError_t, (Ptr{surfaceReference}, cudaArray_const_t, Ptr{cudaChannelFormatDesc}), surfref, array, desc)
end

function cudaGetSurfaceReference(surfref, symbol)
    ccall((:cudaGetSurfaceReference, cuda_runtime_api), cudaError_t, (Ptr{Ptr{surfaceReference}}, Ptr{Void}), surfref, symbol)
end

function cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    ccall((:cudaCreateTextureObject, cuda_runtime_api), cudaError_t, (Ptr{cudaTextureObject_t}, Ptr{cudaResourceDesc}, Ptr{cudaTextureDesc}, Ptr{cudaResourceViewDesc}), pTexObject, pResDesc, pTexDesc, pResViewDesc)
end

function cudaDestroyTextureObject(texObject::cudaTextureObject_t)
    ccall((:cudaDestroyTextureObject, cuda_runtime_api), cudaError_t, (cudaTextureObject_t,), texObject)
end

function cudaGetTextureObjectResourceDesc(pResDesc, texObject::cudaTextureObject_t)
    ccall((:cudaGetTextureObjectResourceDesc, cuda_runtime_api), cudaError_t, (Ptr{cudaResourceDesc}, cudaTextureObject_t), pResDesc, texObject)
end

function cudaGetTextureObjectTextureDesc(pTexDesc, texObject::cudaTextureObject_t)
    ccall((:cudaGetTextureObjectTextureDesc, cuda_runtime_api), cudaError_t, (Ptr{cudaTextureDesc}, cudaTextureObject_t), pTexDesc, texObject)
end

function cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject::cudaTextureObject_t)
    ccall((:cudaGetTextureObjectResourceViewDesc, cuda_runtime_api), cudaError_t, (Ptr{cudaResourceViewDesc}, cudaTextureObject_t), pResViewDesc, texObject)
end

function cudaCreateSurfaceObject(pSurfObject, pResDesc)
    ccall((:cudaCreateSurfaceObject, cuda_runtime_api), cudaError_t, (Ptr{cudaSurfaceObject_t}, Ptr{cudaResourceDesc}), pSurfObject, pResDesc)
end

function cudaDestroySurfaceObject(surfObject::cudaSurfaceObject_t)
    ccall((:cudaDestroySurfaceObject, cuda_runtime_api), cudaError_t, (cudaSurfaceObject_t,), surfObject)
end

function cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject::cudaSurfaceObject_t)
    ccall((:cudaGetSurfaceObjectResourceDesc, cuda_runtime_api), cudaError_t, (Ptr{cudaResourceDesc}, cudaSurfaceObject_t), pResDesc, surfObject)
end

function cudaDriverGetVersion(driverVersion)
    ccall((:cudaDriverGetVersion, cuda_runtime_api), cudaError_t, (Ptr{Cint},), driverVersion)
end

function cudaRuntimeGetVersion(runtimeVersion)
    ccall((:cudaRuntimeGetVersion, cuda_runtime_api), cudaError_t, (Ptr{Cint},), runtimeVersion)
end

function cudaGetExportTable(ppExportTable, pExportTableId)
    ccall((:cudaGetExportTable, cuda_runtime_api), cudaError_t, (Ptr{Ptr{Void}}, Ptr{cudaUUID_t}), ppExportTable, pExportTableId)
end

function make_cudaPitchedPtr(d, p::Cint, xsz::Cint, ysz::Cint)
    ccall((:make_cudaPitchedPtr, driver_functions), cudaPitchedPtr, (Ptr{Void}, Cint, Cint, Cint), d, p, xsz, ysz)
end

function make_cudaPos(x::Cint, y::Cint, z::Cint)
    ccall((:make_cudaPos, driver_functions), cudaPos, (Cint, Cint, Cint), x, y, z)
end

function make_cudaExtent(w::Cint, h::Cint, d::Cint)
    ccall((:make_cudaExtent, driver_functions), cudaExtent, (Cint, Cint, Cint), w, h, d)
end

function make_char1(x::UInt8)
    ccall((:make_char1, vector_functions), char1, (UInt8,), x)
end

function make_uchar1(x::Cuchar)
    ccall((:make_uchar1, vector_functions), uchar1, (Cuchar,), x)
end

function make_char2(x::UInt8, y::UInt8)
    ccall((:make_char2, vector_functions), char2, (UInt8, UInt8), x, y)
end

function make_uchar2(x::Cuchar, y::Cuchar)
    ccall((:make_uchar2, vector_functions), uchar2, (Cuchar, Cuchar), x, y)
end

function make_char3(x::UInt8, y::UInt8, z::UInt8)
    ccall((:make_char3, vector_functions), char3, (UInt8, UInt8, UInt8), x, y, z)
end

function make_uchar3(x::Cuchar, y::Cuchar, z::Cuchar)
    ccall((:make_uchar3, vector_functions), uchar3, (Cuchar, Cuchar, Cuchar), x, y, z)
end

function make_char4(x::UInt8, y::UInt8, z::UInt8, w::UInt8)
    ccall((:make_char4, vector_functions), char4, (UInt8, UInt8, UInt8, UInt8), x, y, z, w)
end

function make_uchar4(x::Cuchar, y::Cuchar, z::Cuchar, w::Cuchar)
    ccall((:make_uchar4, vector_functions), uchar4, (Cuchar, Cuchar, Cuchar, Cuchar), x, y, z, w)
end

function make_short1(x::Int16)
    ccall((:make_short1, vector_functions), short1, (Int16,), x)
end

function make_ushort1(x::UInt16)
    ccall((:make_ushort1, vector_functions), ushort1, (UInt16,), x)
end

function make_short2(x::Int16, y::Int16)
    ccall((:make_short2, vector_functions), short2, (Int16, Int16), x, y)
end

function make_ushort2(x::UInt16, y::UInt16)
    ccall((:make_ushort2, vector_functions), ushort2, (UInt16, UInt16), x, y)
end

function make_short3(x::Int16, y::Int16, z::Int16)
    ccall((:make_short3, vector_functions), short3, (Int16, Int16, Int16), x, y, z)
end

function make_ushort3(x::UInt16, y::UInt16, z::UInt16)
    ccall((:make_ushort3, vector_functions), ushort3, (UInt16, UInt16, UInt16), x, y, z)
end

function make_short4(x::Int16, y::Int16, z::Int16, w::Int16)
    ccall((:make_short4, vector_functions), short4, (Int16, Int16, Int16, Int16), x, y, z, w)
end

function make_ushort4(x::UInt16, y::UInt16, z::UInt16, w::UInt16)
    ccall((:make_ushort4, vector_functions), ushort4, (UInt16, UInt16, UInt16, UInt16), x, y, z, w)
end

function make_int1(x::Cint)
    ccall((:make_int1, vector_functions), int1, (Cint,), x)
end

function make_uint1(x::UInt32)
    ccall((:make_uint1, vector_functions), uint1, (UInt32,), x)
end

function make_int2(x::Cint, y::Cint)
    ccall((:make_int2, vector_functions), int2, (Cint, Cint), x, y)
end

function make_uint2(x::UInt32, y::UInt32)
    ccall((:make_uint2, vector_functions), uint2, (UInt32, UInt32), x, y)
end

function make_int3(x::Cint, y::Cint, z::Cint)
    ccall((:make_int3, vector_functions), int3, (Cint, Cint, Cint), x, y, z)
end

function make_uint3(x::UInt32, y::UInt32, z::UInt32)
    ccall((:make_uint3, vector_functions), uint3, (UInt32, UInt32, UInt32), x, y, z)
end

function make_int4(x::Cint, y::Cint, z::Cint, w::Cint)
    ccall((:make_int4, vector_functions), int4, (Cint, Cint, Cint, Cint), x, y, z, w)
end

function make_uint4(x::UInt32, y::UInt32, z::UInt32, w::UInt32)
    ccall((:make_uint4, vector_functions), uint4, (UInt32, UInt32, UInt32, UInt32), x, y, z, w)
end

function make_long1(x::Clong)
    ccall((:make_long1, vector_functions), long1, (Clong,), x)
end

function make_ulong1(x::Culong)
    ccall((:make_ulong1, vector_functions), ulong1, (Culong,), x)
end

function make_long2(x::Clong, y::Clong)
    ccall((:make_long2, vector_functions), long2, (Clong, Clong), x, y)
end

function make_ulong2(x::Culong, y::Culong)
    ccall((:make_ulong2, vector_functions), ulong2, (Culong, Culong), x, y)
end

function make_long3(x::Clong, y::Clong, z::Clong)
    ccall((:make_long3, vector_functions), long3, (Clong, Clong, Clong), x, y, z)
end

function make_ulong3(x::Culong, y::Culong, z::Culong)
    ccall((:make_ulong3, vector_functions), ulong3, (Culong, Culong, Culong), x, y, z)
end

function make_long4(x::Clong, y::Clong, z::Clong, w::Clong)
    ccall((:make_long4, vector_functions), long4, (Clong, Clong, Clong, Clong), x, y, z, w)
end

function make_ulong4(x::Culong, y::Culong, z::Culong, w::Culong)
    ccall((:make_ulong4, vector_functions), ulong4, (Culong, Culong, Culong, Culong), x, y, z, w)
end

function make_float1(x::Cfloat)
    ccall((:make_float1, vector_functions), float1, (Cfloat,), x)
end

function make_float2(x::Cfloat, y::Cfloat)
    ccall((:make_float2, vector_functions), float2, (Cfloat, Cfloat), x, y)
end

function make_float3(x::Cfloat, y::Cfloat, z::Cfloat)
    ccall((:make_float3, vector_functions), float3, (Cfloat, Cfloat, Cfloat), x, y, z)
end

function make_float4(x::Cfloat, y::Cfloat, z::Cfloat, w::Cfloat)
    ccall((:make_float4, vector_functions), float4, (Cfloat, Cfloat, Cfloat, Cfloat), x, y, z, w)
end

function make_longlong1(x::Clonglong)
    ccall((:make_longlong1, vector_functions), longlong1, (Clonglong,), x)
end

function make_ulonglong1(x::Culonglong)
    ccall((:make_ulonglong1, vector_functions), ulonglong1, (Culonglong,), x)
end

function make_longlong2(x::Clonglong, y::Clonglong)
    ccall((:make_longlong2, vector_functions), longlong2, (Clonglong, Clonglong), x, y)
end

function make_ulonglong2(x::Culonglong, y::Culonglong)
    ccall((:make_ulonglong2, vector_functions), ulonglong2, (Culonglong, Culonglong), x, y)
end

function make_longlong3(x::Clonglong, y::Clonglong, z::Clonglong)
    ccall((:make_longlong3, vector_functions), longlong3, (Clonglong, Clonglong, Clonglong), x, y, z)
end

function make_ulonglong3(x::Culonglong, y::Culonglong, z::Culonglong)
    ccall((:make_ulonglong3, vector_functions), ulonglong3, (Culonglong, Culonglong, Culonglong), x, y, z)
end

function make_longlong4(x::Clonglong, y::Clonglong, z::Clonglong, w::Clonglong)
    ccall((:make_longlong4, vector_functions), longlong4, (Clonglong, Clonglong, Clonglong, Clonglong), x, y, z, w)
end

function make_ulonglong4(x::Culonglong, y::Culonglong, z::Culonglong, w::Culonglong)
    ccall((:make_ulonglong4, vector_functions), ulonglong4, (Culonglong, Culonglong, Culonglong, Culonglong), x, y, z, w)
end

function make_double1(x::Cdouble)
    ccall((:make_double1, vector_functions), double1, (Cdouble,), x)
end

function make_double2(x::Cdouble, y::Cdouble)
    ccall((:make_double2, vector_functions), double2, (Cdouble, Cdouble), x, y)
end

function make_double3(x::Cdouble, y::Cdouble, z::Cdouble)
    ccall((:make_double3, vector_functions), double3, (Cdouble, Cdouble, Cdouble), x, y, z)
end

function make_double4(x::Cdouble, y::Cdouble, z::Cdouble, w::Cdouble)
    ccall((:make_double4, vector_functions), double4, (Cdouble, Cdouble, Cdouble, Cdouble), x, y, z, w)
end

function cudnnGetVersion()
    ccall((:cudnnGetVersion, cudnn), Cint, ())
end

function cudnnGetErrorString(status::cudnnStatus_t)
    ccall((:cudnnGetErrorString, cudnn), Cstring, (cudnnStatus_t,), status)
end

function cudnnCreate(handle)
    ccall((:cudnnCreate, cudnn), cudnnStatus_t, (Ptr{cudnnHandle_t},), handle)
end

function cudnnDestroy(handle::cudnnHandle_t)
    ccall((:cudnnDestroy, cudnn), cudnnStatus_t, (cudnnHandle_t,), handle)
end

function cudnnSetStream(handle::cudnnHandle_t, streamId::cudaStream_t)
    ccall((:cudnnSetStream, cudnn), cudnnStatus_t, (cudnnHandle_t, cudaStream_t), handle, streamId)
end

function cudnnGetStream(handle::cudnnHandle_t, streamId)
    ccall((:cudnnGetStream, cudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{cudaStream_t}), handle, streamId)
end

function cudnnCreateTensorDescriptor(tensorDesc)
    ccall((:cudnnCreateTensorDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnTensorDescriptor_t},), tensorDesc)
end

function cudnnSetTensor4dDescriptor(tensorDesc::cudnnTensorDescriptor_t, format::cudnnTensorFormat_t, dataType::cudnnDataType_t, n::Cint, c::Cint, h::Cint, w::Cint)
    ccall((:cudnnSetTensor4dDescriptor, cudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, Cint, Cint, Cint, Cint), tensorDesc, format, dataType, n, c, h, w)
end

function cudnnSetTensor4dDescriptorEx(tensorDesc::cudnnTensorDescriptor_t, dataType::cudnnDataType_t, n::Cint, c::Cint, h::Cint, w::Cint, nStride::Cint, cStride::Cint, hStride::Cint, wStride::Cint)
    ccall((:cudnnSetTensor4dDescriptorEx, cudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
end

function cudnnGetTensor4dDescriptor(tensorDesc::cudnnTensorDescriptor_t, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
    ccall((:cudnnGetTensor4dDescriptor, cudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
end

function cudnnSetTensorNdDescriptor(tensorDesc::cudnnTensorDescriptor_t, dataType::cudnnDataType_t, nbDims::Cint, dimA, strideA)
    ccall((:cudnnSetTensorNdDescriptor, cudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnDataType_t, Cint, Ptr{Cint}, Ptr{Cint}), tensorDesc, dataType, nbDims, dimA, strideA)
end

function cudnnGetTensorNdDescriptor(tensorDesc::cudnnTensorDescriptor_t, nbDimsRequested::Cint, dataType, nbDims, dimA, strideA)
    ccall((:cudnnGetTensorNdDescriptor, cudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA)
end

function cudnnDestroyTensorDescriptor(tensorDesc::cudnnTensorDescriptor_t)
    ccall((:cudnnDestroyTensorDescriptor, cudnn), cudnnStatus_t, (cudnnTensorDescriptor_t,), tensorDesc)
end

function cudnnTransformTensor(handle::cudnnHandle_t, alpha, xDesc::cudnnTensorDescriptor_t, x, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnTransformTensor, cudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnAddTensor(handle::cudnnHandle_t, alpha, aDesc::cudnnTensorDescriptor_t, A, beta, cDesc::cudnnTensorDescriptor_t, C)
    ccall((:cudnnAddTensor, cudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, aDesc, A, beta, cDesc, C)
end

function cudnnCreateOpTensorDescriptor(opTensorDesc)
    ccall((:cudnnCreateOpTensorDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnOpTensorDescriptor_t},), opTensorDesc)
end

function cudnnSetOpTensorDescriptor(opTensorDesc::cudnnOpTensorDescriptor_t, opTensorOp::cudnnOpTensorOp_t, opTensorCompType::cudnnDataType_t, opTensorNanOpt::cudnnNanPropagation_t)
    ccall((:cudnnSetOpTensorDescriptor, cudnn), cudnnStatus_t, (cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t, cudnnDataType_t, cudnnNanPropagation_t), opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
end

function cudnnGetOpTensorDescriptor(opTensorDesc::cudnnOpTensorDescriptor_t, opTensorOp, opTensorCompType, opTensorNanOpt)
    ccall((:cudnnGetOpTensorDescriptor, cudnn), cudnnStatus_t, (cudnnOpTensorDescriptor_t, Ptr{cudnnOpTensorOp_t}, Ptr{cudnnDataType_t}, Ptr{cudnnNanPropagation_t}), opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt)
end

function cudnnDestroyOpTensorDescriptor(opTensorDesc::cudnnOpTensorDescriptor_t)
    ccall((:cudnnDestroyOpTensorDescriptor, cudnn), cudnnStatus_t, (cudnnOpTensorDescriptor_t,), opTensorDesc)
end

function cudnnOpTensor(handle::cudnnHandle_t, opTensorDesc::cudnnOpTensorDescriptor_t, alpha1, aDesc::cudnnTensorDescriptor_t, A, alpha2, bDesc::cudnnTensorDescriptor_t, B, beta, cDesc::cudnnTensorDescriptor_t, C)
    ccall((:cudnnOpTensor, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnOpTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C)
end

function cudnnSetTensor(handle::cudnnHandle_t, yDesc::cudnnTensorDescriptor_t, y, valuePtr)
    ccall((:cudnnSetTensor, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}), handle, yDesc, y, valuePtr)
end

function cudnnScaleTensor(handle::cudnnHandle_t, yDesc::cudnnTensorDescriptor_t, y, alpha)
    ccall((:cudnnScaleTensor, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}), handle, yDesc, y, alpha)
end

function cudnnCreateFilterDescriptor(filterDesc)
    ccall((:cudnnCreateFilterDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnFilterDescriptor_t},), filterDesc)
end

function cudnnSetFilter4dDescriptor(filterDesc::cudnnFilterDescriptor_t, dataType::cudnnDataType_t, format::cudnnTensorFormat_t, k::Cint, c::Cint, h::Cint, w::Cint)
    ccall((:cudnnSetFilter4dDescriptor, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Cint, Cint, Cint), filterDesc, dataType, format, k, c, h, w)
end

function cudnnGetFilter4dDescriptor(filterDesc::cudnnFilterDescriptor_t, dataType, format, k, c, h, w)
    ccall((:cudnnGetFilter4dDescriptor, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), filterDesc, dataType, format, k, c, h, w)
end

function cudnnSetFilterNdDescriptor(filterDesc::cudnnFilterDescriptor_t, dataType::cudnnDataType_t, format::cudnnTensorFormat_t, nbDims::Cint, filterDimA)
    ccall((:cudnnSetFilterNdDescriptor, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Ptr{Cint}), filterDesc, dataType, format, nbDims, filterDimA)
end

function cudnnGetFilterNdDescriptor(filterDesc::cudnnFilterDescriptor_t, nbDimsRequested::Cint, dataType, format, nbDims, filterDimA)
    ccall((:cudnnGetFilterNdDescriptor, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}), filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
end

function cudnnDestroyFilterDescriptor(filterDesc::cudnnFilterDescriptor_t)
    ccall((:cudnnDestroyFilterDescriptor, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t,), filterDesc)
end

function cudnnCreateConvolutionDescriptor(convDesc)
    ccall((:cudnnCreateConvolutionDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnConvolutionDescriptor_t},), convDesc)
end

function cudnnSetConvolution2dDescriptor(convDesc::cudnnConvolutionDescriptor_t, pad_h::Cint, pad_w::Cint, u::Cint, v::Cint, upscalex::Cint, upscaley::Cint, mode::cudnnConvolutionMode_t)
    ccall((:cudnnSetConvolution2dDescriptor, cudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Cint, Cint, Cint, Cint, Cint, cudnnConvolutionMode_t), convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode)
end

function cudnnSetConvolution2dDescriptor_v5(convDesc::cudnnConvolutionDescriptor_t, pad_h::Cint, pad_w::Cint, u::Cint, v::Cint, upscalex::Cint, upscaley::Cint, mode::cudnnConvolutionMode_t, dataType::cudnnDataType_t)
    ccall((:cudnnSetConvolution2dDescriptor_v5, cudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Cint, Cint, Cint, Cint, Cint, cudnnConvolutionMode_t, cudnnDataType_t), convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode, dataType)
end

function cudnnGetConvolution2dDescriptor(convDesc::cudnnConvolutionDescriptor_t, pad_h, pad_w, u, v, upscalex, upscaley, mode)
    ccall((:cudnnGetConvolution2dDescriptor, cudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cudnnConvolutionMode_t}), convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode)
end

function cudnnGetConvolution2dDescriptor_v5(convDesc::cudnnConvolutionDescriptor_t, pad_h, pad_w, u, v, upscalex, upscaley, mode, dataType)
    ccall((:cudnnGetConvolution2dDescriptor_v5, cudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cudnnConvolutionMode_t}, Ptr{cudnnDataType_t}), convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode, dataType)
end

function cudnnGetConvolution2dForwardOutputDim(convDesc::cudnnConvolutionDescriptor_t, inputTensorDesc::cudnnTensorDescriptor_t, filterDesc::cudnnFilterDescriptor_t, n, c, h, w)
    ccall((:cudnnGetConvolution2dForwardOutputDim, cudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), convDesc, inputTensorDesc, filterDesc, n, c, h, w)
end

function cudnnSetConvolutionNdDescriptor(convDesc::cudnnConvolutionDescriptor_t, arrayLength::Cint, padA, filterStrideA, upscaleA, mode::cudnnConvolutionMode_t, dataType::cudnnDataType_t)
    ccall((:cudnnSetConvolutionNdDescriptor, cudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, cudnnConvolutionMode_t, cudnnDataType_t), convDesc, arrayLength, padA, filterStrideA, upscaleA, mode, dataType)
end

function cudnnGetConvolutionNdDescriptor(convDesc::cudnnConvolutionDescriptor_t, arrayLengthRequested::Cint, arrayLength, padA, strideA, upscaleA, mode, dataType)
    ccall((:cudnnGetConvolutionNdDescriptor, cudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{cudnnConvolutionMode_t}, Ptr{cudnnDataType_t}), convDesc, arrayLengthRequested, arrayLength, padA, strideA, upscaleA, mode, dataType)
end

function cudnnGetConvolutionNdForwardOutputDim(convDesc::cudnnConvolutionDescriptor_t, inputTensorDesc::cudnnTensorDescriptor_t, filterDesc::cudnnFilterDescriptor_t, nbDims::Cint, tensorOuputDimA)
    ccall((:cudnnGetConvolutionNdForwardOutputDim, cudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint}), convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA)
end

function cudnnDestroyConvolutionDescriptor(convDesc::cudnnConvolutionDescriptor_t)
    ccall((:cudnnDestroyConvolutionDescriptor, cudnn), cudnnStatus_t, (cudnnConvolutionDescriptor_t,), convDesc)
end

function cudnnFindConvolutionForwardAlgorithm(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t, wDesc::cudnnFilterDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, yDesc::cudnnTensorDescriptor_t, requestedAlgoCount::Cint, returnedAlgoCount, perfResults)
    ccall((:cudnnFindConvolutionForwardAlgorithm, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionFwdAlgoPerf_t}), handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
end

function cudnnFindConvolutionForwardAlgorithmEx(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t, x, wDesc::cudnnFilterDescriptor_t, w, convDesc::cudnnConvolutionDescriptor_t, yDesc::cudnnTensorDescriptor_t, y, requestedAlgoCount::Cint, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes::Cint)
    ccall((:cudnnFindConvolutionForwardAlgorithmEx, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Ptr{Void}, Cint, Ptr{Cint}, Ptr{cudnnConvolutionFwdAlgoPerf_t}, Ptr{Void}, Cint), handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

function cudnnGetConvolutionForwardAlgorithm(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t, wDesc::cudnnFilterDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, yDesc::cudnnTensorDescriptor_t, preference::cudnnConvolutionFwdPreference_t, memoryLimitInBytes::Cint, algo)
    ccall((:cudnnGetConvolutionForwardAlgorithm, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionFwdPreference_t, Cint, Ptr{cudnnConvolutionFwdAlgo_t}), handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo)
end

function cudnnGetConvolutionForwardWorkspaceSize(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t, wDesc::cudnnFilterDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, yDesc::cudnnTensorDescriptor_t, algo::cudnnConvolutionFwdAlgo_t, sizeInBytes)
    ccall((:cudnnGetConvolutionForwardWorkspaceSize, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t, Ptr{Cint}), handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes)
end

function cudnnConvolutionForward(handle::cudnnHandle_t, alpha, xDesc::cudnnTensorDescriptor_t, x, wDesc::cudnnFilterDescriptor_t, w, convDesc::cudnnConvolutionDescriptor_t, algo::cudnnConvolutionFwdAlgo_t, workSpace, workSpaceSizeInBytes::Cint, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnConvolutionForward, cudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, Ptr{Void}, Cint, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y)
end

function cudnnConvolutionBackwardBias(handle::cudnnHandle_t, alpha, dyDesc::cudnnTensorDescriptor_t, dy, beta, dbDesc::cudnnTensorDescriptor_t, db)
    ccall((:cudnnConvolutionBackwardBias, cudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, dyDesc, dy, beta, dbDesc, db)
end

function cudnnFindConvolutionBackwardFilterAlgorithm(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t, dyDesc::cudnnTensorDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, dwDesc::cudnnFilterDescriptor_t, requestedAlgoCount::Cint, returnedAlgoCount, perfResults)
    ccall((:cudnnFindConvolutionBackwardFilterAlgorithm, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}), handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
end

function cudnnFindConvolutionBackwardFilterAlgorithmEx(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t, x, dyDesc::cudnnTensorDescriptor_t, y, convDesc::cudnnConvolutionDescriptor_t, dwDesc::cudnnFilterDescriptor_t, dw, requestedAlgoCount::Cint, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes::Cint)
    ccall((:cudnnFindConvolutionBackwardFilterAlgorithmEx, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, Ptr{Void}, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}, Ptr{Void}, Cint), handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

function cudnnGetConvolutionBackwardFilterAlgorithm(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t, dyDesc::cudnnTensorDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, dwDesc::cudnnFilterDescriptor_t, preference::cudnnConvolutionBwdFilterPreference_t, memoryLimitInBytes::Cint, algo)
    ccall((:cudnnGetConvolutionBackwardFilterAlgorithm, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionBwdFilterPreference_t, Cint, Ptr{cudnnConvolutionBwdFilterAlgo_t}), handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo)
end

function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t, dyDesc::cudnnTensorDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, gradDesc::cudnnFilterDescriptor_t, algo::cudnnConvolutionBwdFilterAlgo_t, sizeInBytes)
    ccall((:cudnnGetConvolutionBackwardFilterWorkspaceSize, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, Ptr{Cint}), handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes)
end

function cudnnConvolutionBackwardFilter(handle::cudnnHandle_t, alpha, xDesc::cudnnTensorDescriptor_t, x, dyDesc::cudnnTensorDescriptor_t, dy, convDesc::cudnnConvolutionDescriptor_t, algo::cudnnConvolutionBwdFilterAlgo_t, workSpace, workSpaceSizeInBytes::Cint, beta, dwDesc::cudnnFilterDescriptor_t, dw)
    ccall((:cudnnConvolutionBackwardFilter, cudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, Ptr{Void}, Cint, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}), handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw)
end

function cudnnFindConvolutionBackwardDataAlgorithm(handle::cudnnHandle_t, wDesc::cudnnFilterDescriptor_t, dyDesc::cudnnTensorDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, dxDesc::cudnnTensorDescriptor_t, requestedAlgoCount::Cint, returnedAlgoCount, perfResults)
    ccall((:cudnnFindConvolutionBackwardDataAlgorithm, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdDataAlgoPerf_t}), handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults)
end

function cudnnFindConvolutionBackwardDataAlgorithmEx(handle::cudnnHandle_t, wDesc::cudnnFilterDescriptor_t, w, dyDesc::cudnnTensorDescriptor_t, dy, convDesc::cudnnConvolutionDescriptor_t, dxDesc::cudnnTensorDescriptor_t, dx, requestedAlgoCount::Cint, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes::Cint)
    ccall((:cudnnFindConvolutionBackwardDataAlgorithmEx, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, Ptr{Void}, Cint, Ptr{Cint}, Ptr{cudnnConvolutionBwdDataAlgoPerf_t}, Ptr{Void}, Cint), handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes)
end

function cudnnGetConvolutionBackwardDataAlgorithm(handle::cudnnHandle_t, wDesc::cudnnFilterDescriptor_t, dyDesc::cudnnTensorDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, dxDesc::cudnnTensorDescriptor_t, preference::cudnnConvolutionBwdDataPreference_t, memoryLimitInBytes::Cint, algo)
    ccall((:cudnnGetConvolutionBackwardDataAlgorithm, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionBwdDataPreference_t, Cint, Ptr{cudnnConvolutionBwdDataAlgo_t}), handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo)
end

function cudnnGetConvolutionBackwardDataWorkspaceSize(handle::cudnnHandle_t, wDesc::cudnnFilterDescriptor_t, dyDesc::cudnnTensorDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, dxDesc::cudnnTensorDescriptor_t, algo::cudnnConvolutionBwdDataAlgo_t, sizeInBytes)
    ccall((:cudnnGetConvolutionBackwardDataWorkspaceSize, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionBwdDataAlgo_t, Ptr{Cint}), handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes)
end

function cudnnConvolutionBackwardData(handle::cudnnHandle_t, alpha, wDesc::cudnnFilterDescriptor_t, w, dyDesc::cudnnTensorDescriptor_t, dy, convDesc::cudnnConvolutionDescriptor_t, algo::cudnnConvolutionBwdDataAlgo_t, workSpace, workSpaceSizeInBytes::Cint, beta, dxDesc::cudnnTensorDescriptor_t, dx)
    ccall((:cudnnConvolutionBackwardData, cudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, Ptr{Void}, Cint, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx)
end

function cudnnIm2Col(handle::cudnnHandle_t, xDesc::cudnnTensorDescriptor_t, x, wDesc::cudnnFilterDescriptor_t, convDesc::cudnnConvolutionDescriptor_t, colBuffer)
    ccall((:cudnnIm2Col, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t, Ptr{Void}), handle, xDesc, x, wDesc, convDesc, colBuffer)
end

function cudnnSoftmaxForward(handle::cudnnHandle_t, algo::cudnnSoftmaxAlgorithm_t, mode::cudnnSoftmaxMode_t, alpha, xDesc::cudnnTensorDescriptor_t, x, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnSoftmaxForward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, algo, mode, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnSoftmaxBackward(handle::cudnnHandle_t, algo::cudnnSoftmaxAlgorithm_t, mode::cudnnSoftmaxMode_t, alpha, yDesc::cudnnTensorDescriptor_t, y, dyDesc::cudnnTensorDescriptor_t, dy, beta, dxDesc::cudnnTensorDescriptor_t, dx)
    ccall((:cudnnSoftmaxBackward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx)
end

function cudnnCreatePoolingDescriptor(poolingDesc)
    ccall((:cudnnCreatePoolingDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnPoolingDescriptor_t},), poolingDesc)
end

function cudnnSetPooling2dDescriptor(poolingDesc::cudnnPoolingDescriptor_t, mode::cudnnPoolingMode_t, maxpoolingNanOpt::cudnnNanPropagation_t, windowHeight::Cint, windowWidth::Cint, verticalPadding::Cint, horizontalPadding::Cint, verticalStride::Cint, horizontalStride::Cint)
    ccall((:cudnnSetPooling2dDescriptor, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint, Cint, Cint, Cint, Cint, Cint), poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

function cudnnGetPooling2dDescriptor(poolingDesc::cudnnPoolingDescriptor_t, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    ccall((:cudnnGetPooling2dDescriptor, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Ptr{cudnnPoolingMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

function cudnnSetPoolingNdDescriptor(poolingDesc::cudnnPoolingDescriptor_t, mode::cudnnPoolingMode_t, maxpoolingNanOpt::cudnnNanPropagation_t, nbDims::Cint, windowDimA, paddingA, strideA)
    ccall((:cudnnSetPoolingNdDescriptor, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
end

function cudnnGetPoolingNdDescriptor(poolingDesc::cudnnPoolingDescriptor_t, nbDimsRequested::Cint, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
    ccall((:cudnnGetPoolingNdDescriptor, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Cint, Ptr{cudnnPoolingMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
end

function cudnnGetPoolingNdForwardOutputDim(poolingDesc::cudnnPoolingDescriptor_t, inputTensorDesc::cudnnTensorDescriptor_t, nbDims::Cint, outputTensorDimA)
    ccall((:cudnnGetPoolingNdForwardOutputDim, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Cint, Ptr{Cint}), poolingDesc, inputTensorDesc, nbDims, outputTensorDimA)
end

function cudnnGetPooling2dForwardOutputDim(poolingDesc::cudnnPoolingDescriptor_t, inputTensorDesc::cudnnTensorDescriptor_t, n, c, h, w)
    ccall((:cudnnGetPooling2dForwardOutputDim, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, inputTensorDesc, n, c, h, w)
end

function cudnnDestroyPoolingDescriptor(poolingDesc::cudnnPoolingDescriptor_t)
    ccall((:cudnnDestroyPoolingDescriptor, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t,), poolingDesc)
end

function cudnnPoolingForward(handle::cudnnHandle_t, poolingDesc::cudnnPoolingDescriptor_t, alpha, xDesc::cudnnTensorDescriptor_t, x, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnPoolingForward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnPoolingBackward(handle::cudnnHandle_t, poolingDesc::cudnnPoolingDescriptor_t, alpha, yDesc::cudnnTensorDescriptor_t, y, dyDesc::cudnnTensorDescriptor_t, dy, xDesc::cudnnTensorDescriptor_t, x, beta, dxDesc::cudnnTensorDescriptor_t, dx)
    ccall((:cudnnPoolingBackward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnPoolingDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnCreateActivationDescriptor(activationDesc)
    ccall((:cudnnCreateActivationDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnActivationDescriptor_t},), activationDesc)
end

function cudnnSetActivationDescriptor(activationDesc::cudnnActivationDescriptor_t, mode::cudnnActivationMode_t, reluNanOpt::cudnnNanPropagation_t, reluCeiling::Cdouble)
    ccall((:cudnnSetActivationDescriptor, cudnn), cudnnStatus_t, (cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnNanPropagation_t, Cdouble), activationDesc, mode, reluNanOpt, reluCeiling)
end

function cudnnGetActivationDescriptor(activationDesc::cudnnActivationDescriptor_t, mode, reluNanOpt, reluCeiling)
    ccall((:cudnnGetActivationDescriptor, cudnn), cudnnStatus_t, (cudnnActivationDescriptor_t, Ptr{cudnnActivationMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cdouble}), activationDesc, mode, reluNanOpt, reluCeiling)
end

function cudnnDestroyActivationDescriptor(activationDesc::cudnnActivationDescriptor_t)
    ccall((:cudnnDestroyActivationDescriptor, cudnn), cudnnStatus_t, (cudnnActivationDescriptor_t,), activationDesc)
end

function cudnnActivationForward(handle::cudnnHandle_t, activationDesc::cudnnActivationDescriptor_t, alpha, xDesc::cudnnTensorDescriptor_t, x, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnActivationForward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnActivationBackward(handle::cudnnHandle_t, activationDesc::cudnnActivationDescriptor_t, alpha, yDesc::cudnnTensorDescriptor_t, y, dyDesc::cudnnTensorDescriptor_t, dy, xDesc::cudnnTensorDescriptor_t, x, beta, dxDesc::cudnnTensorDescriptor_t, dx)
    ccall((:cudnnActivationBackward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnCreateLRNDescriptor(normDesc)
    ccall((:cudnnCreateLRNDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnLRNDescriptor_t},), normDesc)
end

function cudnnSetLRNDescriptor(normDesc::cudnnLRNDescriptor_t, lrnN::UInt32, lrnAlpha::Cdouble, lrnBeta::Cdouble, lrnK::Cdouble)
    ccall((:cudnnSetLRNDescriptor, cudnn), cudnnStatus_t, (cudnnLRNDescriptor_t, UInt32, Cdouble, Cdouble, Cdouble), normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
end

function cudnnGetLRNDescriptor(normDesc::cudnnLRNDescriptor_t, lrnN, lrnAlpha, lrnBeta, lrnK)
    ccall((:cudnnGetLRNDescriptor, cudnn), cudnnStatus_t, (cudnnLRNDescriptor_t, Ptr{UInt32}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), normDesc, lrnN, lrnAlpha, lrnBeta, lrnK)
end

function cudnnDestroyLRNDescriptor(lrnDesc::cudnnLRNDescriptor_t)
    ccall((:cudnnDestroyLRNDescriptor, cudnn), cudnnStatus_t, (cudnnLRNDescriptor_t,), lrnDesc)
end

function cudnnLRNCrossChannelForward(handle::cudnnHandle_t, normDesc::cudnnLRNDescriptor_t, lrnMode::cudnnLRNMode_t, alpha, xDesc::cudnnTensorDescriptor_t, x, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnLRNCrossChannelForward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnLRNCrossChannelBackward(handle::cudnnHandle_t, normDesc::cudnnLRNDescriptor_t, lrnMode::cudnnLRNMode_t, alpha, yDesc::cudnnTensorDescriptor_t, y, dyDesc::cudnnTensorDescriptor_t, dy, xDesc::cudnnTensorDescriptor_t, x, beta, dxDesc::cudnnTensorDescriptor_t, dx)
    ccall((:cudnnLRNCrossChannelBackward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnDivisiveNormalizationForward(handle::cudnnHandle_t, normDesc::cudnnLRNDescriptor_t, mode::cudnnDivNormMode_t, alpha, xDesc::cudnnTensorDescriptor_t, x, means, temp, temp2, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnDivisiveNormalizationForward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y)
end

function cudnnDivisiveNormalizationBackward(handle::cudnnHandle_t, normDesc::cudnnLRNDescriptor_t, mode::cudnnDivNormMode_t, alpha, xDesc::cudnnTensorDescriptor_t, x, means, dy, temp, temp2, beta, dXdMeansDesc::cudnnTensorDescriptor_t, dx, dMeans)
    ccall((:cudnnDivisiveNormalizationBackward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}), handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans)
end

function cudnnDeriveBNTensorDescriptor(derivedBnDesc::cudnnTensorDescriptor_t, xDesc::cudnnTensorDescriptor_t, mode::cudnnBatchNormMode_t)
    ccall((:cudnnDeriveBNTensorDescriptor, cudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnBatchNormMode_t), derivedBnDesc, xDesc, mode)
end

function cudnnBatchNormalizationForwardTraining(handle::cudnnHandle_t, mode::cudnnBatchNormMode_t, alpha, beta, xDesc::cudnnTensorDescriptor_t, x, yDesc::cudnnTensorDescriptor_t, y, bnScaleBiasMeanVarDesc::cudnnTensorDescriptor_t, bnScale, bnBias, exponentialAverageFactor::Cdouble, resultRunningMean, resultRunningVariance, epsilon::Cdouble, resultSaveMean, resultSaveInvVariance)
    ccall((:cudnnBatchNormalizationForwardTraining, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Cdouble, Ptr{Void}, Ptr{Void}, Cdouble, Ptr{Void}, Ptr{Void}), handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance)
end

function cudnnBatchNormalizationForwardInference(handle::cudnnHandle_t, mode::cudnnBatchNormMode_t, alpha, beta, xDesc::cudnnTensorDescriptor_t, x, yDesc::cudnnTensorDescriptor_t, y, bnScaleBiasMeanVarDesc::cudnnTensorDescriptor_t, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon::Cdouble)
    ccall((:cudnnBatchNormalizationForwardInference, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Cdouble), handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
end

function cudnnBatchNormalizationBackward(handle::cudnnHandle_t, mode::cudnnBatchNormMode_t, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc::cudnnTensorDescriptor_t, x, dyDesc::cudnnTensorDescriptor_t, dy, dxDesc::cudnnTensorDescriptor_t, dx, dBnScaleBiasDesc::cudnnTensorDescriptor_t, bnScale, dBnScaleResult, dBnBiasResult, epsilon::Cdouble, savedMean, savedInvVariance)
    ccall((:cudnnBatchNormalizationBackward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnBatchNormMode_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Cdouble, Ptr{Void}, Ptr{Void}), handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance)
end

function cudnnCreateSpatialTransformerDescriptor(stDesc)
    ccall((:cudnnCreateSpatialTransformerDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnSpatialTransformerDescriptor_t},), stDesc)
end

function cudnnSetSpatialTransformerNdDescriptor(stDesc::cudnnSpatialTransformerDescriptor_t, samplerType::cudnnSamplerType_t, dataType::cudnnDataType_t, nbDims::Cint, dimA)
    ccall((:cudnnSetSpatialTransformerNdDescriptor, cudnn), cudnnStatus_t, (cudnnSpatialTransformerDescriptor_t, cudnnSamplerType_t, cudnnDataType_t, Cint, Ptr{Cint}), stDesc, samplerType, dataType, nbDims, dimA)
end

function cudnnDestroySpatialTransformerDescriptor(stDesc::cudnnSpatialTransformerDescriptor_t)
    ccall((:cudnnDestroySpatialTransformerDescriptor, cudnn), cudnnStatus_t, (cudnnSpatialTransformerDescriptor_t,), stDesc)
end

function cudnnSpatialTfGridGeneratorForward(handle::cudnnHandle_t, stDesc::cudnnSpatialTransformerDescriptor_t, theta, grid)
    ccall((:cudnnSpatialTfGridGeneratorForward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Void}, Ptr{Void}), handle, stDesc, theta, grid)
end

function cudnnSpatialTfGridGeneratorBackward(handle::cudnnHandle_t, stDesc::cudnnSpatialTransformerDescriptor_t, dgrid, dtheta)
    ccall((:cudnnSpatialTfGridGeneratorBackward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Void}, Ptr{Void}), handle, stDesc, dgrid, dtheta)
end

function cudnnSpatialTfSamplerForward(handle::cudnnHandle_t, stDesc::cudnnSpatialTransformerDescriptor_t, alpha, xDesc::cudnnTensorDescriptor_t, x, grid, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnSpatialTfSamplerForward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y)
end

function cudnnSpatialTfSamplerBackward(handle::cudnnHandle_t, stDesc::cudnnSpatialTransformerDescriptor_t, alpha, xDesc::cudnnTensorDescriptor_t, x, beta, dxDesc::cudnnTensorDescriptor_t, dx, alphaDgrid, dyDesc::cudnnTensorDescriptor_t, dy, grid, betaDgrid, dgrid)
    ccall((:cudnnSpatialTfSamplerBackward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}), handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid)
end

function cudnnCreateDropoutDescriptor(dropoutDesc)
    ccall((:cudnnCreateDropoutDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnDropoutDescriptor_t},), dropoutDesc)
end

function cudnnDestroyDropoutDescriptor(dropoutDesc::cudnnDropoutDescriptor_t)
    ccall((:cudnnDestroyDropoutDescriptor, cudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t,), dropoutDesc)
end

function cudnnDropoutGetStatesSize(handle::cudnnHandle_t, sizeInBytes)
    ccall((:cudnnDropoutGetStatesSize, cudnn), cudnnStatus_t, (cudnnHandle_t, Ptr{Cint}), handle, sizeInBytes)
end

function cudnnDropoutGetReserveSpaceSize(xdesc::cudnnTensorDescriptor_t, sizeInBytes)
    ccall((:cudnnDropoutGetReserveSpaceSize, cudnn), cudnnStatus_t, (cudnnTensorDescriptor_t, Ptr{Cint}), xdesc, sizeInBytes)
end

function cudnnSetDropoutDescriptor(dropoutDesc::cudnnDropoutDescriptor_t, handle::cudnnHandle_t, dropout::Cfloat, states, stateSizeInBytes::Cint, seed::Culonglong)
    ccall((:cudnnSetDropoutDescriptor, cudnn), cudnnStatus_t, (cudnnDropoutDescriptor_t, cudnnHandle_t, Cfloat, Ptr{Void}, Cint, Culonglong), dropoutDesc, handle, dropout, states, stateSizeInBytes, seed)
end

function cudnnDropoutForward(handle::cudnnHandle_t, dropoutDesc::cudnnDropoutDescriptor_t, xdesc::cudnnTensorDescriptor_t, x, ydesc::cudnnTensorDescriptor_t, y, reserveSpace, reserveSpaceSizeInBytes::Cint)
    ccall((:cudnnDropoutForward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Cint), handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnDropoutBackward(handle::cudnnHandle_t, dropoutDesc::cudnnDropoutDescriptor_t, dydesc::cudnnTensorDescriptor_t, dy, dxdesc::cudnnTensorDescriptor_t, dx, reserveSpace, reserveSpaceSizeInBytes::Cint)
    ccall((:cudnnDropoutBackward, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnDropoutDescriptor_t, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Cint), handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnCreateRNNDescriptor(rnnDesc)
    ccall((:cudnnCreateRNNDescriptor, cudnn), cudnnStatus_t, (Ptr{cudnnRNNDescriptor_t},), rnnDesc)
end

function cudnnDestroyRNNDescriptor(rnnDesc::cudnnRNNDescriptor_t)
    ccall((:cudnnDestroyRNNDescriptor, cudnn), cudnnStatus_t, (cudnnRNNDescriptor_t,), rnnDesc)
end

function cudnnSetRNNDescriptor(rnnDesc::cudnnRNNDescriptor_t, hiddenSize::Cint, numLayers::Cint, dropoutDesc::cudnnDropoutDescriptor_t, inputMode::cudnnRNNInputMode_t, direction::cudnnDirectionMode_t, mode::cudnnRNNMode_t, dataType::cudnnDataType_t)
    ccall((:cudnnSetRNNDescriptor, cudnn), cudnnStatus_t, (cudnnRNNDescriptor_t, Cint, Cint, cudnnDropoutDescriptor_t, cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnDataType_t), rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, dataType)
end

function cudnnGetRNNWorkspaceSize(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint, xDesc, sizeInBytes)
    ccall((:cudnnGetRNNWorkspaceSize, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Cint}), handle, rnnDesc, seqLength, xDesc, sizeInBytes)
end

function cudnnGetRNNTrainingReserveSize(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint, xDesc, sizeInBytes)
    ccall((:cudnnGetRNNTrainingReserveSize, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Cint}), handle, rnnDesc, seqLength, xDesc, sizeInBytes)
end

function cudnnGetRNNParamsSize(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t, xDesc::cudnnTensorDescriptor_t, sizeInBytes, dataType::cudnnDataType_t)
    ccall((:cudnnGetRNNParamsSize, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, cudnnTensorDescriptor_t, Ptr{Cint}, cudnnDataType_t), handle, rnnDesc, xDesc, sizeInBytes, dataType)
end

function cudnnGetRNNLinLayerMatrixParams(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t, layer::Cint, xDesc::cudnnTensorDescriptor_t, wDesc::cudnnFilterDescriptor_t, w, linLayerID::Cint, linLayerMatDesc::cudnnFilterDescriptor_t, linLayerMat)
    ccall((:cudnnGetRNNLinLayerMatrixParams, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Ptr{Void}, Cint, cudnnFilterDescriptor_t, Ptr{Ptr{Void}}), handle, rnnDesc, layer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat)
end

function cudnnGetRNNLinLayerBiasParams(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t, layer::Cint, xDesc::cudnnTensorDescriptor_t, wDesc::cudnnFilterDescriptor_t, w, linLayerID::Cint, linLayerBiasDesc::cudnnFilterDescriptor_t, linLayerBias)
    ccall((:cudnnGetRNNLinLayerBiasParams, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, Ptr{Void}, Cint, cudnnFilterDescriptor_t, Ptr{Ptr{Void}}), handle, rnnDesc, layer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias)
end

function cudnnRNNForwardInference(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint, xDesc, x, hxDesc::cudnnTensorDescriptor_t, hx, cxDesc::cudnnTensorDescriptor_t, cx, wDesc::cudnnFilterDescriptor_t, w, yDesc, y, hyDesc::cudnnTensorDescriptor_t, hy, cyDesc::cudnnTensorDescriptor_t, cy, workspace, workSpaceSizeInBytes::Cint)
    ccall((:cudnnRNNForwardInference, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Cint), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes)
end

function cudnnRNNForwardTraining(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint, xDesc, x, hxDesc::cudnnTensorDescriptor_t, hx, cxDesc::cudnnTensorDescriptor_t, cx, wDesc::cudnnFilterDescriptor_t, w, yDesc, y, hyDesc::cudnnTensorDescriptor_t, hy, cyDesc::cudnnTensorDescriptor_t, cy, workspace, workSpaceSizeInBytes::Cint, reserveSpace, reserveSpaceSizeInBytes::Cint)
    ccall((:cudnnRNNForwardTraining, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Cint, Ptr{Void}, Cint), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnRNNBackwardData(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint, yDesc, y, dyDesc, dy, dhyDesc::cudnnTensorDescriptor_t, dhy, dcyDesc::cudnnTensorDescriptor_t, dcy, wDesc::cudnnFilterDescriptor_t, w, hxDesc::cudnnTensorDescriptor_t, hx, cxDesc::cudnnTensorDescriptor_t, cx, dxDesc, dx, dhxDesc::cudnnTensorDescriptor_t, dhx, dcxDesc::cudnnTensorDescriptor_t, dcx, workspace, workSpaceSizeInBytes::Cint, reserveSpace, reserveSpaceSizeInBytes::Cint)
    ccall((:cudnnRNNBackwardData, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnFilterDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, Cint, Ptr{Void}, Cint), handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnRNNBackwardWeights(handle::cudnnHandle_t, rnnDesc::cudnnRNNDescriptor_t, seqLength::Cint, xDesc, x, hxDesc::cudnnTensorDescriptor_t, hx, yDesc, y, workspace, workSpaceSizeInBytes::Cint, dwDesc::cudnnFilterDescriptor_t, dw, reserveSpace, reserveSpaceSizeInBytes::Cint)
    ccall((:cudnnRNNBackwardWeights, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnRNNDescriptor_t, Cint, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{cudnnTensorDescriptor_t}, Ptr{Void}, Ptr{Void}, Cint, cudnnFilterDescriptor_t, Ptr{Void}, Ptr{Void}, Cint), handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes)
end

function cudnnSetFilter4dDescriptor_v3(filterDesc::cudnnFilterDescriptor_t, dataType::cudnnDataType_t, k::Cint, c::Cint, h::Cint, w::Cint)
    ccall((:cudnnSetFilter4dDescriptor_v3, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, Cint, Cint, Cint, Cint), filterDesc, dataType, k, c, h, w)
end

function cudnnSetFilter4dDescriptor_v4(filterDesc::cudnnFilterDescriptor_t, dataType::cudnnDataType_t, format::cudnnTensorFormat_t, k::Cint, c::Cint, h::Cint, w::Cint)
    ccall((:cudnnSetFilter4dDescriptor_v4, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Cint, Cint, Cint), filterDesc, dataType, format, k, c, h, w)
end

function cudnnGetFilter4dDescriptor_v3(filterDesc::cudnnFilterDescriptor_t, dataType, k, c, h, w)
    ccall((:cudnnGetFilter4dDescriptor_v3, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), filterDesc, dataType, k, c, h, w)
end

function cudnnGetFilter4dDescriptor_v4(filterDesc::cudnnFilterDescriptor_t, dataType, format, k, c, h, w)
    ccall((:cudnnGetFilter4dDescriptor_v4, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), filterDesc, dataType, format, k, c, h, w)
end

function cudnnSetFilterNdDescriptor_v3(filterDesc::cudnnFilterDescriptor_t, dataType::cudnnDataType_t, nbDims::Cint, filterDimA)
    ccall((:cudnnSetFilterNdDescriptor_v3, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, Cint, Ptr{Cint}), filterDesc, dataType, nbDims, filterDimA)
end

function cudnnSetFilterNdDescriptor_v4(filterDesc::cudnnFilterDescriptor_t, dataType::cudnnDataType_t, format::cudnnTensorFormat_t, nbDims::Cint, filterDimA)
    ccall((:cudnnSetFilterNdDescriptor_v4, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, Cint, Ptr{Cint}), filterDesc, dataType, format, nbDims, filterDimA)
end

function cudnnGetFilterNdDescriptor_v3(filterDesc::cudnnFilterDescriptor_t, nbDimsRequested::Cint, dataType, nbDims, filterDimA)
    ccall((:cudnnGetFilterNdDescriptor_v3, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{Cint}, Ptr{Cint}), filterDesc, nbDimsRequested, dataType, nbDims, filterDimA)
end

function cudnnGetFilterNdDescriptor_v4(filterDesc::cudnnFilterDescriptor_t, nbDimsRequested::Cint, dataType, format, nbDims, filterDimA)
    ccall((:cudnnGetFilterNdDescriptor_v4, cudnn), cudnnStatus_t, (cudnnFilterDescriptor_t, Cint, Ptr{cudnnDataType_t}, Ptr{cudnnTensorFormat_t}, Ptr{Cint}, Ptr{Cint}), filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)
end

function cudnnSetPooling2dDescriptor_v3(poolingDesc::cudnnPoolingDescriptor_t, mode::cudnnPoolingMode_t, windowHeight::Cint, windowWidth::Cint, verticalPadding::Cint, horizontalPadding::Cint, verticalStride::Cint, horizontalStride::Cint)
    ccall((:cudnnSetPooling2dDescriptor_v3, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, Cint, Cint, Cint, Cint, Cint, Cint), poolingDesc, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

function cudnnSetPooling2dDescriptor_v4(poolingDesc::cudnnPoolingDescriptor_t, mode::cudnnPoolingMode_t, maxpoolingNanOpt::cudnnNanPropagation_t, windowHeight::Cint, windowWidth::Cint, verticalPadding::Cint, horizontalPadding::Cint, verticalStride::Cint, horizontalStride::Cint)
    ccall((:cudnnSetPooling2dDescriptor_v4, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint, Cint, Cint, Cint, Cint, Cint), poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

function cudnnGetPooling2dDescriptor_v3(poolingDesc::cudnnPoolingDescriptor_t, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    ccall((:cudnnGetPooling2dDescriptor_v3, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Ptr{cudnnPoolingMode_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

function cudnnGetPooling2dDescriptor_v4(poolingDesc::cudnnPoolingDescriptor_t, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
    ccall((:cudnnGetPooling2dDescriptor_v4, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Ptr{cudnnPoolingMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)
end

function cudnnSetPoolingNdDescriptor_v3(poolingDesc::cudnnPoolingDescriptor_t, mode::cudnnPoolingMode_t, nbDims::Cint, windowDimA, paddingA, strideA)
    ccall((:cudnnSetPoolingNdDescriptor_v3, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, nbDims, windowDimA, paddingA, strideA)
end

function cudnnSetPoolingNdDescriptor_v4(poolingDesc::cudnnPoolingDescriptor_t, mode::cudnnPoolingMode_t, maxpoolingNanOpt::cudnnNanPropagation_t, nbDims::Cint, windowDimA, paddingA, strideA)
    ccall((:cudnnSetPoolingNdDescriptor_v4, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
end

function cudnnGetPoolingNdDescriptor_v3(poolingDesc::cudnnPoolingDescriptor_t, nbDimsRequested::Cint, mode, nbDims, windowDimA, paddingA, strideA)
    ccall((:cudnnGetPoolingNdDescriptor_v3, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Cint, Ptr{cudnnPoolingMode_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, nbDimsRequested, mode, nbDims, windowDimA, paddingA, strideA)
end

function cudnnGetPoolingNdDescriptor_v4(poolingDesc::cudnnPoolingDescriptor_t, nbDimsRequested::Cint, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
    ccall((:cudnnGetPoolingNdDescriptor_v4, cudnn), cudnnStatus_t, (cudnnPoolingDescriptor_t, Cint, Ptr{cudnnPoolingMode_t}, Ptr{cudnnNanPropagation_t}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA)
end

function cudnnActivationForward_v3(handle::cudnnHandle_t, mode::cudnnActivationMode_t, alpha, xDesc::cudnnTensorDescriptor_t, x, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnActivationForward_v3, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, mode, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnActivationForward_v4(handle::cudnnHandle_t, activationDesc::cudnnActivationDescriptor_t, alpha, xDesc::cudnnTensorDescriptor_t, x, beta, yDesc::cudnnTensorDescriptor_t, y)
    ccall((:cudnnActivationForward_v4, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
end

function cudnnActivationBackward_v3(handle::cudnnHandle_t, mode::cudnnActivationMode_t, alpha, yDesc::cudnnTensorDescriptor_t, y, dyDesc::cudnnTensorDescriptor_t, dy, xDesc::cudnnTensorDescriptor_t, x, beta, dxDesc::cudnnTensorDescriptor_t, dx)
    ccall((:cudnnActivationBackward_v3, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationMode_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, mode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end

function cudnnActivationBackward_v4(handle::cudnnHandle_t, activationDesc::cudnnActivationDescriptor_t, alpha, yDesc::cudnnTensorDescriptor_t, y, dyDesc::cudnnTensorDescriptor_t, dy, xDesc::cudnnTensorDescriptor_t, x, beta, dxDesc::cudnnTensorDescriptor_t, dx)
    ccall((:cudnnActivationBackward_v4, cudnn), cudnnStatus_t, (cudnnHandle_t, cudnnActivationDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}, Ptr{Void}, cudnnTensorDescriptor_t, Ptr{Void}), handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
end
