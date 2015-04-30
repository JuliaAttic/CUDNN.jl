type Tensor
    data::CudaArray
    desc::cudnnTensorDescriptor_t
    function Tensor(a::Array)
        dt = (eltype(a) == Float64 ? CUDNN_DATA_DOUBLE :
              eltype(a) == Float32 ? CUDNN_DATA_FLOAT :
              error("Supported data types are Float32 and Float64"))
        c = CudaArray(a)
        d = cudnnTensorDescriptor_t[0]
        # TODO: We should make sure this gets deallocated properly:
        cudnnCreateTensorDescriptor(d)
        # Default order in CUDNN is n,c,h,w with w fastest changing
        # Default order in Julia is a[w,h,c,n] with w fastest changing
        # I think the reverse below is what CUDNN expects
        cudnnSetTensorNdDescriptor(d[1], dt, ndims(a), Cint[reverse(size(a))...], Cint[reverse(strides(a))...])
        new(c, d[1])
    end
end

CUDArt.to_host(t::Tensor)=to_host(t.data)
CUDArt.free(t::Tensor)=(free(t.data); cudnnDestroyTensorDescriptor(t.desc))
cudnnGetTensorNdDescriptor(t::Tensor)=cudnnGetTensorNdDescriptor(t.desc)

function cudnnGetTensorNdDescriptor(td::cudnnTensorDescriptor_t, nbDimsRequested=8)
    dataType = cudnnDataType_t[0]
    nbDims = Array(Cint, 1)
    dimA = Array(Cint, nbDimsRequested)
    strideA = Array(Cint, nbDimsRequested)
    cudnnGetTensorNdDescriptor(td,nbDimsRequested,dataType,nbDims,dimA,strideA)
    return (dataType[1], nbDims[1], dimA[1:nbDims[1]], strideA[1:nbDims[1]])
    # nbDimsRequested > 8 gives error
end

