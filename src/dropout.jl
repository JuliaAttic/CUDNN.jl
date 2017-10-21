
## WARNING: this is an attempt to wrap cuDNNs dropout implementation
##          it doesn't work and should be considered work in progress

## Dropout descriptor

mutable struct DOD
    ptr::Ptr{Void}
    states::CuArray{UInt8,1}
end

# TODO: what should be passed for states? what size should it have?
function DOD(dropout::Float32; handle=cudnnhandle(), states=C_NULL,
             seed=UInt64(42))
    d = Ref{Cptr}(0)
    @cuda(cudnn, cudnnCreateDropoutDescriptor, (Ref{Cptr},), d)

    statesSizeInBytes = Ref{Csize_t}(0)
    @cuda(cudnn, cudnnDropoutGetStatesSize, (Cptr, Ref{Csize_t}), handle, statesSizeInBytes)

    states = CuArray{UInt8,1}(statesSizeInBytes[])
    # TODO: states will be garbage collected and thus its memory will be reclained
    # should we return it in DOD?
    @cuda(cudnn, cudnnSetDropoutDescriptor,
          (Ref{Cptr}, Cptr, Cfloat, Cptr, Cuint, Culong),
          d, handle, dropout, states.ptr, statesSizeInBytes[], seed)
    dod = DOD(d[], states)
    finalizer(dod, x->@cuda(cudnn, cudnnDestroyDropoutDescriptor, (Cptr,), x.ptr))
    # finalizer(dod, x -> cudnnDestroyDropoutDescriptor(x.ptr))
    return dod
end




# doesn't work on my machine
function get_dropout_desc(dod::DOD)
    dropout = Ref{Cfloat}(0)
    states = Ref{Cptr}(0)
    seed = Ref{Culonglong}(0)
    # TODO: cudnnGetDropoutDescriptor cannot be found. Wrong library version?
    @cuda(cudnn, cudnnGetDropoutDescriptor,
          (Cptr, Cptr, Ptr{Cfloat}, Ptr{Cptr}, Ptr{Culong}),
          d, handle, dropout, states, seed)
end


function  get_reserve_space_size(x::CuArray{T}) where T
    td = TD(x)
    sz = Ref{Csize_t}(0)
    @cuda(cudnn, cudnnDropoutGetReserveSpaceSize, (Cptr, Ref{Csize_t}), td.ptr, sz)
    return sz[]
end


function  dropout_get_states_size(handle)
    sz = Ref{Csize_t}(0)
    @cuda(cudnn, cudnnDropoutGetStatesSize, (Cptr, Ref{Csize_t}), handle, sz)
    return sz[]
end


function dropout_forward!(y::CuArray{T}, x::CuArray{T}, dropout::Float64;
                          handle=cudnnhandle(), rs=nothing) where T
    dod = DOD(Float32(dropout))
    xtd = TD(x)
    ytd = TD(y)
    if rs == nothing
        rs_sz = get_reserve_space_size(x)
        rs = CuArray(zeros(UInt8, rs_sz))
        # rs = Ptr{Void}(0)
    else
        rs_sz = length(rs)
    end
    @cuda(cudnn, cudnnDropoutForward,
          #  (Cptr, Cptr, Cptr, Cptr, Cptr, Cptr, Cptr, Csize_t),
          (Cptr, Cptr, Cptr, Cptr, Cptr, Cptr, Cptr, Csize_t),
          handle, dod.ptr, xtd.ptr, x.ptr, ytd.ptr, y.ptr, rs.ptr, rs_sz)
    
end



function main()
    x = CuArray(randn(Float32, 5, 4, 3, 2))
    y = similar(x)
    dropout = 0.5
    handle = cudnnhandle()
    rs = nothing


    dropoutDesc = DOD(Float32(dropout))
    xdesc = TD(x)
    ydesc = TD(y)
    x = x
    y = y

    reserveSpace = C_NULL
    reserveSizeInBytes = Cint(get_reserve_space_size(x))
    
    
    cudnnDropoutForward(handle, dropoutDesc.ptr, xdesc, x.ptr, ydesc, y.ptr, reserveSpace,
                        reserveSizeInBytes)
end
