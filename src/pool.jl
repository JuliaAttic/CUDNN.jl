
function pool{T}(x::CuArray{T}; handle=cudnnhandle(), alpha=1, 
                 o...) # window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0
    y = similar(x, pdims(x; o...))
    beta = 0
    @cuda(cudnn, cudnnPoolingForward,
          (Cptr, Cptr, Ptr{T}, Cptr, Ptr{T}, Ptr{T}, Cptr,Ptr{T}),
          handle, PD(x;o...), Ref(T(alpha)), TD(x), x.ptr, Ref(T(beta)), TD(y), y.ptr)
    return y
end

function pool_grad{T}(x::CuArray{T}, y::CuArray{T}, dy::CuArray{T};
                      handle=cudnnhandle(), alpha=1, mode=0,
                      o...) # window=2, padding=0, stride=window, maxpoolingNanOpt=0
    if alpha!=1 && mode==0; error("Gradient of pool(alpha!=1,mode=0) broken in CUDNN"); end
    dx = similar(x)
    beta = 0
    @cuda(cudnn,cudnnPoolingBackward,
          (Cptr, Cptr, Ptr{T}, Cptr, Ptr{T}, Cptr, Ptr{T}, Cptr, Ptr{T}, Ptr{T}, Cptr, Ptr{T}),
          handle, PD(x; mode=mode, o...), Ref(T(alpha)), TD(y), y.ptr,
          TD(dy), dy.ptr, TD(x), x.ptr, Ref(T(beta)), TD(dx), dx.ptr)
    return dx
end


function unpool(x; window=2, alpha=1, o...) # padding=0, stride=window, mode=0, maxpoolingNanOpt=0
    w = prod(psize(window,x))
    y = similar(x,updims(x; window=window, o...))
    poolx(y,x,x.*w; o..., window=window, mode=1, alpha=1/alpha)
end

function unpool_grad(dy; window=2, alpha=1, o...) # padding=0, stride=window, mode=0,
                                                    # maxpoolingNanOpt=0
    w = prod(psize(window,dy))
    pool(dy.ptr; o..., window=window, mode=1, alpha=1/alpha) * w
end
