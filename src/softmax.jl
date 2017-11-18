
function softmax4d!(y::CuArray{T}, x::CuArray{T};
                 handle=cudnnhandle(),
                 algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                 mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                 alpha=1.0, beta=0.0) where T
    @cuda(cudnn, cudnnSoftmaxForward,
          (Cptr, Cuint, Cuint, Ptr{T}, Cptr, Ptr{T}, Ptr{T}, Cptr, Ptr{T}),
          handle, algorithm, mode, Ref(T(alpha)), TD(x), x.ptr, Ref(T(beta)), TD(y), y.ptr)
    return y
end

softmax4d(x::CuArray{T}) where T = softmax4d!(similar(x), x)


function softmax4d_grad!(dx::CuArray{T}, y::CuArray{T}, dy::CuArray{T};
                         handle=cudnnhandle(),
                         algorithm=CUDNN_SOFTMAX_ACCURATE, # or CUDNN_SOFTMAX_FAST
                         mode=CUDNN_SOFTMAX_MODE_INSTANCE, # or CUDNN_SOFTMAX_MODE_CHANNEL
                         alpha=1.0, beta=0.0) where T
    @cuda(cudnn, cudnnSoftmaxBackward,
          (Cptr, Cuint, Cuint,
           Ptr{T}, Cptr, Ptr{T},
           Cptr, Ptr{T},
           Ptr{T}, Cptr, Ptr{T}),
          handle, algorithm, mode,
          Ref(T(alpha)), TD(y), y.ptr,
          TD(dy), dy.ptr,
          Ref(T(beta)), TD(dx), dx.ptr)
    return dx
end

softmax4d_grad(y::CuArray{T}, dy::CuArray{T}) where T = softmax4d_grad!(similar(dy), y, dy)
