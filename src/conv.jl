
## convolution

function conv2d!{T}(y::CuArray{T}, x::CuArray{T}, w::CuArray{T};
                    handle=cudnnhandle(), algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0,
                    alpha=1, beta=0, o...) # padding=0, stride=1, upscale=1, mode=0
    @cuda(cudnn, cudnnConvolutionForward,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,UInt32,Cptr,Csize_t,Ptr{T},Cptr,Ptr{T}),
          handle,Ref(T(alpha)),TD(x),x.ptr,FD(w),w.ptr,CD(w,x;o...),algo,workSpace,
          workSpaceSizeInBytes,Ref(T(beta)),TD(y),y.ptr)
    return y
end


function conv2d{T}(x::CuArray{T}, w::CuArray{T};
                  handle=cudnnhandle(), algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, alpha=1,
                  o...) # padding=0, stride=1, upscale=1, mode=0
    y = similar(x, cdims(w,x;o...))
    conv2d!(y, x, w; handle=handle, algo=algo, workSpace=workSpace,
            workSpaceSizeInBytes=workSpaceSizeInBytes, alpha=1, o...)
    return y
end


function conv2_grad_x!{T}(dx::CuArray{T}, x::CuArray{T}, w::CuArray{T}, dy::CuArray{T};
                          handle=cudnnhandle(), alpha=1, algo=0, beta=0, workSpace=C_NULL,
                          workSpaceSizeInBytes=0, o...) # padding=0, stride=1, upscale=1, mode=0
    cudnnVersion = cudnn_version()
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,UInt32,Cptr,Csize_t,Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w.ptr,TD(dy),dy.ptr,CD(w,x;o...),algo,workSpace,
              workSpaceSizeInBytes,Ref(T(beta)),TD(dx),dx.ptr)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardData_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,UInt32,Cptr,Csize_t,Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w.ptr,TD(dy),dy.ptr,CD(w,x;o...),algo,workSpace,
              workSpaceSizeInBytes,Ref(T(beta)),TD(dx),dx.ptr)
    else
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w.ptr,TD(dy),dy.ptr,CD(w,x;o...),
              Ref(T(beta)),TD(dx),dx.ptr)
    end
    return dx
end


function conv2d_grad_x{T}(x::CuArray{T}, w::CuArray{T}, dy::CuArray{T};
                          handle=cudnnhandle(), alpha=1, algo=0, workSpace=C_NULL,
                          workSpaceSizeInBytes=0, o...) # padding=0, stride=1, upscale=1, mode=0    
    dx = similar(x)
    conv2_grad_x!(dx, x, w, dy; handle=handle, alpha=alpha, algo=algo, workSpace=workSpace,
                   workSpaceSizeInBytes=workSpaceSizeInBytes, o...)
    return dx
end


function conv2d_grad_w!{T}(dw::CuArray{T}, x::CuArray{T}, w::CuArray{T}, dy::CuArray{T};
                           handle=cudnnhandle(), alpha=1, beta=0, algo=0, workSpace=C_NULL,
                           workSpaceSizeInBytes=0,
                           o...) # padding=0, stride=1, upscale=1, mode=0
    cudnnVersion = cudnn_version()
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,UInt32,Cptr,Csize_t,Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x.ptr,TD(dy),dy.ptr,CD(w,x;o...),algo,workSpace,
              workSpaceSizeInBytes,Ref(T(beta)),FD(dw),dw.ptr)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardFilter_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,UInt32,Cptr,Csize_t,Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x.ptr,TD(dy),dy.ptr,CD(w,x;o...),algo,workSpace,
              workSpaceSizeInBytes,Ref(T(beta)),FD(dw),dw.ptr)
    else
        @cuda(cudnn,cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,       Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x.ptr,TD(dy),dy.ptr,CD(w,x;o...),Ref(T(beta)),FD(dw),dw.ptr)
    end
    return dw
end



function conv2d_grad_w{T}(x::CuArray{T}, w::CuArray{T}, dy::CuArray{T};
                          handle=cudnnhandle(), alpha=1, algo=0, workSpace=C_NULL,
                          workSpaceSizeInBytes=0,
                   o...) # padding=0, stride=1, upscale=1, mode=0    
    dw = similar(w)
    conv2d_grad_w!(dw, x, w, dy; handle=handle, alpha=alpha, algo=algo,  workSpace=workSpace,
                   workSpaceSizeInBytes=workSpaceSizeInBytes, o...)
    return dw
end




## deconvolution

function deconv2d(x, w; o...)
    y = similar(x, dcdims(w, x; o...))
    return conv2_grad_x(y, w, x; o...)
end

function deconv2_grad_w(x, w, dy; o...)
    return conv2_grad_w(dy, w, x; o...)
end

function deconv2_grad_x(x, w, dy; o...)
    return conv2d(dy, w; o...)
end
