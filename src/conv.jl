
## convolution

function conv2d{T}(x::CuArray{T}, w::CuArray{T};
                  handle=cudnnhandle(), algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, alpha=1,
                  o...) # padding=0, stride=1, upscale=1, mode=0
    y = similar(x, cdims(w,x;o...))
    beta=0 # nonzero beta does not make sense when we create y
    @cuda(cudnn, cudnnConvolutionForward,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,UInt32,Cptr,Csize_t,Ptr{T},Cptr,Ptr{T}),
          handle,Ref(T(alpha)),TD(x),x.ptr,FD(w),w.ptr,CD(w,x;o...),algo,workSpace,
          workSpaceSizeInBytes,Ref(T(beta)),TD(y),y.ptr)
    return y
end

function conv2d_back_x{T}(x::CuArray{T}, w::CuArray{T}, dy::CuArray{T};
                          handle=cudnnhandle(), alpha=1, algo=0, workSpace=C_NULL,
                          workSpaceSizeInBytes=0, o...) # padding=0, stride=1, upscale=1, mode=0
    beta = 0
    dx = similar(x)
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

function conv2d_back_w{T}(x::CuArray{T},w::CuArray{T},dy::CuArray{T};
                          handle=cudnnhandle(), alpha=1, algo=0, workSpace=C_NULL,
                          workSpaceSizeInBytes=0,
                   o...) # padding=0, stride=1, upscale=1, mode=0
    beta = 0
    dw = similar(w)
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




## deconvolution

function deconv2d(w, x; o...)
    y = similar(x, dcdims(w, x; o...))
    return conv4x(w, y, x; o...)
end

function deconv2d_back_w(w,x,dy; o...)
    return conv4w(w,dy,x;o...)
end

function deconv2d_back_x(w, x, dy; o...)
    return conv4(w, dy; o...)
end


