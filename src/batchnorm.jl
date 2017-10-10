
mutable struct BND
    ptr::Ptr{Void}
end

function BND(xtd::TD, mode::Cuint)
    d = Ref{Cptr}(0)
    @cuda(cudnn, cudnnCreateTensorDescriptor, (Ptr{Cptr},), d)
    @cuda(cudnn, cudnnDeriveBNTensorDescriptor,
          (Ref{Void}, Cptr, Cuint),
          d[], xtd.ptr, mode)
    return BND(d[])
end


function batchnorm_param_size(x::CuArray)
    xsz = size(x)
    if mode == CUDNN_BATCHNORM_PER_ACTIVATION
        psz = (1, xsz[3], xsz[2], xsz[1])  # 1xCxHxW
    elseif mode == CUDNN_BATCHNORM_SPATIAL
        psz = (1, xsz[3], 1, 1)            # 1xCx1x1
    else
        error("Mode $mode is not supported")
    end
    return psz
end


function batnchnorm_train!(y::CuArray{T,4}, x::CuArray{T,4}; handle=cudnnhandle(),
                           alpha=1, beta=0, mode=CUDNN_BATCHNORM_SPATIAL,
                           exponentialAverageFactor=T(1), epsilon=CUDNN_BN_MIN_EPSILON) where T
    xtd = TD(x)
    ytd = TD(y)
    # this descriptor will be used in CUDA call
    bnScaleBiasMeanVarDesc = BND(xtd, mode)
    # but we also need to infer the same description for use in user-friendly CuArray
    psz = batchnorm_param_size(x)
    
    bnScale = CuArray(randn(T, psz))
    bnBias = CuArray(randn(T, psz))

    resultRunningMean = CuArray(zeros(T, psz))
    resultRunningVariance = CuArray(zeros(T, psz))

    resultSaveMean = CuArray(zeros(T, psz))
    resultSaveInvVariance = CuArray(zeros(T, psz))

    @cuda(cudnn, cudnnBatchNormalizationForwardTraining,
          (Cptr, UInt32, Cptr, Cptr, Cptr, Cptr, Cptr, Cptr,
           Cptr, Cptr, Cptr, Cdouble,
           Cptr, Cptr, Cdouble,
           Cptr, Cptr),
          handle, mode, Ref(T(alpha)), Ref(T(beta)), xtd.ptr, x.ptr, ytd.ptr, y.ptr,
          bnScaleBiasMeanVarDesc.ptr, bnScale.ptr, bnBias.ptr, exponentialAverageFactor,
          resultRunningMean.ptr, resultRunningVariance.ptr, epsilon,
          resultSaveMean.ptr, resultSaveInvVariance.ptr)

end



function batnchnorm_infer!(y::CuArray{T,4}, x::CuArray{T,4}) where T
    # TODO: move to parameters
    handle = cudnnhandle()
    mode = CUDNN_BATCHNORM_PER_ACTIVATION
    alpha = 1
    beta = 0
    xdesc = TD(x)
    ydesc = TD(y)
    xsz = size(x)
    paramSz = (1, xsz[3], xsz[2], xsz[1])  # 1xCxHxW
    bnScale = CuArray(randn(T, paramSz))
    bnBias = CuArray(randn(T, paramSz))
    exponentialAverageFactor = T(0.5)
    bnScaleBiasMeanVarDesc = TD(bnScale)
    epsilon = CUDNN_BN_MIN_EPSILON
    estimatedMean = CuArray(zeros(T, paramSz))
    estimatedVariance = CuArray(zeros(T, paramSz))

    @cuda(cudnn, cudnnBatchNormalizationForwardInference,
          (Cptr, UInt32, Cptr, Cptr, Cptr, Cptr, Cptr, Cptr,
           Cptr, Cptr, Cptr,
           Cptr, Cptr, Cdouble),
          handle, mode, Ref(T(alpha)), Ref(T(beta)), xdesc.ptr, x.ptr, ydesc.ptr, y.ptr,
          bnScaleBiasMeanVarDesc.ptr, bnScale.ptr, bnBias.ptr,
          estimatedMean.ptr, estimatedVariance.ptr, epsilon)

end





function batchnorm_grad!(dx::CuArray{T,4}, x::CuArray{T,4}, dy::CuArray{T,4}) where T
    handle = cudnnhandle()
    # mode = CUDNN_BATCHNORM_PER_ACTIVATION
    mode = CUDNN_BATCHNORM_SPATIAL
    alpha_data = 1
    beta_data = 0
    alpha_param = 1
    beta_param = 0
    xdesc = TD(x)
    dydesc = TD(dy)
    dxdesc = TD(dx)
    xsz = size(x)

    bnScaleBiasMeanVarDesc = BND(xdesc, mode)
    # infer the same description for CuArray
    if mode == CUDNN_BATCHNORM_PER_ACTIVATION
        paramSz = (1, xsz[3], xsz[2], xsz[1])  # 1xCxHxW
    elseif mode == CUDNN_BATCHNORM_SPATIAL
        paramSz = (1, xsz[3], 1, 1)            # 1xCx1x1
    else
        error("Mode $mode is not supported") #
    end

    # TODO: use cudnnDeriveBNTensorDescriptor to create descriptor

    bnScale = CuArray(randn(T, paramSz))
    resultBnScaleDiff = CuArray(zeros(T, paramSz))
    resultBnBiasDiff = CuArray(zeros(T, paramSz))
    epsilon = CUDNN_BN_MIN_EPSILON
    savedMean = CuArray(randn(T, paramSz))
    savedInvVariance = CuArray(randn(T, paramSz))


    @cuda(cudnn, cudnnBatchNormalizationBackward,
          (Cptr, UInt32, Cptr, Cptr, Cptr, Cptr,
           Cptr, Cptr, Cptr, Cptr, Cptr, Cptr,
           Cptr, Cptr, Cptr, Cptr,
           Cdouble, Cptr, Cptr),
          handle, mode, Ref(T(alpha_data)), Ref(T(beta_data)), Ref(T(alpha_param)), Ref(T(beta_param)),
          xdesc.ptr, x.ptr, dydesc.ptr, dy.ptr, dxdesc.ptr, dx.ptr,
          bnScaleBiasMeanVarDesc.ptr, bnScale.ptr, resultBnScaleDiff.ptr, resultBnBiasDiff.ptr,
          epsilon, savedMean.ptr, savedInvVariance.ptr)
end





function main()
    T = Float32
    x = CuArray(randn(T, 5, 4, 3, 2))
    y = similar(x)

    dy = CuArray(randn(T, 5, 4, 3, 2))
    dx = similar(dy)



    handle = cudnnhandle()
end
