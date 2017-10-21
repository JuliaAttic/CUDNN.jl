
# WARNING: I still have doubts this works correctly, use with caution

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


function batchnorm_param_size(x::CuArray, mode::UInt32)
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


mutable struct BatchNormState{T}
    bnScale::CuArray{T,4}
    bnBias::CuArray{T,4}
    resultRunningMean::CuArray{T,4}
    resultRunningVariance::CuArray{T,4}
    resultSaveMean::CuArray{T,4}
    resultSaveInvVariance::CuArray{T,4}
end

function BatchNormState(x::CuArray{T,4}, mode=CUDNN_BATCHNORM_SPATIAL) where T
    psz = batchnorm_param_size(x, mode)
    bnScale = CuArray(randn(T, psz))
    bnBias = CuArray(randn(T, psz))
    resultRunningMean = CuArray(zeros(T, psz))
    resultRunningVariance = CuArray(zeros(T, psz))
    resultSaveMean = CuArray(zeros(T, psz))
    resultSaveInvVariance = CuArray(zeros(T, psz))
    return BatchNormState(bnScale, bnBias, resultRunningMean, resultRunningVariance,
                          resultSaveMean, resultSaveInvVariance)
end


function batchnorm_train!(y::CuArray{T,4}, x::CuArray{T,4}, s::BatchNormState;
                          handle=cudnnhandle(), alpha=1, beta=0,
                          exponentialAverageFactor=T(1), mode=CUDNN_BATCHNORM_SPATIAL,
                          epsilon=CUDNN_BN_MIN_EPSILON) where T
    xtd = TD(x)
    ytd = TD(y)
    bnScaleBiasMeanVarDesc = BND(xtd, mode)
    @cuda(cudnn, cudnnBatchNormalizationForwardTraining,
          (Cptr, UInt32, Cptr, Cptr, Cptr, Cptr, Cptr, Cptr,
           Cptr, Cptr, Cptr, Cdouble,
           Cptr, Cptr, Cdouble,
           Cptr, Cptr),
          handle, mode, Ref(T(alpha)), Ref(T(beta)), xtd.ptr, x.ptr, ytd.ptr, y.ptr,
          bnScaleBiasMeanVarDesc.ptr, s.bnScale.ptr, s.bnBias.ptr, exponentialAverageFactor,
          s.resultRunningMean.ptr, s.resultRunningVariance.ptr, epsilon,
          s.resultSaveMean.ptr, s.resultSaveInvVariance.ptr)

end

function batchnorm_train(x::CuArray{T,4}, s::BatchNormState; opts...) where T
    y = similar(x)
    batchnorm_train!(y, x, s;  opts...)
    return y
end



function batchnorm_infer!(y::CuArray{T,4}, x::CuArray{T,4}, s::BatchNormState;
                          handle=cudnnhandle(), alpha=1, beta=0,
                          exponentialAverageFactor=T(1), mode=CUDNN_BATCHNORM_SPATIAL,
                          epsilon=CUDNN_BN_MIN_EPSILON) where T
    xtd = TD(x)
    ytd = TD(y)
    xsz = size(x)
    bnScaleBiasMeanVarDesc = BND(xtd, mode)
    estimatedMean = s.resultRunningMean
    estimatedVariance = s.resultRunningVariance
    @cuda(cudnn, cudnnBatchNormalizationForwardInference,
          (Cptr, UInt32, Cptr, Cptr, Cptr, Cptr, Cptr, Cptr,
           Cptr, Cptr, Cptr,
           Cptr, Cptr, Cdouble),
          handle, mode, Ref(T(alpha)), Ref(T(beta)), xtd.ptr, x.ptr, ytd.ptr, y.ptr,
          bnScaleBiasMeanVarDesc.ptr, s.bnScale.ptr, s.bnBias.ptr,
          estimatedMean.ptr, estimatedVariance.ptr, epsilon)
end

function batchnorm_infer(x::CuArray{T,4}, s::BatchNormState; opts...) where T
    y = similar(x)
    batchnorm_infer!(y, x, s; opts...)
    return y
end



function batchnorm_grad!(dx::CuArray{T,4}, x::CuArray{T,4}, dy::CuArray{T,4}, s::BatchNormState;
                         handle=cudnnhandle(), alpha_data=1, beta_data=0,
                         alpha_param=1, beta_param=0,
                         exponentialAverageFactor=T(1), mode=CUDNN_BATCHNORM_SPATIAL,
                         epsilon=CUDNN_BN_MIN_EPSILON) where T    
    xtd = TD(x)
    dytd = TD(dy)
    dxtd = TD(dx)
    xsz = size(x)
    bnScaleBiasMeanVarDesc = BND(xtd, mode)
    # should we update bnScale & bnBias manually or cuDNN does it automatically?
    resultBnScaleDiff = similar(s.bnScale)
    resultBnBiasDiff = similar(s.bnScale)    
    savedMean = s.resultSaveMean
    savedInvVariance = s.resultSaveInvVariance
    @cuda(cudnn, cudnnBatchNormalizationBackward,
          (Cptr, UInt32, Cptr, Cptr, Cptr, Cptr,
           Cptr, Cptr, Cptr, Cptr, Cptr, Cptr,
           Cptr, Cptr, Cptr, Cptr,
           Cdouble, Cptr, Cptr),
          handle, mode, Ref(T(alpha_data)), Ref(T(beta_data)), Ref(T(alpha_param)), Ref(T(beta_param)),
          xtd.ptr, x.ptr, dytd.ptr, dy.ptr, dxtd.ptr, dx.ptr,
          bnScaleBiasMeanVarDesc.ptr, s.bnScale.ptr, resultBnScaleDiff.ptr, resultBnBiasDiff.ptr,
          epsilon, savedMean.ptr, savedInvVariance.ptr)
end


function batchnorm_grad(x::CuArray{T,4}, dy::CuArray{T,4}, s::BatchNormState; opts...) where T
    dx = similar(x)
    batchnorm_grad!(dx, x, dy, s; opts...)
    return dx
end
