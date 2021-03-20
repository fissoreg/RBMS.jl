using Logging
using CUDA: cu

cpu(x) = to_cpu(x)

function gpu(x)
    if use_cuda[]
        return to_gpu(x)
    else
        @warn "GPU not available, falling back to CPU."
    end

    x
end

to_gpu(x::AbstractArray) = cu(x)
to_gpu(rbm::AbstractRBM{T,V,H}) where {T,V,H} =
    RBM{T,V,H}(cu(rbm.W), cu(rbm.vbias), cu(rbm.hbias))

to_cpu(x::AbstractArray) = Array(x)
to_cpu(rbm::AbstractRBM{T,V,H}) where {T,V,H} =
    RBM{T,V,H}(Array(rbm.W), Array(rbm.vbias), Array(rbm.hbias))
