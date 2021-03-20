module RBMS

export
    AbstractRBM,
    RBM,
    Gaussian,
    Bernoulli,
    Unitary,
    BernoulliRBM,
    fit!,
    pseudolikelihood,
    console_logger,
    CDk,
    cpu,
    gpu

include("rbm.jl")

# Conditional GPU support
# https://juliagpu.gitlab.io/CUDA.jl/installation/conditional/#Scenario-2:-GPU-is-optional
using CUDA
const use_cuda = Ref(false)

include("cuda.jl")

function __init__()
  use_cuda[] = CUDA.functional() # Can be overridden after load
  if CUDA.functional()
      @info "GPU available."
  end
end

end # module
