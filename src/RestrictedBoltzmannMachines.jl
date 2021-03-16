module RestrictedBoltmannMachines

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
    CDk

include("rbm.jl")

end # module
