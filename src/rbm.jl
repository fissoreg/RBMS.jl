### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 7a52e292-72d9-11eb-153c-d19df523db65
begin
	using Distributions
	using Random
	using LinearAlgebra

	using Printf
end

# ╔═╡ 4eda75bc-896a-11eb-2ab1-7580d3fcf4d1
begin
	supertype(Matrix)
	DenseMatrix
	AbstractMatrix
end

# ╔═╡ c60a0d20-72c0-11eb-30f4-475ad9e6fdaf
begin
	# TODO: rename to `BernoulliNode`? These are exported, certainly confilicting!
	struct Gaussian end
	struct Bernoulli end
	struct Unitary end

	Grad{T} = Tuple{AbstractMatrix{T}, AbstractVector{T}, AbstractVector{T}}
	Samples{T} = NTuple{4, AbstractMatrix{T}}

	abstract type AbstractRBM{T,V,H} end

	"""
	Restricted Boltzmann Machine, parametrized by element type T, visible
	unit type V and hidden unit type H.
	"""
	struct RBM{T,V,H} <: AbstractRBM{T,V,H}
		W::AbstractMatrix{T}     # matrix of weights between vis and hid vars
		vbias::AbstractVector{T} # biases for visible variables
		hbias::AbstractVector{T} # biases for hidden variables
	end

	# TODO: rewrite nicely
	function vbias_init(::Type{Bernoulli}, X; eps=1e-8)
	  p = mean(X, dims = 2)

	  p = max.(p, eps)
	  p = min.(p, 1-eps)

	  v = log.(p ./ (1 .- p))
	  dropdims(v, dims = 2)
	end

	function vbias_init(::Type{Unitary}, X; kwargs...)
	  vbias_init(Bernoulli, X; kwargs...)
	end

	"""
	Construct RBM. Parameters:
	 * T - type of RBM parameters (e.g. weights and biases; by default, Float32)
	 * V - type of visible units
	 * H - type of hidden units
	 * n_vis - number of visible units
	 * n_hid - number of hidden units
	Optional parameters:
	 * sigma - variance to use during parameter initialization
	"""
	# TODO: better manage the weights init!
	function RBM(
		T::Type, V::Type, H::Type, n_vis::Int, n_hid::Int;
			σ::Float32=1f-2, X=nothing
		)

		vbias = X == nothing ? zeros(n_vis) : vbias_init(V, X)

		RBM{T,V,H}(
			map(T, rand(Normal(0, σ), n_vis, n_hid)),
			vbias,
			zeros(n_hid)
		)
	end

	RBM(V::Type, H::Type, n_vis::Int, n_hid::Int; σ=1f-2, X=nothing) =
		RBM(Float32, V, H, n_vis, n_hid; σ=σ, X=X)

	# some well-known RBM kinds
	"""Same as RBM{Float32,Degenerate,Bernoulli}"""
	BernoulliRBM(n_vis::Int, n_hid::Int; σ=1f-2, X=nothing) =
		RBM(Float32, Bernoulli, Bernoulli, n_vis, n_hid; σ=σ, X=X)
end

# ╔═╡ 7d4764bc-7786-11eb-0eb9-3d4ac85b9d6c
begin
	σ(x) = 1 / (1 + ℯ^-x)

	function sample!(::Type{Bernoulli}, x::AbstractMatrix{T}) where T
		# benchmark this!!
		# `SpecializedArray` is introduced to make sure that the sampled values
		# `r` have the same type as `x`. This is beneficial in general and
		# necessary on GPU.
		SpecializedArray = typeof(x)
		r = rand(T, size(x)...) |> SpecializedArray
		@. x = T(r < x)
	end

	function sample!(::Type{Unitary}, x::AbstractMatrix{T}) where T
		x
	end

	function nonlinearity!(::Type{Bernoulli}, x::AbstractMatrix{T}) where T
		@. x = σ(x)
	end

	nonlinearity!(::Type{Unitary}, x::AbstractMatrix{T}) where T = nonlinearity!(Bernoulli, x)

	function affine(W::AbstractMatrix{T}, bias::AbstractVector{T}, x::AbstractMatrix{T}) where T
		W * x .+ bias
	end

	function sample_hiddens(rbm::AbstractRBM{T,V,H}, v::AbstractMatrix{T}) where {T,V,H}
		p = affine(rbm.W', rbm.hbias, v)
		nonlinearity!(H, p)
		sample!(H, p)

		p
	end

	function sample_visibles(rbm::AbstractRBM{T,V,H}, h::AbstractMatrix{T}) where {T,V,H}
		p = affine(rbm.W, rbm.vbias, h)
		nonlinearity!(V, p)
		sample!(V, p)

		p
	end
end

# ╔═╡ 3fb0fe53-254e-4115-b645-62316ba5504a
randn(Float32, (2, 10)...)

# ╔═╡ 177a5cde-1af2-42ec-8400-00646929775f
begin
	function init_samples(::T, ::Type{Bernoulli}, size::NTuple{2, Int}) where T
		rand(T.([0, 1]), size...)
	end
	
	function init_samples(::T, ::Type{Unitary}, size::NTuple{2, Int}) where T
		rand(T, size...)
	end
	
	function init_samples(::T, ::Type{Gaussian}, size::NTuple{2, Int}) where T
		randn(T, size...)
	end
end

# ╔═╡ 01c89b7a-72c0-11eb-1061-2d697f2a980c
begin
	import Base: iterate

	repeat(iter::I, n::Int) where I = Iterators.take(Iterators.cycle(iter), n)

	function consume(iter)
		x = nothing
		for y in iter x = y end
		return x
	end

	struct TeeIterable{I}
    	iter::I
    	fun::Function
	end

	tee(iter::I, fun::Function) where {I} = TeeIterable{I}(iter, fun)

	struct PickIterable{I}
		iter::I
		period::UInt
	end

	function iterate(iter::PickIterable, state=iter.iter)
		current = iterate(state)
		if current === nothing return nothing end
		for i = 1:iter.period-1
			next = iterate(state, current[2])
			if next === nothing return current[1], Iterators.rest(state, current[2]) end
			current = next
		end
		return current[1], Iterators.rest(state, current[2])
	end

	pick(iter::I, period) where I = PickIterable{I}(iter, period)

	struct StopwatchIterable{I}
		iter::I
	end

	function iterate(iter::StopwatchIterable)
		t0 = time()
		next = iterate(iter.iter)
		return dispatch(iter, t0, t0, next)
	end

	function iterate(iter::StopwatchIterable, (ts, state))
		next = iterate(iter.iter, state)
		return dispatch(iter, ts..., next)
	end

	function dispatch(iter::StopwatchIterable, t0, t_last, next)
		if next === nothing return nothing end
		t = time()
		return ((t_last=t-t_last, t_total=t-t0), next[1]), ((t0, t), next[2])
	end

	stopwatch(iter::I) where I = StopwatchIterable{I}(iter)

	struct GibbsSampling{T,V,H}
	    rbm::AbstractRBM{T,V,H}
		init::AbstractMatrix{T}
	end

	function GibbsSampling(rbm::AbstractRBM{T,V,H}, batch_size::Int) where {T,V,H}
		nv, _ = size(rbm.W)
		v0 = init_samples(T, V, (nv, batch_size))

		GibbsSampling(rbm, v0)
	end

	struct Training{T,V,H}
		rbm::AbstractRBM{T,V,H}
		X::AbstractMatrix{T}
		epoch_iter::Function
		state::NamedTuple
	end

	function iterate(iter::GibbsSampling{T,V,H}, state) where {T,V,H}
		rbm = iter.rbm
		v0, h0 = state

		# TODO: rewrite `h`, and benchmark!
		h = sample_hiddens(rbm, v0)
		v = sample_visibles(rbm, h)

		(v, h), (v, h)
	end

	function iterate(iter::GibbsSampling{T,V,H}) where {T,V,H}
		iterate(iter, (iter.init, nothing))
	end

	function iterate(iter::Training{T,V,H}, state) where {T,V,H}
		rbm, state = iter.epoch_iter(iter.rbm, X, state)

		(rbm, state), state
	end

	# TODO: we don't need a state here! How to deal with it?
	# Maybe I should return the iter itself?
	function iterate(iter::Training{T,V,H}) where {T,V,H}
		iterate(iter, iter.state)
	end

	function iterate(iter::TeeIterable, args...)
    	next = iterate(iter.iter, args...)
    	if next !== nothing iter.fun(next[1]) end
    	return next
	end

end

# ╔═╡ c7c3bfc0-7785-11eb-32c1-2bedf825cf4e
begin
	# TODO: we can just shuffle in-place modifying X...
	function batchify(X::AbstractMatrix{T}, batch_size::Int; shuffled=true) where T
		n = size(X, 2)
		idxs = shuffled ? randperm(n) : 1:n
		ranges = Iterators.partition(idxs, batch_size)
		map(range -> X[:, range], ranges)
	end
	
	function CDk(rbm::AbstractRBM{T}, X::AbstractMatrix{T}, state::NamedTuple) where T
		CDk(rbm, X, state.k), state
	end

	function CDk(rbm::AbstractRBM{T}, X::AbstractMatrix{T}, k::Int) where T
		sampler = GibbsSampling(rbm, X)

		(v1, h0), rest = Iterators.peel(sampler)
		vk, hk = consume(Iterators.take(rest, k))

		X, h0, vk, hk
	end

	function dW(
		vpos::AbstractMatrix{T}, hpos::AbstractMatrix{T},
		vneg::AbstractMatrix{T}, hneg::AbstractMatrix{T}) where T

		n = size(vpos, 2)
		ΔW = vpos * hpos'
		mul!(ΔW, vneg, hneg', -1/n, 1/n)

		ΔW
	end

	function db(vpos::AbstractMatrix{T}, vneg::AbstractMatrix{T}) where {T}
		Δb = dropdims(mean(vpos - vneg, dims=2), dims=2)

		Δb
	end

	function dc(hpos::AbstractMatrix{T}, hneg::AbstractMatrix{T}) where {T}
		Δc = dropdims(mean(hpos - hneg, dims=2), dims=2)

		Δc
	end

	function dΘ(rbm::AbstractRBM{T}, samples::Samples{T}, state::NamedTuple) where T
		dΘ(rbm, samples), state
	end
	
	function dΘ(rbm::AbstractRBM{T}, samples::Samples{T}) where T
		vpos, hpos, vneg, hneg = samples

		ΔW = dW(vpos, hpos, vneg, hneg)
		Δb = db(vpos, vneg)
		Δc = dc(hpos, hneg)
		ΔΘ = (ΔW, Δb, Δc)

		ΔΘ
	end

	function grad_apply_learning_rate!(rbm::AbstractRBM{T}, ΔΘ::Grad{T}, α) where T
		@. lmul!(α, ΔΘ)
		
		ΔΘ
	end

	function update_Θ!(rbm::AbstractRBM{T}, ΔΘ::Grad{T}) where T
		ΔW, Δb, Δc = ΔΘ
		@. rbm.W += ΔW
		@. rbm.vbias += Δb
		@. rbm.hbias += Δc
		
		rbm
	end

	function update!(rbm::AbstractRBM{T}, ΔΘ::Grad{T}, state::NamedTuple) where T
		update!(rbm, ΔΘ; state.α), state
	end
	
	function update!(rbm::AbstractRBM{T}, ΔΘ::Grad{T}; α::Float32=1f-2) where T
		grad_apply_learning_rate!(rbm, ΔΘ, α)
		update_Θ!(rbm, ΔΘ)
		
		rbm
	end

	function flip(::Type{Bernoulli}, value)
  		1 - value
	end

	function flip(::Type{Unitary}, value)
  		1 - value
	end

	function pseudolikelihood(
		rbm::AbstractRBM{T,V,H}, batch::AbstractMatrix{T};
		n_vars::Int=1
	) where {T,V,H}

		d, n = size(batch)
		idxs = sample(1:d, n_vars; replace=false)
		corrupted = copy(batch)
		corrupted[idxs, :] = flip.(V, batch[idxs, :])

		# TODO: maybe there's a better looking form (with some dots)?
		log_p(rbm, v) =
			rbm.vbias' * v .+ sum(1 .+ exp.(rbm.W' * v .+ rbm.hbias); dims=1)

		mean(log.(σ.(log_p(rbm, corrupted) - log_p(rbm, batch))))
	end

	function partial(f, args; kwargs...)
		(new_args...) -> f(new_args..., args...; kwargs...)
	end

	# TODO: we want the loggers to be composable!
	function console_logger(rbm, X, state)
		batch = X[:, sample(1:size(X, 2), state.batch_size)]
		pl = pseudolikelihood(rbm, batch)

		@info @sprintf(
			"Epoch %4d | PL %4.4f | Time (epoch/total) %.3f/%.3f",
			state.epoch,
			pl,
			state.t_last,
			state.t_total
		)
	end
	
	default_state = (
		α=1f-3,
		batch_size=10,
		k=1,
		shuffled=true,
		sampler=CDk,
		gradient=dΘ,
		n_epochs=1,
		logger=console_logger,
		log_every=1,
	)
	
	# TODO: where to ensure type for `α`?
	init_state(state) = merge(default_state, state)
	
	function merge_iter_state(iter::I) where I
		((rbm, merge(state, iter_state)) for (iter_state, (rbm, state)) in iter)
	end
	
	function batchify(
		X::AbstractMatrix{T},
		state::NamedTuple
	) where T
		batches = batchify(X, state.batch_size; shuffled=state.shuffled)
		
		batches, state
	end
	
	function sampler(
		rbm::AbstractRBM{T},
		batch::AbstractMatrix{T},
		state::NamedTuple
	) where {T}
		state.sampler(rbm, batch, state)
	end
	
	function gradient(
		rbm::AbstractRBM{T},
		samples::I,
		state::NamedTuple
	) where {T,I}
		Iterators.map(sample -> state.gradient(rbm, sample, state), samples)
	end
end

function train_batch(rbm::AbstractRBM{T}, batch::AbstractMatrix{T}, state::NamedTuple) where T
	sample, state = state.sampler(rbm, batch, state)
	ΔΘ, state = state.gradient(rbm, sample, state)
	rbm, state = update!(rbm, ΔΘ, state)
end

function train_epoch(rbm::AbstractRBM{T}, X::AbstractMatrix{T}, state::NamedTuple) where T
	batches, state = batchify(X, state)

	for batch in batches
		rbm, state = train_batch(rbm, batch, state)
    end

	rbm, state
end

# NOTE: the iterator approach is probably overkill for the batches loop,
# this can be rewritten very easily as a `for` loop. No loss in readibility
# (rather a gain, here)
# and we probably don't need the extra flexibility that we have now
# (you rarely want to do stuff "per-batch")
struct TrainingEpoch{T,V,H}
	rbm::AbstractRBM{T,V,H}
	X::AbstractArray{T}
	state::NamedTuple
end

function iterate(iter::TrainingEpoch{T,V,H}, state) where {T,V,H}
	if length(state.remaining_batches) < 1
		return nothing
    end

	batch, remaining_batches = Iterators.peel(state.remaining_batches)
	print("Batch ", typeof(batch))
	rbm, state = train_batch(iter.rbm, batch, state)

	rbm, merge(state, (remaining_batches=remaining_batches,))
end

# TODO: define a macro to easily update the state?
function iterate(iter::TrainingEpoch{T,V,H}) where {T,V,H}
	batches, state = batchify(iter.X, iter.state)

	iter.rbm, merge(state, (remaining_batches=batches,))
end

function fit!(rbm::AbstractRBM{T}, X::AbstractMatrix{T}, args::NamedTuple) where T
	state = init_state(args)

	epochs = Training(rbm, X, train_epoch, state)
	epochs = Iterators.take(epochs, state.n_epochs)

	epochs = enumerate(epochs)
	epochs = (((epoch=i,), (rbm, state)) for (i, (rbm, state)) in epochs)
	epochs = epochs |> merge_iter_state
		
	epochs = stopwatch(epochs) |> merge_iter_state
	epochs = pick(epochs, state.log_every)
		
	epochs = tee(
		epochs,
		((rbm, state),) -> state.logger(rbm, X, state)
	)
		
	rbm, state = consume(epochs)
	rbm, state
end
	
function fit!(rbm::AbstractRBM{T}, X::AbstractMatrix{T}; args...) where T
	fit!(rbm, X, NamedTuple(args))
end

# ╔═╡ e961bf40-75ef-11eb-1802-6355ab8eabf0
begin
	rbm = BernoulliRBM(100, 20)

	bs = 10
	X = rand(size(rbm.W, 1), 10*bs) .|> Float32
	CDk(rbm, X, default_state)
	
	fit!(rbm, X; n_epochs=10, log_every=1)
end

# ╔═╡ afc06fde-8cc7-4fd3-8358-79819a044a30
begin
	Iterators.map(x -> x^2, (1, 2, 3))
end

# ╔═╡ bf6b607a-737b-11eb-1cd3-2b4d4e95b1b8
begin
	# test batchify
	function test_batchify()
		d = 10
		n = 100
		bs = 10
		nh = 20

		X = ones(d, n) .|> Float32

		batches = batchify(X, bs)

		rbm = BernoulliRBM(d, nh)
		sum(size(sample_hiddens(rbm, batch), 2) for batch in batches)


	end

	test_batchify()
end

# ╔═╡ Cell order:
# ╠═7a52e292-72d9-11eb-153c-d19df523db65
# ╠═4eda75bc-896a-11eb-2ab1-7580d3fcf4d1
# ╠═c60a0d20-72c0-11eb-30f4-475ad9e6fdaf
# ╠═7d4764bc-7786-11eb-0eb9-3d4ac85b9d6c
# ╠═3fb0fe53-254e-4115-b645-62316ba5504a
# ╠═177a5cde-1af2-42ec-8400-00646929775f
# ╠═01c89b7a-72c0-11eb-1061-2d697f2a980c
# ╠═c7c3bfc0-7785-11eb-32c1-2bedf825cf4e
# ╠═7079e557-c26c-4572-8704-913373d3dc32
# ╠═e961bf40-75ef-11eb-1802-6355ab8eabf0
# ╠═afc06fde-8cc7-4fd3-8358-79819a044a30
# ╠═bf6b607a-737b-11eb-1cd3-2b4d4e95b1b8
