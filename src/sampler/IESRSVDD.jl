"""
Li, D., Wang, Z., Cao, C. and Liu, Y., 2018.
Information entropy based sample reduction for support vector data description.
Applied Soft Computing, 71, pp.1153-1160.

Information Entropy based Sample Reduction for Support Vector Data Description (IESRSVDD)Parameter guideline from the paper, given a data set of size N:
eps (on benchmark data eps ∈ [0.5, 0.6, 0.7])

Simplification of equations:

Dis_ij = K(x_i * x_i) + K(x_j * x_j) - 2 K(x_i * x_j) (Eq. 10)
       = 2 (1 - K(x_i * x_j)) (with RBF Kernel identity == 1)

p_ij = dis_ij / sum(dis_ik for k in 1:n) (Eq. 11)
     = 2 (1 - K(x_i * x_j)) / sum(2 (1 - K(x_i * x_k)) for k in 1:n)
     = (1 - K(x_i * x_j)) / sum(1 - K(x_i * x_k) for k in 1:n)

p_ii = 1

H_i = - sum(p_ik log_2(p_ik) for k in 1:n) (Eq. 12)
    = - sum(p_ik log_2(p_ik) for k in 1:n if k != i) (p_ii == 1 and log_2(1) = 0)
"""
mutable struct IESRSVDD <: Sampler
    kde_cache::Union{KDECache, Nothing}
    gamma::Union{Symbol, Real}
    eps::Real
    function IESRSVDD(kde_cache::Union{KDECache, Nothing}, gamma::Union{Symbol, Real}, eps::Real=0.5)
        (eps >= 0 && eps <= 1) || throw(ArgumentError("Invalid threshold $(eps)."))
        return new(kde_cache, gamma, eps)
    end
end

function IESRSVDD(kde_cache::KDECache, eps::Real=0.5)
    return IESRSVDD(kde_cache, :precomputed, eps)
end

function IESRSVDD(gamma::Union{Symbol, Real}, eps::Real=0.5)
    return IESRSVDD(nothing, gamma, eps)
end

function sample(sampler::IESRSVDD, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    if sampler.kde_cache === nothing || sampler.kde_cache.data != data
        sampler.kde_cache = KDECache(data, sampler.gamma)
    end
    dis_ij = 1 .- sampler.kde_cache.K
    p_ij = dis_ij ./ sum(dis_ij, dims=1)
    p_ij[diagind(p_ij)] .= 1
    h_i = [-sum(p_ij[i, k] * log(2, p_ij[i, k]) for k in 1:size(p_ij, 2)) for i in 1:size(p_ij, 2)]
    h_i_min, h_i_max = extrema(h_i)
    θ = h_i_max - sampler.eps * (h_i_max - h_i_min)
    sample_mask = h_i .> θ
    return sample_mask
end
