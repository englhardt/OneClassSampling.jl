function get_gaussian_kernel(data::DataSet, gamma::Union{Symbol, Real})::Kernel
    if gamma == :scott
        gamma = SVDD.rule_of_scott(data)
    elseif gamma == :mean_crit
        gamma = SVDD.mean_criterion(data)
    elseif gamma == :mod_mean_crit
        gamma = SVDD.modified_mean_criterion(data)
    elseif isa(gamma, Symbol)
        throw(ArgumentError("Unsupported gamma init method '$(gamma)'"))
    end
    return get_gaussian_kernel(gamma)
end

function get_gaussian_kernel(gamma::Real)::Kernel
    return MLKernels.GaussianKernel(gamma)
end

kde(x; gamma::Union{Symbol, Real}=:mod_mean_crit)::Vector{Float64} = kde(x, x, gamma=gamma)

function kde(x, y; gamma::Union{Symbol, Real}=:mod_mean_crit)::Vector{Float64}
    return kde(x, y, get_gaussian_kernel(x, gamma))
end

kde(x::DataSet, kernel_fct::Kernel)::Vector{Float64} = kde(x, x, kernel_fct)

function kde(x::DataSet, y::DataSet, kernel_fct::Kernel)::Vector{Float64}
    return sum(MLKernels.kernelmatrix(Val(:col), kernel_fct, x, y), dims=1)[1, :]
end


struct KDECache
    data::DataSet
    kernel_fct::Kernel
    K::Matrix{Float64}

    function KDECache(data::DataSet, gamma::Union{Symbol, Real}=:mod_mean_crit)
        kernel_fct = get_gaussian_kernel(data, gamma)
        K = MLKernels.kernelmatrix(Val(:col), kernel_fct, data, data)
        return new(data, kernel_fct, K)
    end
end

function Base.show(io::IO, cache::KDECache)
    print(io, "KDECache(DATA, ")
    print(io, cache.kernel_fct)
    print(io, ", K)")
end

function kde(cache::KDECache)::Vector{Float64}
    return sum(cache.K, dims=1)[1, :]
end

function kde(cache::KDECache, x::BitArray)::Vector{Float64}
    return kde(cache, x, x)
end

function kde(cache::KDECache, x::BitArray, y::BitArray)::Vector{Float64}
    return sum(cache.K[x, y], dims=1)[1, :]
end

function kde(cache::KDECache, y::DataSet)::Vector{Float64}
    return kde(cache.data, y, cache.kernel_fct)
end

function kde(cache::KDECache, x::BitArray, y::DataSet)::Vector{Float64}
    return kde(cache.data[:, x], y, cache.kernel_fct)
end
