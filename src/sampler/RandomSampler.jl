abstract type RandomSampler <: Sampler end

struct RandomRatioSampler <: RandomSampler
    target_ratio::Real
    function RandomRatioSampler(target_ratio::Real)
        0 <= target_ratio <= 1 || throw(ArgumentError("Invalid sampling ratio $(target_ratio)."))
        return new(target_ratio)
    end
end

function sample(sampler::RandomRatioSampler, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    return rand_sample(data, ceil(Int, sampler.target_ratio * length(labels)))
end

struct RandomNSampler <: RandomSampler
    n::Int
    function RandomNSampler(n::Int)
        n >= 0 || throw(ArgumentError("Invalid sampling amount $(n)."))
        return new(n)
    end
end

function sample(sampler::RandomNSampler, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    return rand_sample(data, sampler.n)
end

function rand_sample(data::DataSet, n::Int)::BitArray{1}
    m = size(data, 2)
    sample_mask = falses(m)
    sample_mask[randperm(m)[1:n]] .= true
    return sample_mask
end
