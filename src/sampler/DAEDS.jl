"""
Hu, C., Zhou, B. and Hu, J., 2014, July.
Fast Support Vector Data Description training using edge detection on large datasets.
In 2014 International Joint Conference on Neural Networks (IJCNN) (pp. 2176-2182). IEEE.

Density-angle based edge detection sampling (DAEDS)
Parameter:
k: nearest neighborhood (experiments k ∈ [30, 35])
eps: ratio of candidate set points (experiments eps ∈ [0.1, 0.3, 0.4])
delta: ratio of complement set points (experiments delta ∈ [0.3, 0.4, 0.5, 0.6])
"""
mutable struct DAEDS <: Sampler
    k::Int
    eps::Real
    delta::Real
    function DAEDS(k::Int=30, eps::Real=0.1, delta::Real=0.3)
        k > 0 || throw(ArgumentError("Non-positive number of neighbors to consider $(k)."))
        eps >= 0 || throw(ArgumentError("Negative ratio of candidate points $(eps)."))
        delta >= 0 || throw(ArgumentError("Negative ratio of complement points $(delta)."))
        return new(k, eps, delta)
    end
end

function reachability_distance(data, neighbor_indices)
    reach_dist = zeros(size(data, 2), size(data, 2))
    d_ij = Distances.pairwise(Distances.Euclidean(), data, dims=2)
    for i in 1:size(data, 2)
        for j in 1:size(data, 2)
            k_dist = d_ij[i, neighbor_indices[i][end]]
            reach_dist[i, j] = max(k_dist, d_ij[i, j])
        end
    end
    return reach_dist
end

function local_reachability_distance(data, neighbor_indices)
    reach_dist = reachability_distance(data, neighbor_indices)
    lrd = zeros(size(reach_dist, 2))
    for i in 1:size(reach_dist, 2)
        lrd[i] = mean(reach_dist[i, neighbor_indices[i]])
    end
    return lrd
end

function local_outlier_factor(data, neighbor_indices)
    lrd = local_reachability_distance(data, neighbor_indices)
    lof = zeros(size(data, 2))
    for i in 1:size(data, 2)
        lof[i] = mean(lrd[j] / lrd[i] for j in neighbor_indices[i])
    end
    return lof
end

function support_neighbor(data, neighbor_indices, lof)
    sn = zeros(Int, size(data, 2))
    for i in 1:size(data, 2)
        _, idx = findmin(lof[neighbor_indices[i]])
        sn[i] = idx
    end
    return sn
end

function angle_variance(data, neighbor_indices, sn, C1)
    v_i = zeros(length(C1))
    for (i, c_i) in enumerate(C1)
        angles = [cosine_similarity(data[:, sn[c_i]] .- data[:, c_i],
                                    data[:, j] .- data[:, c_i]) for j in neighbor_indices[c_i]]
        v_i[i] = var(angles)
    end
    return v_i
end

function sample(sampler::DAEDS, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    n = size(data, 2)
    sampler.k = min(n - 1, sampler.k)
    # Step 1: choose candidate sets C1 and C2 based on LOF
    tree = KDTree(data)
    neighbor_indices = [knn(tree, data[:, i], sampler.k + 1, true)[1][2:end] for i in 1:n]
    lof = local_outlier_factor(data, neighbor_indices)
    lof_sorted_indices = sortperm(lof)
    C1 = lof_sorted_indices[end-floor(Int, sampler.delta * n)+1:end]
    C2 = lof_sorted_indices[1:floor(Int, sampler.eps * n)]
    # Step 2: determine support neighbors
    sn = support_neighbor(data, neighbor_indices, lof)
    # Step 3: detect edges
    v_i = angle_variance(data, neighbor_indices, sn, C1)
    v_i_sorted_indices = sortperm(v_i, rev=true)
    A = v_i_sorted_indices[1:floor(Int, sampler.eps * n)]
    # Step 4: combine A and C2
    sample_mask = falses(n)
    sample_mask[A] .= true
    sample_mask[C2] .= true
    return sample_mask
end
