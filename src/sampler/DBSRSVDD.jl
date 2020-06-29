"""
Li, Z., Wang, L., Yang, Y., Du, X. and Song, H., 2019.
Health evaluation of MVB based on SVDD and sample reduction.
IEEE Access, 7, pp.35330-35343.

Density-Based Sample Reduction for Support Vector Data Description (DBSRSVDD)
Parameter guideline from the paper:
min_pts ∈ [5, 6, 7, 8, 9, 10]
eps ∈ [0.5, 1.0]
"""
struct DBSRSVDD <: Sampler
    min_pts::Int
    eps::Real
    function DBSRSVDD(min_pts::Int=7, eps::Real=0.5)
        min_pts > 0 || throw(ArgumentError("Invalid number of min_pts $(min_pts)."))
        eps >= 0 || throw(ArgumentError("Invalid threshold $(eps)."))
        return new(min_pts, eps)
    end
end

function sample(sampler::DBSRSVDD, data::DataSet, labels::Vector{Symbol})::BitArray{1}
    d_ij = Distances.pairwise(Distances.Euclidean(), data, dims=2)
    threshold = sampler.eps * 1 / length(labels)^2 * sum(d_ij)
    c = dbscan(d_ij, threshold, sampler.min_pts)
    sample_mask = c.assignments .== 0
    return sample_mask
end
