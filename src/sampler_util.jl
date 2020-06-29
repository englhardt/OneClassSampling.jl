function cosine_similarity(x, y)
    return dot(x, y) / (norm(x) * norm(y))
end

function train_and_find_support_vectors(model, data, solver)

    SVDD.set_data!(model, data)
    try
        SVDD.fit!(model, solver)
    catch e
        # For small data sets fall back to returning all points as SVs
        if size(data, 2) <= 5
            return collect(1:size(data, 2))
        end
        throw(SamplingException("Failed fitting model on data set of size $(size(data, 2)) due to $e."))
    end
    return SVDD.get_support_vectors(model)
end

function mcc(classification, ground_truth; pos_class = :outlier, neg_class = :inlier)::Float64
    @assert length(classification) == length(ground_truth)
    tp = sum((classification .== pos_class) .& (ground_truth .== pos_class))
    fp = sum((classification .== pos_class) .& (ground_truth .== neg_class))
    tn = sum((classification .== neg_class) .& (ground_truth .== neg_class))
    fn = sum((classification .== neg_class) .& (ground_truth .== pos_class))

    d = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return d == 0 ? 0.0 : (tp * tn - fp * fn) / d
end
