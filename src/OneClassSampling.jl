module OneClassSampling

using Memento

const LOGGER = getlogger(@__MODULE__)

function __init__()
    Memento.register(LOGGER)
    Memento.config!(LOGGER, "warn"; fmt="[{level} | {name}]: {msg}")
end

include("sampler_base.jl")

export
    Sampler,
    sample,
    SamplingException,

    RAPID,

    HDS,
    RandomRatioSampler, RandomNSampler,
    CSSVDD, BPS, DAEDS, DBSRSVDD, FBPE, HSC, HSR, IESRSVDD, KFNCBD, KMSVDD, NDPSR, RSSVDD,

    PreFilteringWrapper,

    KDECache,
    kde,

    ThresholdStrategy,
    FixedThresholdStrategy, NumLabelTresholdStrategy, MaxDensityThresholdStrategy,
    OutlierPercentageThresholdStrategy, GroundTruthThresholdStrategy,
    calculate_threshold, calculate_threshold!, split_masks, predict,
    PWC

end
