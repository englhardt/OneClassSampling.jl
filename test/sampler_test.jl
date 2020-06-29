@testset "Sampler" begin
    Random.seed!(0)
    d, l = hcat(randn(5, 95), randn(5, 5) .+ 2), fill(:inlier, 100)
    l[end-4:end] .= :outlier
    gamma = 0.5
    c = KDECache(d, gamma)
    chunk_mask = trues(length(l))
    chunk_mask[1:5] .= false
    threshold_strat = MaxDensityThresholdStrategy(0.5)
    threshold_strat_gt = GroundTruthThresholdStrategy()

    init_strat = SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(MLKernels.GaussianKernel(gamma)),
                                             SVDD.FixedCStrategy(0.5))


    @testset "RandomSampler" begin
        @test_throws ArgumentError RandomRatioSampler(-1)
        s = RandomRatioSampler(1)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test all(sample_mask)
        s = RandomRatioSampler(0.5)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "RandomNSampler" begin
        @test_throws ArgumentError RandomNSampler(-1)
        s = RandomNSampler(length(l))
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test all(sample_mask)
        s = RandomNSampler(length(l) - 5)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "BPS" begin
        @test_throws ArgumentError BPS(-1, 0.1)
        @test_throws ArgumentError BPS(0, 0.1)
        @test_throws ArgumentError BPS(5, -1)
        s = BPS(nothing, 0.1)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "CSSVDD" begin
        @test_throws ArgumentError CSSVDD(-1, 1, 1, 1, 0.05, init_strat, TEST_SOLVER)
        @test_throws ArgumentError CSSVDD(0, 1, 1, 1, 0.05, init_strat, TEST_SOLVER)
        @test_throws ArgumentError CSSVDD(1, -1, 1, 1, 0.05, init_strat, TEST_SOLVER)
        @test_throws ArgumentError CSSVDD(1, 0, 1, 1, 0.05, init_strat, TEST_SOLVER)
        @test_throws ArgumentError CSSVDD(1, 1, -1, 1, 0.05, init_strat, TEST_SOLVER)
        @test_throws ArgumentError CSSVDD(1, 1, 0, 1, 0.05, init_strat, TEST_SOLVER)
        @test_throws ArgumentError CSSVDD(1, 1, 1, -1, 0.05, init_strat, TEST_SOLVER)
        @test_throws ArgumentError CSSVDD(1, 1, 1, 0, 0.05, init_strat, TEST_SOLVER)
        @test_throws ArgumentError CSSVDD(1, 1, 1, -1, -0.05, init_strat, TEST_SOLVER)
        @test_throws ArgumentError CSSVDD(1, 1, 1, 0, 1.05, init_strat, TEST_SOLVER)
        s = CSSVDD(0.1, 0.05, init_strat, TEST_SOLVER)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "DAEDS" begin
        @test_throws ArgumentError DAEDS(-1, 0.1, 0.3)
        @test_throws ArgumentError DAEDS(0, 0.1, 0.3)
        @test_throws ArgumentError DAEDS(5, -1, 0.3)
        @test_throws ArgumentError DAEDS(5, 0.1, -1)
        s = DAEDS(5, 0.1, 0.3)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "DBSRSVDD" begin
        @test_throws ArgumentError DBSRSVDD(-1, 0.1)
        @test_throws ArgumentError DBSRSVDD(0, 0.1)
        @test_throws ArgumentError DBSRSVDD(5, -1)
        s = DBSRSVDD(5, 0.5)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "FBPE" begin
        @test_throws ArgumentError FBPE(-1)
        @test_throws ArgumentError FBPE(0)
        s = FBPE(2)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "HSC" begin
        @test_throws ArgumentError HSC(-1, 1, threshold_strat, c, init_strat, TEST_SOLVER)
        @test_throws ArgumentError HSC(0, 1, threshold_strat, c, init_strat, TEST_SOLVER)
        @test_throws ArgumentError HSC(1, -1, threshold_strat, c, init_strat, TEST_SOLVER)
        @test_throws ArgumentError HSC(1, 0, threshold_strat, c, init_strat, TEST_SOLVER)
        s = HSC(5, 2, threshold_strat, :test, init_strat, TEST_SOLVER)
        @test_throws ArgumentError sample(s, d, l)
        s = HSC(5, 2, threshold_strat, c, init_strat, TEST_SOLVER)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "HSR" begin
        @test_throws ArgumentError HSR(-1, 0.1)
        @test_throws ArgumentError HSR(0, 0.1)
        @test_throws ArgumentError HSR(5, -1)
        s = HSR(5, 0.5)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "IESRSVDD" begin
        @test_throws ArgumentError IESRSVDD(c, -0.1)
        @test_throws ArgumentError IESRSVDD(c, 1.1)
        s = IESRSVDD(:foo, 0.1)
        @test_throws ArgumentError sample(s, d, l)
        s = IESRSVDD(c, 0.5)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "KFNCBD" begin
        @test_throws ArgumentError KFNCBD(-1, 0.1)
        @test_throws ArgumentError KFNCBD(0, 0.1)
        @test_throws ArgumentError KFNCBD(5, -0.1)
        @test_throws ArgumentError KFNCBD(5, 1.5)
        s = KFNCBD(20, 0.5)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "KMSVDD" begin
        @test_throws ArgumentError KMSVDD(-1, init_strat, TEST_SOLVER)
        @test_throws ArgumentError KMSVDD(0, init_strat, TEST_SOLVER)
        s = KMSVDD(2, init_strat, TEST_SOLVER)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "NDPSR" begin
        @test_throws ArgumentError NDPSR(-1, 5)
        @test_throws ArgumentError NDPSR(0, 5)
        @test_throws ArgumentError NDPSR(5, -1)
        s = NDPSR(20, 15)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "RSSVDD" begin
        @test_throws ArgumentError RSSVDD(-1, 1, 1, 1, init_strat, TEST_SOLVER)
        @test_throws ArgumentError RSSVDD(0, 1, 1, 1, init_strat, TEST_SOLVER)
        @test_throws ArgumentError RSSVDD(1, -1, 1, 1, init_strat, TEST_SOLVER)
        @test_throws ArgumentError RSSVDD(1, 0, 1, 1, init_strat, TEST_SOLVER)
        @test_throws ArgumentError RSSVDD(1, 1, -1, 1, init_strat, TEST_SOLVER)
        @test_throws ArgumentError RSSVDD(1, 1, 0, 1, init_strat, TEST_SOLVER)
        @test_throws ArgumentError RSSVDD(1, 1, 1, -1, init_strat, TEST_SOLVER)
        @test_throws ArgumentError RSSVDD(1, 1, 1, 0, init_strat, TEST_SOLVER)
        s = RSSVDD(10, 2, init_strat, TEST_SOLVER)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
    end

    @testset "RAPID" begin
        s = RAPID(threshold_strat, :test)
        @test_throws ArgumentError sample(s, d, l)
        s = RAPID(threshold_strat, c)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
        s = RAPID(threshold_strat_gt, c)
        sample_mask = sample(s, d, l)
        @test length(sample_mask) == length(l)
        @test !all(sample_mask)
        _, inlier_mask, _ = split_masks(threshold_strat_gt, c, l)
        @test all(inlier_mask[sample_mask])
    end

    @testset "HDS" begin
        @test_throws ArgumentError HDS(1.5, c)
        @test_throws ArgumentError HDS(-1.0, c)
        s = HDS(1.0, :foo)
        @test_throws ArgumentError sample(s, d, l)
        s = HDS(1.0, c, nothing)
        sample_mask1 = sample(s, d, l)
        @test length(sample_mask1) == length(l)
        @test all(sample_mask1)
        s = HDS(1.0, c, threshold_strat)
        sample_mask2 = sample(s, d, l)
        @test length(sample_mask2) == length(l)
        @test !all(sample_mask2)
        @test count(sample_mask1) > count(sample_mask2)
        s = HDS(0.1, c, threshold_strat)
        sample_mask3 = sample(s, d, l)
        @test length(sample_mask3) == length(l)
        @test count(sample_mask2) > count(sample_mask3)
    end

    @testset "PreFilteringWrapper" begin
        sampler = RandomRatioSampler(1.0)
        s = PreFilteringWrapper(sampler, :test, threshold_strat)
        @test_throws ArgumentError sample(s, d, l)
        s1 = PreFilteringWrapper(sampler, c, nothing)
        sample_mask1 = sample(s1, d, l)
        @test length(sample_mask1) == length(l)
        @test all(sample_mask1)
        s2 = PreFilteringWrapper(sampler, c, threshold_strat)
        sample_mask2 = sample(s2, d, l)
        @test length(sample_mask2) == length(l)
        @test count(sample_mask1) > count(sample_mask2)
        sampler = RandomRatioSampler(0.5)
        s3 = PreFilteringWrapper(sampler, c, threshold_strat)
        sample_mask3 = sample(s3, d, l)
        @test length(sample_mask3) == length(l)
        @test count(sample_mask2) > count(sample_mask3)
    end
end
