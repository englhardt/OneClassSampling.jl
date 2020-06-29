@testset "Density Util" begin
    Random.seed!(0)
    d, l = hcat(randn(5, 95), randn(5, 5) .+ 2), fill(:inlier, 100)
    l[end-4:end] .= :outlier
    gamma = 0.5
    c = KDECache(d, gamma)

    @testset "Threshold Strategies" begin
        @testset "FixedThresholdStrategy" begin
            n = 3
            s = FixedThresholdStrategy(n)
            @test n == calculate_threshold(s, d, l, gamma)
            @test n == calculate_threshold(s, c, l)
        end

        for x in [NumLabelTresholdStrategy, MaxDensityThresholdStrategy, OutlierPercentageThresholdStrategy]
            @testset "$x" begin
                @test_throws ArgumentError x(-1)
                @test_throws ArgumentError x(2)
                s = x(0.1)
                t, i, o = split_masks(s, c, l)
                @test t > 0
                @test all(xor.(i, o))
                @test any(o)
            end
        end

        for x in [OutlierPercentageThresholdStrategy, GroundTruthThresholdStrategy]
            @testset "$x" begin
                s = x()
                t, i, o = split_masks(s, c, l)
                @test t > 0
                @test all(xor.(i, o))
                @test any(o)
            end
        end
    end

    @testset "PWC" begin
        pwc = PWC(d, gamma)
        @test_throws ArgumentError predict(pwc, d)
        calculate_threshold!(pwc, l)
        pred = predict(pwc, d)
        @test length(pred) == size(d, 2)
        @test !all(pred .== :inlier)
    end
end
