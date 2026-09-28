using Test
using AGMetaHSI
using Lux, NNlib, Random, Zygote, Serialization

@testset "Few-label split" begin
    labels = [1 1 1 1 1 1; 2 2 2 2 2 2]
    a = make_pixel_split(labels; seed=93)
    b = make_pixel_split(labels; seed=93)
    @test a.train == b.train
    @test length(a.train) == 10
    @test length(a.test) == 2
    @test validate_pixel_split(labels, a)
    @test count(x -> x[3] == 1, a.train) == 5
    @test count(x -> x[3] == 2, a.train) == 5
    @test_throws ArgumentError make_pixel_split(labels; k=6)
    @test_throws ArgumentError validate_pixel_split(labels,
        PixelSplit(93, 5, [a.train; a.train[1]], a.test))

    mktempdir() do dir
        path = joinpath(dir, "split.tsv")
        save_pixel_split(path, a)
        loaded = load_pixel_split(path)
        @test loaded.seed == a.seed
        @test loaded.k == a.k
        @test loaded.train == a.train
        @test loaded.test == a.test
        @test validate_pixel_split(labels, loaded)
    end
end

@testset "SSARN forward, equations and differentiation (no training)" begin
    rng = MersenneTwister(93)
    model = ssarn()
    ps, st = Lux.setup(rng, model)
    x = randn(rng, Float32, 9, 9, 103, 1, 2)
    @test size(check_ssarn(model, ps, st, x[:, :, :, :, 1:1])) == (9, 1)
    state = Lux.testmode(st)
    z, _ = model.stem(x, ps.stem, state.stem)
    @test size(z) == (9, 9, 8, 32, 2)
    z, _ = model.spectral_residual(z, ps.spectral_residual, state.spectral_residual)
    @test size(z) == (9, 9, 8, 32, 2)
    z, _ = model.collapse(z, ps.collapse, state.collapse)
    @test size(z) == (7, 7, 32, 2)
    z, _ = model.spatial_residual(z, ps.spatial_residual, state.spatial_residual)
    @test size(z) == (7, 7, 32, 2)

    # Check the attention axes/orientation against a direct per-sample Eq. (7).
    att = AGMetaHSI.SpatialAttention(3)
    ap, ast = Lux.setup(rng, att)
    ax = randn(rng, Float32, 2, 2, 3, 2)
    actual, _ = att(ax, ap, ast)
    q, _ = att.query(ax, ap.query, ast.query)
    k, _ = att.key(ax, ap.key, ast.key)
    v, _ = att.value(ax, ap.value, ast.value)
    for b in 1:2
        Q = permutedims(reshape(q[:, :, :, b], 4, 3))
        K = permutedims(reshape(k[:, :, :, b], 4, 3))
        V = permutedims(reshape(v[:, :, :, b], 4, 3))
        expected = reshape(permutedims(V * NNlib.softmax(Q' * K; dims=2)), 2, 2, 3)
        @test actual[:, :, :, b] ≈ expected
    end

    y = Float32.([c == b for c in 1:9, b in 1:2])
    # Differentiate one synthetic batch; no optimizer update and no UP training.
    (loss, _), pullback = Zygote.pullback(p -> supervised_loss(model, p, st, x, y), ps)
    grads = pullback((one(loss), nothing))[1]
    finite_tree(a::AbstractArray) = all(isfinite, a)
    finite_tree(a::NamedTuple) = all(finite_tree, values(a))
    finite_tree(::Nothing) = true
    @test isfinite(loss)
    @test finite_tree(grads)
    @test any(!iszero, grads.classifier.weight)

    before, _ = model(x, ps, state)
    mktempdir() do dir
        path = joinpath(dir, "checkpoint.jls")
        serialize(path, (; ps, st))
        saved = deserialize(path)
        after, _ = model(x, saved.ps, Lux.testmode(saved.st))
        @test before == after
    end
end

@testset "Classification metrics" begin
    m = classification_metrics([8 2; 1 9])
    @test m.OA ≈ 0.85
    @test m.AA ≈ 0.85
    @test m.per_class ≈ [0.8, 0.9]
    @test m.kappa ≈ 0.7
    m = classification_metrics([2 0; 2 6])
    @test m.OA ≈ 0.8
    @test m.AA ≈ 0.875
    @test m.kappa ≈ (0.8 - 0.56) / (1 - 0.56)
    @test_throws ArgumentError classification_metrics([0 0; 1 1])
end

@testset "Center-pixel patch and border" begin
    cube = reshape(Float32.(1:24), 3, 4, 2)
    scene = HSIScene(cube, ones(Int, 3, 4))
    patch = extract_patch(scene, 1, 1; patch_size=3)
    @test size(patch) == (3, 3, 2)
    @test patch[2, 2, :] == cube[1, 1, :]
    @test all(iszero, patch[1, :, :])
    @test all(iszero, patch[:, 1, :])
    @test patch[3, 3, :] == cube[2, 2, :]
    @test_throws ArgumentError extract_patch(scene, 1, 1; patch_size=4)
    @test_throws BoundsError extract_patch(scene, 0, 1)
end
