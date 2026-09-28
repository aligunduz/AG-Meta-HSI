using Test
using AGMetaHSI

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
