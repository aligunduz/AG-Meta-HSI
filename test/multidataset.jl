using Test, AGMetaHSI, MAT, Random, Lux, TOML

@testset "Dataset-aware SSARN baseline" begin
    @test dataset_spec("sa").bands == 204
    @test dataset_spec("IP").classes == 16
    @test_throws ArgumentError dataset_spec("unknown")
    @test_throws ArgumentError ssarn(; bands=20, classes=16)

    rng = MersenneTwister(17)
    ip_model = ssarn(; bands=200, classes=16)
    ip_ps, ip_st = Lux.setup(rng, ip_model)
    @test size(check_ssarn(ip_model, ip_ps, ip_st,
                           rand(rng, Float32, 9, 9, 200, 1, 1); classes=16)) == (16, 1)
    labels = repeat(reshape(collect(1:16), 16, 1), 1, 7)
    cube = rand(rng, Float32, 16, 7, 204)
    mktempdir() do root
        data_dir = joinpath(root, "data")
        mkpath(data_dir)
        matwrite(joinpath(data_dir, "Salinas_corrected.mat"), Dict("salinas_corrected" => cube))
        matwrite(joinpath(data_dir, "Salinas_gt.mat"), Dict("salinas_gt" => labels))
        scene = load_hsi_dataset("SA", data_dir)
        @test size(scene.cube) == (16, 7, 204)
        split = make_pixel_split(scene.labels; k=5, seed=93)
        @test length(split.train) == 80
        @test length(split.test) == 32
        split_path = joinpath(root, "sa_split.tsv")
        save_pixel_split(split_path, split)
        @test validate_dataset_protocol(scene, load_pixel_split(split_path), "SA", 93, 5)
        @test size(patch_batch(scene, split.train[1:1])) == (9, 9, 204, 1, 1)

        model = ssarn(; bands=204, classes=16)
        ps, st = Lux.setup(rng, model)
        @test size(check_ssarn(model, ps, st, patch_batch(scene, split.train[1:1]);
                               classes=16)) == (16, 1)

        output_dir = joinpath(root, "run")
        metrics = run_supervised(; dataset="SA", data_dir, split_path, output_dir,
                                 seed=93, k=5, epochs=1, batch_size=16, train=true)
        @test 0 <= metrics.OA <= 1
        @test 0 <= metrics.AA <= 1
        @test isfile(joinpath(output_dir, "checkpoint.jls"))
        saved = TOML.parsefile(joinpath(output_dir, "config.toml"))
        @test saved["dataset"] == "SA"
        @test saved["classes"] == collect(1:16)
        @test length(readlines(joinpath(output_dir, "class_accuracy.tsv"))) == 17
        @test length(readlines(joinpath(output_dir, "confusion.tsv"))) == 17
    end
end
