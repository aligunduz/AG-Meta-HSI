using Test, SHA
include(joinpath(@__DIR__, "..", "scripts", "ssarn_up.jl"))

@testset "Mode selection and smoke guards" begin
    @test parse_ssarn_args(String[]) == parse_ssarn_args(["--check"])
    @test parse_ssarn_args(["--train"])[:train]
    @test !parse_ssarn_args(["--train"])[:smoke]
    @test parse_ssarn_args(["--smoke"])[:smoke]
    @test !parse_ssarn_args(["--smoke"])[:train]
    for args in (["--smoke", "--check"], ["--smoke", "--train"],
                 ["--check", "--train"], ["--smoke", "--check", "--train"])
        @test_throws r"Choose only one mode" parse_ssarn_args(args)
    end
    @test_throws r"batch_size=15" run_supervised(; smoke=true, batch_size=14)
    @test_throws r"Choose only one mode" run_supervised(; smoke=true, train=true)
    @test !AGMetaHSI.finite_leaves((w=[1f0, Inf32],))
    @test AGMetaHSI.changed_parameters((w=[1f0, 2f0],), (w=[1f0, 3f0],)) == 1
end

# Explicit real-data integration test; not part of the portable synthetic suite.
@testset "Real UP smoke CLI: one update and no output files" begin
    root = dirname(@__DIR__)
    snapshot() = Dict(relpath(joinpath(dir, f), root) =>
        bytes2hex(open(sha256, joinpath(dir, f)))
        for (dir, _, files) in walkdir(joinpath(root, "outputs")) for f in files)
    directories() = sort([dir for (dir, _, _) in walkdir(joinpath(root, "outputs"))])
    before, dirs_before = snapshot(), directories()
    command = `$(Base.julia_cmd()) --project=$root $(joinpath(root, "scripts", "ssarn_up.jl")) --smoke`
    output = read(Cmd(command; dir=root), String)
    print(output)
    @test occursin("batch_size=15; Adam updates=1", output)
    @test occursin(r"Loss before=.*finite=true", output)
    @test occursin(r"Loss after=.*finite=true", output)
    @test occursin("Gradients finite=true", output)
    @test occursin(r"parameters changed=[1-9][0-9]*", output)
    @test snapshot() == before
    @test directories() == dirs_before
end
