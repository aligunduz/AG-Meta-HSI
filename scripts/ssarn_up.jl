using AGMetaHSI

function main(args)
    if "--help" in args
        println("Usage: julia --project=. scripts/ssarn_up.jl [--check | --train] [--data DIR] [--split TSV] [--output DIR] [--seed 93] [--epochs 300] [--lr 0.001] [--batch-size 15] [--test-batch-size 32]")
        println("Default: --check (forward pass only; no training or test evaluation).")
        return
    end
    options = Dict{Symbol,Any}(:train => false)
    names = Dict("--data" => :data_dir, "--split" => :split_path, "--output" => :output_dir,
        "--seed" => :seed, "--epochs" => :epochs, "--lr" => :learning_rate,
        "--batch-size" => :batch_size, "--test-batch-size" => :test_batch_size)
    "--train" in args && "--check" in args && error("Choose --train or --check")
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--train", "--check")
            options[:train] = arg == "--train"
        else
            haskey(names, arg) || error("Unknown argument: $arg")
            i < length(args) || error("Missing value for $arg")
            key = names[arg]
            i += 1
            options[key] = key in (:seed, :epochs, :batch_size, :test_batch_size) ?
                parse(Int, args[i]) : key == :learning_rate ? parse(Float64, args[i]) : args[i]
        end
        i += 1
    end
    run_supervised(; options...)
end

main(ARGS)
