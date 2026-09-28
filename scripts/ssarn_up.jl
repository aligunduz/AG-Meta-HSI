using AGMetaHSI

function parse_ssarn_args(args)
    modes = filter(arg -> arg in ("--smoke", "--check", "--train"), args)
    length(unique(modes)) <= 1 || error("Choose only one mode: --smoke, --check, or --train")
    if "--help" in args
        println("Usage: julia --project=. scripts/ssarn_up.jl [--check | --smoke | --train] [--data DIR] [--split TSV] [--output DIR] [--seed 93] [--epochs 300] [--lr 0.001] [--batch-size 15] [--test-batch-size 32]")
        println("Default: --check (forward pass only; no training or test evaluation).")
        return
    end
    options = Dict{Symbol,Any}(:train => false, :smoke => false)
    names = Dict("--data" => :data_dir, "--split" => :split_path, "--output" => :output_dir,
        "--seed" => :seed, "--epochs" => :epochs, "--lr" => :learning_rate,
        "--batch-size" => :batch_size, "--test-batch-size" => :test_batch_size)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--train", "--check", "--smoke")
            options[:train] = arg == "--train"
            options[:smoke] = arg == "--smoke"
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
    return options
end

function main(args)
    options = parse_ssarn_args(args)
    options === nothing && return
    return run_supervised(; options...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
