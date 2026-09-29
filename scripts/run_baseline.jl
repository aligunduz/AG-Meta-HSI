using AGMetaHSI

function parse_baseline_args(args)
    if "--help" in args
        println("Usage: julia --project=. scripts/run_baseline.jl --baseline SSARN --dataset UP|SA|IP --train [--data DIR] [--split TSV] [--output DIR] [--seed 93] [--k 5] [--epochs 300] [--lr 0.001] [--batch-size 15] [--test-batch-size 32]")
        return nothing
    end
    options = Dict{String,String}()
    train = false
    i = 1
    valid = Set(["--baseline", "--dataset", "--data", "--split", "--output",
                 "--seed", "--k", "--epochs", "--lr", "--batch-size", "--test-batch-size"])
    while i <= length(args)
        flag = args[i]
        if flag == "--train"
            train = true
        else
            flag in valid || error("Unknown argument: $flag")
            i < length(args) || error("Missing value for $flag")
            haskey(options, flag) && error("Duplicate argument: $flag")
            i += 1
            options[flag] = args[i]
        end
        i += 1
    end
    train || error("Training requires --train")
    baseline = uppercase(get(options, "--baseline", "SSARN"))
    baseline == "SSARN" || error("Unsupported baseline $baseline; available: SSARN")
    dataset = uppercase(get(options, "--dataset", "UP"))
    dataset_spec(dataset)
    seed = parse(Int, get(options, "--seed", "93"))
    k = parse(Int, get(options, "--k", "5"))
    epochs = parse(Int, get(options, "--epochs", "300"))
    lr = parse(Float64, get(options, "--lr", "0.001"))
    batch_size = parse(Int, get(options, "--batch-size", "15"))
    test_batch_size = parse(Int, get(options, "--test-batch-size", "32"))
    data_dir = get(options, "--data", "data")
    default_split = joinpath("outputs", "$(lowercase(dataset))_split_seed$(seed)_k$(k).tsv")
    split_path = get(options, "--split", default_split)
    output_dir = get(options, "--output", joinpath("outputs", "$(lowercase(baseline))_$(lowercase(dataset))_seed$(seed)"))
    return (; baseline, dataset, seed, k, epochs, lr, batch_size,
            test_batch_size, data_dir, split_path, output_dir,
            split_given=haskey(options, "--split"))
end

function main(args)
    options = parse_baseline_args(args)
    options === nothing && return
    if !isfile(options.split_path)
        options.split_given && error("Explicit split file does not exist: $(options.split_path)")
        scene = load_hsi_dataset(options.dataset, options.data_dir)
        split = make_pixel_split(scene.labels; k=options.k, seed=options.seed)
        save_pixel_split(options.split_path, split)
        println("Created split: $(options.split_path)")
    end
    return run_supervised(; dataset=options.dataset, data_dir=options.data_dir,
        split_path=options.split_path, output_dir=options.output_dir,
        seed=options.seed, k=options.k, epochs=options.epochs,
        learning_rate=options.lr, batch_size=options.batch_size,
        test_batch_size=options.test_batch_size, train=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
