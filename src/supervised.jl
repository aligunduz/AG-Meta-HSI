export classification_metrics, run_supervised, supervised_loss, validate_up_protocol

function validate_up_protocol(scene, split)
    size(scene.cube, 3) == 103 || error("UP must have 103 bands")
    split.seed == 93 && split.k == 5 || error("Expected fixed seed93/k5 split")
    sort(unique(filter(!iszero, vec(scene.labels)))) == collect(1:9) ||
        error("Expected class IDs 1:9")
    validate_pixel_split(scene.labels, split)
    return true
end

"""Rows = truth, columns = prediction; accuracies in [0,1], kappa unscaled."""
function classification_metrics(cm::AbstractMatrix{<:Integer})
    size(cm, 1) == size(cm, 2) || throw(ArgumentError("Confusion matrix must be square"))
    all(>=(0), cm) || throw(ArgumentError("Counts must be nonnegative"))
    support = vec(sum(cm; dims=2))
    all(>(0), support) || throw(ArgumentError("Every class needs test support"))
    n = sum(cm)
    correct = [cm[i, i] for i in axes(cm, 1)]
    per_class = correct ./ support
    oa = sum(correct) / n
    pe = sum((support ./ n) .* (vec(sum(cm; dims=1)) ./ n))
    kappa = pe == 1 ? NaN : (oa - pe) / (1 - pe)
    return (; OA=oa, AA=mean(per_class), per_class, kappa, support)
end

function supervised_loss(model, ps, st, x, y)
    logits, newst = model(x, ps, st)
    loss = -sum(y .* NNlib.logsoftmax(logits; dims=1)) / size(x, 5)
    return loss, newst
end

function evaluate_ssarn(model, ps, st, scene, records; batch_size=32)
    cm = zeros(Int, 9, 9)
    st = Lux.testmode(st)
    for first in 1:batch_size:length(records)
        batch = records[first:min(first + batch_size - 1, length(records))]
        logits, _ = model(patch_batch(scene, batch), ps, st)
        all(isfinite, logits) || error("Nonfinite test logits")
        for (j, (_, _, truth)) in enumerate(batch)
            cm[truth, argmax(view(logits, :, j))] += 1
        end
    end
    return classification_metrics(cm), cm
end

filehash(path) = open(io -> bytes2hex(sha256(io)), path)
write_toml(path, data) = open(io -> TOML.print(io, data), path, "w")

"""Check-only by default. Training requires an explicit train=true.
Fixed final epoch selection; test labels enter metrics only after checkpoint save.
"""
function run_supervised(; data_dir="data", split_path="outputs/up_split_seed93.tsv",
        output_dir="outputs/ssarn_seed93", seed=93, epochs=300, learning_rate=0.001,
        batch_size=15, test_batch_size=32, train=false)
    epochs > 0 || error("epochs must be positive")
    isfinite(learning_rate) && learning_rate > 0 || error("learning_rate must be positive and finite")
    batch_size >= 2 || error("batch_size must be at least 2")
    test_batch_size > 0 || error("test_batch_size must be positive")
    scene = load_pavia_university(data_dir)
    split = load_pixel_split(split_path) # Never regenerate or rewrite the split.
    validate_up_protocol(scene, split)
    split_digest = filehash(split_path)
    rng = MersenneTwister(seed)
    model = ssarn()
    ps, st = Lux.setup(rng, model)
    xcheck = patch_batch(scene, split.train[1:1])
    logits = check_ssarn(model, ps, st, xcheck)
    println("Forward check: $(size(xcheck)) -> $(size(logits)); finite logits")
    println("Fixed split: $(length(split.train)) train, $(length(split.test)) test; SHA256=$split_digest")
    train || return (; logits, split_sha256=split_digest)

    # Refuse to overwrite a previous or partially completed experiment.
    ispath(output_dir) && error("Output directory already exists: $output_dir")
    mkpath(output_dir)
    root = dirname(@__DIR__)
    config = Dict("model" => "SSARN-Fig2-assumptions-v1", "seed" => seed,
        "split_seed" => split.seed, "split_sha256" => split_digest,
        "epochs" => epochs, "learning_rate" => learning_rate, "optimizer" => "Adam",
        "batch_size" => batch_size, "test_batch_size" => test_batch_size,
        "train_count" => length(split.train), "test_count" => length(split.test),
        "patch_size" => [9, 9, 103], "classes" => collect(1:9),
        "preprocessing" => "raw Float32; zero padding; no augmentation",
        "selection" => "fixed final epoch; test once after checkpoint",
        "julia_version" => string(VERSION), "started_at" => string(now()),
        "device" => "CPU", "checkpoint" => "checkpoint.jls")
    config["data_sha256"] = Dict(f => filehash(joinpath(data_dir, f)) for f in ("PaviaU.mat", "PaviaU_gt.mat"))
    config["source_sha256"] = Dict(f => filehash(joinpath(@__DIR__, f)) for f in ("AGMetaHSI.jl", "ssarn.jl", "supervised.jl"))
    write_toml(joinpath(output_dir, "config.toml"), config)
    cp(split_path, joinpath(output_dir, "split.tsv"))
    for f in ("Project.toml", "Manifest.toml")
        isfile(joinpath(root, f)) && cp(joinpath(root, f), joinpath(output_dir, f))
    end
    xtrain = patch_batch(scene, split.train)
    ytrain = Float32.([c == record[3] for c in 1:9, record in split.train])
    opt = Optimisers.setup(Optimisers.Adam(Float32(learning_rate)), ps)
    st = Lux.trainmode(st)
    open(joinpath(output_dir, "training.tsv"), "w") do io
        println(io, "epoch\ttrain_cross_entropy")
        for epoch in 1:epochs
            order = randperm(rng, length(split.train))
            total = 0.0
            for first in 1:batch_size:length(order)
                ids = order[first:min(first + batch_size - 1, length(order))]
                x, y = xtrain[:, :, :, :, ids], ytrain[:, ids]
                (loss, nextst), back = Zygote.pullback(p -> supervised_loss(model, p, st, x, y), ps)
                isfinite(loss) || error("Nonfinite training loss at epoch $epoch")
                grads = back((one(loss), nothing))[1]
                opt, ps = Optimisers.update(opt, ps, grads)
                st = nextst
                total += Float64(loss) * length(ids)
            end
            println(io, "$epoch\t$(total / length(order))")
            flush(io)
            println("Epoch $epoch/$epochs; train CE=$(total / length(order))")
        end
    end
    filehash(split_path) == split_digest || error("Split changed during training")
    # Save final parameters AND BatchNorm state before reading test predictions.
    serialize(joinpath(output_dir, "checkpoint.jls"),
        (; ps, st, optimizer_state=opt, rng, epoch=epochs, config))
    metrics, cm = evaluate_ssarn(model, ps, st, scene, split.test; batch_size=test_batch_size)
    write_toml(joinpath(output_dir, "metrics.toml"), Dict(
        "OA" => metrics.OA, "AA" => metrics.AA, "kappa" => metrics.kappa,
        "accuracy_units" => "fraction", "epoch" => epochs, "seed" => seed,
        "learning_rate" => learning_rate, "checkpoint" => "checkpoint.jls",
        "split_sha256" => split_digest, "finished_at" => string(now())))
    open(joinpath(output_dir, "class_accuracy.tsv"), "w") do io
        println(io, "class\tsupport\tcorrect\taccuracy")
        for c in 1:9
            println(io, "$c\t$(metrics.support[c])\t$(cm[c,c])\t$(metrics.per_class[c])")
        end
    end
    open(joinpath(output_dir, "confusion.tsv"), "w") do io
        println(io, "truth/prediction\t", join(1:9, '\t'))
        for c in 1:9
            println(io, c, '\t', join(cm[c, :], '\t'))
        end
    end
    println("Saved final checkpoint and test metrics to $output_dir")
    return metrics
end
