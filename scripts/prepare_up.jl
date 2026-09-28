using AGMetaHSI

function main(args)
    length(args) in (2, 3) || error(
        "Usage: julia --project=. scripts/prepare_up.jl DATA_DIR OUTPUT_TSV [SEED]")
    data_dir, output_path = args[1:2]
    seed = length(args) == 3 ? parse(Int, args[3]) : 93
    scene = load_pavia_university(data_dir)
    split = make_pixel_split(scene.labels; k=5, seed=seed)
    save_pixel_split(output_path, split)
    println("Scene: $(size(scene.cube)); seed: $seed")
    println("Train: $(length(split.train)); test: $(length(split.test))")
    println("Saved split: $output_path")
end

main(ARGS)
