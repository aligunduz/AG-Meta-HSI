module AGMetaHSI

using MAT
using Random
using Lux
using NNlib
using Statistics
using Optimisers
using Zygote
using Serialization
using SHA
using TOML
using Dates

export HSIScene, PixelSplit, load_pavia_university, make_pixel_split,
       save_pixel_split, load_pixel_split, validate_pixel_split, extract_patch

"""One hyperspectral cube (height, width, bands) and its center-pixel labels."""
struct HSIScene
    cube::Array{Float32,3}
    labels::Matrix{Int}
    function HSIScene(cube::Array{Float32,3}, labels::Matrix{Int})
        size(cube)[1:2] == size(labels) ||
            throw(ArgumentError("Cube and ground-truth spatial dimensions differ"))
        all(x -> x >= 0, labels) || throw(ArgumentError("Labels must be nonnegative"))
        new(cube, labels)
    end
end

"""Coordinates are (row, column, original class ID); zero labels are background."""
struct PixelSplit
    seed::Int
    k::Int
    train::Vector{NTuple{3,Int}}
    test::Vector{NTuple{3,Int}}
end

function load_pavia_university(data_dir::AbstractString)
    cube_file = joinpath(data_dir, "PaviaU.mat")
    gt_file = joinpath(data_dir, "PaviaU_gt.mat")
    isfile(cube_file) || throw(ArgumentError("Missing $cube_file"))
    isfile(gt_file) || throw(ArgumentError("Missing $gt_file"))
    cube_data = matread(cube_file)
    label_data = matread(gt_file)
    haskey(cube_data, "paviaU") || throw(ArgumentError("paviaU variable missing in $cube_file"))
    haskey(label_data, "paviaU_gt") || throw(ArgumentError("paviaU_gt variable missing in $gt_file"))
    cube = cube_data["paviaU"]
    labels = label_data["paviaU_gt"]
    ndims(cube) == 3 || throw(ArgumentError("Expected height × width × bands cube"))
    ndims(labels) == 2 || throw(ArgumentError("Expected 2-D ground truth"))
    return HSIScene(Float32.(cube), Int.(labels))
end

function make_pixel_split(labels::Matrix{Int}; k::Int=5, seed::Int=93)
    k > 0 || throw(ArgumentError("k must be positive"))
    classes = sort([c for c in unique(labels) if c != 0])
    isempty(classes) && throw(ArgumentError("Ground truth has no labeled pixels"))
    rng = MersenneTwister(seed)
    train = NTuple{3,Int}[]
    test = NTuple{3,Int}[]
    for class_id in classes
        positions = findall(==(class_id), labels)
        length(positions) > k ||
            throw(ArgumentError("Class $class_id needs more than $k pixels to leave a test set"))
        order = randperm(rng, length(positions))
        for (i, j) in enumerate(order)
            pos = positions[j]
            record = (pos[1], pos[2], class_id)
            push!(i <= k ? train : test, record)
        end
    end
    sort!(train; by=x -> (x[3], x[1], x[2]))
    sort!(test; by=x -> (x[3], x[1], x[2]))
    split = PixelSplit(seed, k, train, test)
    validate_pixel_split(labels, split)
    return split
end

function validate_pixel_split(labels::Matrix{Int}, split::PixelSplit)
    split.k > 0 || throw(ArgumentError("k must be positive"))
    seen = Set{Tuple{Int,Int}}()
    counts = Dict{Int,Int}()
    for (group, records) in ((:train, split.train), (:test, split.test))
        for (row, col, class_id) in records
            checkbounds(Bool, labels, row, col) ||
                throw(ArgumentError("$group coordinate outside image: ($row, $col)"))
            labels[row, col] == class_id && class_id != 0 ||
                throw(ArgumentError("$group label mismatch at ($row, $col)"))
            (row, col) in seen && throw(ArgumentError("Duplicate coordinate ($row, $col)"))
            push!(seen, (row, col))
            if group == :train
                counts[class_id] = get(counts, class_id, 0) + 1
            end
        end
    end
    classes = Set([c for c in unique(labels) if c != 0])
    Set(keys(counts)) == classes || throw(ArgumentError("Train set does not cover all classes"))
    all(c -> counts[c] == split.k, classes) ||
        throw(ArgumentError("Train set must contain exactly k pixels for each class"))
    length(seen) == count(x -> x != 0, labels) ||
        throw(ArgumentError("All labeled pixels must appear exactly once in train or test"))
    return true
end

"""Write one portable TSV with the seed and the exact chosen pixel coordinates."""
function save_pixel_split(path::AbstractString, split::PixelSplit)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# seed=$(split.seed) k=$(split.k)")
        println(io, "set\tclass\trow\tcol")
        for (group, records) in (("train", split.train), ("test", split.test))
            for (row, col, class_id) in records
                println(io, "$group\t$class_id\t$row\t$col")
            end
        end
    end
    return path
end

function load_pixel_split(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 2 || throw(ArgumentError("Incomplete split file"))
    header = match(r"^# seed=(-?\d+) k=(\d+)$", lines[1])
    header === nothing && throw(ArgumentError("Invalid split metadata"))
    lines[2] == "set\tclass\trow\tcol" || throw(ArgumentError("Invalid TSV columns"))
    train = NTuple{3,Int}[]
    test = NTuple{3,Int}[]
    for line in lines[3:end]
        fields = split(line, '\t')
        length(fields) == 4 || throw(ArgumentError("Invalid TSV row: $line"))
        group, class_str, row_str, col_str = fields
        record = (parse(Int, row_str), parse(Int, col_str), parse(Int, class_str))
        if group == "train"
            push!(train, record)
        elseif group == "test"
            push!(test, record)
        else
            throw(ArgumentError("Unknown split group: $group"))
        end
    end
    return PixelSplit(parse(Int, header.captures[1]), parse(Int, header.captures[2]),
                      train, test)
end

"""A zero-padded (patch_size, patch_size, bands) patch at a 1-based pixel."""
function extract_patch(scene::HSIScene, row::Int, col::Int; patch_size::Int=9)
    patch_size > 0 && isodd(patch_size) ||
        throw(ArgumentError("patch_size must be positive and odd"))
    checkbounds(Bool, scene.labels, row, col) ||
        throw(BoundsError(scene.labels, (row, col)))
    radius = patch_size ÷ 2
    height, width, bands = size(scene.cube)
    r1, r2 = max(1, row - radius), min(height, row + radius)
    c1, c2 = max(1, col - radius), min(width, col + radius)
    patch = zeros(Float32, patch_size, patch_size, bands)
    pr1, pc1 = r1 - row + radius + 1, c1 - col + radius + 1
    @views patch[pr1:pr1+(r2-r1), pc1:pc1+(c2-c1), :] .=
        scene.cube[r1:r2, c1:c2, :]
    return patch
end

include("ssarn.jl")
include("supervised.jl")

end # module
