# AG-Meta-HSI

Julia experiments for few-shot hyperspectral image classification. The first
milestone implements the **single-scene Pavia University data protocol** used
in the QMTN/QLOML paper. No classifier or QLOML optimizer is included yet.

## Setup

Install Julia 1.10 or newer. From the repository root:

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Download **Pavia University** and its ground truth from the
[UPV/EHU hyperspectral scenes page](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).
Place `PaviaU.mat` and `PaviaU_gt.mat` in `data/`. The loader expects the
MATLAB variables `paviaU` and `paviaU_gt`.

```julia
julia --project=. scripts/prepare_up.jl data outputs/up_split_seed93.tsv 93
```

The script records the seed and every chosen **1-based center-pixel
coordinate**. For each of UP's nine labeled classes it chooses five real
pixels for training; all other labeled pixels belong to the test set.
Background pixels (class 0) are excluded. The split is reused by loading the
TSV, and `validate_pixel_split` checks it against the original ground truth.

`extract_patch(scene, row, col)` returns a zero-padded 9 × 9 × B patch;
its label is the ground-truth class of the center pixel. The extractor
can be called on demand so the full test set need not be materialized.

## Research protocol

The QMTN paper uses five labeled center pixels per class, full spectral
bands, 9 × 9 patches, 5-way UP training tasks, and scene-wide testing. The
data code here fixes only the pixel selection and patch geometry. A model
implementation must explicitly resolve how its episode-specific 5-way head
produces predictions for all nine UP classes during evaluation. The paper's
twin-network update also needs an explicit parameter-flow specification
before implementing QLOML. No test-set labels should be used in optimization
or model selection.

**Preprocessing assumption:** patches currently contain the raw values from
the corrected scene. No normalization, spectral band selection, or spatial
buffer between train and test pixels is applied. These choices must be
recorded and held fixed before comparing models. Pixel-center disjointness
does not imply disjoint 9 × 9 neighborhoods.

## Files

- `src/AGMetaHSI.jl`: scene reader, split creation and validation, patch extraction.
- `scripts/prepare_up.jl`: create a reproducible UP split.
- `test/runtests.jl`: synthetic checks for split membership and patch borders.

The original GitHub repository was empty at the time of this starter package.
This package can be copied into its local clone and committed there.
