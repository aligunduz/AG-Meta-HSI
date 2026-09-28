export ssarn, patch_batch, check_ssarn

# Figure 3 and Eq. (7): full-width Q/K/V, no transformer scaling.
struct SpatialAttention{Q,K,V} <: Lux.AbstractLuxContainerLayer{(:query, :key, :value)}
    query::Q
    key::K
    value::V
end
SpatialAttention(c::Int) = SpatialAttention(
    Lux.Conv((1, 1), c => c), Lux.Conv((1, 1), c => c), Lux.Conv((1, 1), c => c))

function (m::SpatialAttention)(x, ps, st)
    q, sq = m.query(x, ps.query, st.query)
    k, sk = m.key(x, ps.key, st.key)
    v, sv = m.value(x, ps.value, st.value)
    h, w, c, n = size(x)
    # Lux layout H,W,C,N -> C,HW,N. Softmax over columns of Q'K.
    q = permutedims(reshape(q, h*w, c, n), (2, 1, 3))
    k = permutedims(reshape(k, h*w, c, n), (2, 1, 3))
    v = permutedims(reshape(v, h*w, c, n), (2, 1, 3))
    weights = NNlib.softmax(NNlib.batched_mul(permutedims(q, (2, 1, 3)), k); dims=2)
    y = NNlib.batched_mul(v, weights)
    return reshape(permutedims(y, (2, 1, 3)), h, w, c, n),
           (query=sq, key=sk, value=sv)
end

leaky(x) = NNlib.leakyrelu.(x, 0.01f0)
bn(c) = Lux.BatchNorm(c; epsilon=1.0f-5, momentum=0.1f0)

"""Figure-2-based SSARN; H,W,B,1,N input and nine raw logits per sample.
Unspecified paper details are fixed explicitly in README.md.
"""
function ssarn()
    stem(cin) = Lux.Chain(Lux.Conv((1, 1, 7), cin => 32; stride=(1, 1, 2)),
                         bn(32), Lux.WrappedFunction(leaky))
    spectral() = Lux.Chain(Lux.Conv((1, 1, 7), 32 => 32; pad=(0, 0, 3)),
                          Lux.WrappedFunction(leaky), bn(32))
    # Eq. (5): x + F2(x + F1(x)), not two sequential ordinary residual blocks.
    sr = Lux.SkipConnection(Lux.Chain(Lux.SkipConnection(spectral(), +), spectral()), +)
    spatial = Lux.Chain(Lux.Conv((3, 3), 32 => 32; pad=1),
                        Lux.WrappedFunction(leaky), bn(32))
    # Eq. (8): x + F(x) + Att(F(x)).
    sar = Lux.SkipConnection(Lux.Chain(spatial, Lux.SkipConnection(SpatialAttention(32), +)), +)
    return Lux.Chain(
        stem=Lux.Chain(stem(1), stem(32), stem(32)),
        spectral_residual=sr,
        collapse=Lux.Chain(Lux.WrappedFunction(leaky),
            Lux.Conv((3, 3, 8), 32 => 32), Lux.WrappedFunction(leaky), bn(32),
            Lux.WrappedFunction(x -> dropdims(x; dims=3))),
        spatial_residual=sar,
        pool=Lux.Chain(Lux.WrappedFunction(leaky),
            Lux.WrappedFunction(x -> reshape(mean(x; dims=(1, 2)), 32, size(x, 4)))),
        classifier=Lux.Dense(32 => 9))
end

function patch_batch(scene::HSIScene, records)
    size(scene.cube, 3) == 103 || throw(ArgumentError("SSARN expects 103 UP bands"))
    x = Array{Float32}(undef, 9, 9, 103, 1, length(records))
    for (i, (row, col, _)) in enumerate(records)
        x[:, :, :, 1, i] = extract_patch(scene, row, col; patch_size=9)
    end
    return x
end

function check_ssarn(model, ps, st, x)
    size(x) == (9, 9, 103, 1, 1) || throw(ArgumentError("Expected one 9x9x103 patch"))
    logits, _ = model(x, ps, Lux.testmode(deepcopy(st)))
    size(logits) == (9, 1) || error("Expected nine logits, got $(size(logits))")
    all(isfinite, logits) || error("Nonfinite logits")
    return logits
end
