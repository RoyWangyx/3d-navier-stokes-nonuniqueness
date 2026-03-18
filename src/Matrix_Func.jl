"""
Compute the weights for composite Newton–Cotes's rule.
# Input:
- n: number of intervals (must be a multiple of 6)
- h: step size
# Return:
- A vector of length n+1, where the k-th entry is the weight for the k-th point.
"""
function newton_cotes_weights(n::Int, h::dtype)
    @assert n % 6 == 0 "n must be a multiple of 6 for Newton–Cotes's rule"
    w = zeros(dtype, n + 1)
    c = h / dt(140)
    # endpoints
    w[1]   = dt(41) * c
    w[end] = dt(41) * c
    for i in 2:n
        r = (i - 1) % 6
        if r == 1 || r == 5
            w[i] = dt(216) * c
        elseif r == 2 || r == 4
            w[i] = dt(27) * c
        elseif r == 3
            w[i] = dt(272) * c
        else
            # r == 0 : interior panel boundary, shared by two panels
            w[i] = dt(82) * c   # 41+41 from adjacent panels
        end
    end
    return w
end

"""
Compute a 1D numerical integral on [0, 1] using composite Newton–Cotes's rule,
and return an a priori error bound based on the 8th derivative.

# Input:
- f: vector of function values sampled on a uniform grid over [0, 1]
- Wk∞_bound: upper bound of the infinity norm of the 8th derivative of f

# Return:
- S: numerical approximation of the integral ∫₀¹ f(x) dx
- err_bd: theoretical upper bound of the quadrature error
"""
function newton_cotes_with_bound(f::AbstractVector{dtype}, Wk∞_bound::dtype)
    N = length(f) - 1
    h = inv(dt(N))

    w = newton_cotes_weights(N, h)
    S = sum(w .* f)

    err_bd = Wk∞_bound * h^8 * dt(3) / dt(2800)

    return S, err_bd
end

"""
Finite-difference derivative operator (centered stencil).

Apply a k-th order finite-difference operator along the first dimension of `Xpad`,
assuming data are sampled on a uniform grid with spacing `h`. The input is expected
to be padded by `k` points on both ends, so that the output has length `N = NX - 2k`
in the first dimension.

# Input
- `Xpad::AbstractArray{dtype}`: padded samples (size `NX × ...`, with `NX ≥ 2k+1`)
- `h::dtype`: grid spacing
- `k::Int`: derivative order (k ≥ 0)

# Return
- If `k == 0`: return `Xpad` unchanged
- Otherwise: an array `Y` of size `(NX - 2k, ...)` approximating the k-th derivative
  on the interior grid points using a centered (k+1)-point stencil.
"""
function finite_difference(Xpad::AbstractArray{dtype}, h::dtype, k::Int)
    NX = size(Xpad, 1)
    N = NX - 2 * k
    if k == 0
        return Xpad
    end

    # centered stencil
    shift = iseven(k) ? -(k ÷ 2) : -((k - 1) ÷ 2)

    # precompute offsets and coefficients (length k+1)
    offs  = Vector{Int}(undef, k+1)
    coeff = Vector{dtype}(undef, k+1)

    inv_scale = inv(h^dt(k))
    @inbounds for jj in 0:k
        offs[jj+1] = shift + jj
        c = ((-1)^(k - jj)) * binomial(k, jj)
        coeff[jj+1] = dt(c) * inv_scale
    end

    # output
    outsz = (N, tail(size(Xpad))...)
    Y = Array{dtype}(undef, outsz)

    X2 = reshape(Xpad, NX, :)
    Y2 = reshape(Y,     N, :)

    @inbounds @threads for col in axes(X2, 2)
        for i in 1:N
            ip = i + k
            s = zero(dtype)
            for t in eachindex(offs)
                s += coeff[t] * X2[ip + offs[t], col]
            end
            Y2[i, col] = s
        end
    end

    return Y
end


"""
Suppose `f(β, θ)` is vector-valued, with `l` components. This function
builds the auxiliary parameter array `paras` used by `norm_lst` when
estimating one-variable derivative chains for each component.

The return value is a matrix `paras` of size `(l, 2)`:
- `i = 1, ..., l` indexes the i-th component of the vector-valued function;
- `j = 1, 2` indexes the two one-variable derivative chains:
  - `j = 1`: the β-derivative chain
        f_i, ∂β f_i, ∂β² f_i, ..., ∂β^k f_i
  - `j = 2`: the θ-derivative chain
        f_i, ∂θ f_i, ∂θ² f_i, ..., ∂θ^k f_i
Thus `paras[i, j]` is the parameter package for estimating the derivatives
of the i-th component with respect to the j-th variable. Its internal format 
follows the convention used by `get_Wk∞_norm_lst`; see the docstring of 
`get_Wk∞_norm_lst` for more details.

# Ordinary mode (`ind === nothing`)
In the ordinary mode, `f(β, θ)` is a trigonometric polynomial in both β and θ, 
so can be closed by the same frequency-based mechanism. For each component `i`,
the package for each chain is constructed from:
- a pair of high-order bounds used to start the backward recursion;
- a frequency from the other direction.

# Special mode (`ind !== nothing`)
In the special mode, the specified column `ind` is used. The β-chain
(`j = 1`) is constructed in the same way as in the ordinary mode. For 
θ-chain (`j = 2`), the first entry of `paras[i, 2]` is also the same.
The second entry of `paras[i, 2]` is treated differently, because the 
β-direction is no longer trigonometric. It is a per-order list of explicit 
auxiliary `Wk∞` bounds. More precisely, it provides upper bounds for
    ∂β² f_i, ∂β²∂θ f_i, ..., ∂β²∂θ^(k-2) f_i,
which are used in place of β-frequency closure when estimating the
θ-derivative chain. These bounds enter the second auxiliary slot of
`estimate_Wk∞_norm`, namely the term associated with the `h1^2 / 8`
correction.

# Input
- `Wk∞_norms_lst`: precomputed auxiliary Wk∞ bound tables.
- `k`: target maximum derivative order.
- `freq_lst`: length-2 vector of wavenumbers, ordered as [β, θ].
- `ind`: optional column index. If `nothing`, ordinary mode is used;
  otherwise the specified column is used.

# Return
- A matrix `paras` of size `(l, 2)`.
"""
function Linf_to_paras(Wk∞_norms_lst, k::Int, freq_lst::Vector{dtype}; ind::Union{Nothing, Int}=nothing)
    if ind !== nothing
        sp_flag = true
    else
        sp_flag = false
        ind = 1
    end
    l = size(Wk∞_norms_lst[1, 1], 1)
    paras = [[] for _ in 1:l, _ in 1:2]
    for i in 1:l, j in 1:2
        if j == 2 && sp_flag
            push!(paras[i, j], ["Wk∞", Wk∞_norms_lst[1][i, ind]*freq_lst[2].^dt.(k-1:k)])
            push!(paras[i, j], ["Wk∞", Wk∞_norms_lst[3][i, ind]*freq_lst[2].^dt.(0:k-2)])
        else
            push!(paras[i, j], ["Wk∞", [Wk∞_norms_lst[k][i, ind], Wk∞_norms_lst[k+1][i, ind]]])
            push!(paras[i, j], ["freq", freq_lst[3-j]])
        end
    end
    return paras
end

"""
Compute L∞ bounds for all derivatives from order 0 up to order k.
This function returns [‖f‖∞, ‖f'‖∞, ..., ‖f^(k)‖∞]
for one fixed single-variable derivative chain, by calling
`estimate_Wk∞_norm` recursively from high order down to low order.

# Input
- `f`: sampled data array.
- `h`: length-2 vector of step sizes `[h0, h1]`.
- `k`: maximum derivative order.
- `paras`: one complete parameter package, typically produced by `Linf_to_paras`; 
  it is a length-2 vector: paras = [p1, p2].

# Meaning of every entry 'p':
- `nothing`: no information is used.
- `["freq", M]`: the same frequency bound `M` is used in the auxiliary slot
  for every derivative order.
- `["Wk∞", [A, B]]`: only for p1. Recursive mode is activated. The length-2 list
  `[A, B]` provides the two highest-order seeds used to initialize the backward
  recursion.
- `["Wk∞", Wlist]`: only for p2. `Wlist` is a per-order list of explicit auxiliary
  bounds. Its entry `Wlist[m]` is attached to the second auxiliary slot when
  estimating the derivative of order `m-1`.

# Internal recursion
If p1 = ["Wk∞", [A, B]], the bounds for the highest two derivatives are initialized 
as [A, B] in the internal result array. The current higher-order bound result[m+2] 
is then used recursively to estimate the lower-order bound result[m].

# Return
- A vector of length `k+1` containing the L∞ bounds for derivative orders
  `0, 1, ..., k`.
"""
function get_Wk∞_norm_lst(f::AbstractArray{dtype}, h::Vector{dtype}, k::Int; 
    paras::Vector=[nothing, nothing])

    result = zeros(dtype, k+3)
    paras_lst = [[] for _ in 1:k+1]
    rec_flag = false

    # build parameter list for each derivative order
    for i in 1:2
        if paras[i] === nothing
            for m in 1:k+1
                push!(paras_lst[m], nothing)
            end
        else
            @assert isa(paras[i], Vector) "paras[i] must be a vector when provided"
            if paras[i][1] == "freq"
                # same frequency bound for all orders
                M = paras[i][2]
                for m in 1:k+1
                    push!(paras_lst[m], ["freq", M])
                end
            elseif i == 2
                # explicit Wk∞ bounds provided for each order
                @assert paras[i][1] == "Wk∞" "paras[i][1] must be 'freq' or 'Wk∞' when provided"
                Wk∞_bound = paras[i][2]
                for m in 1:k+1
                    push!(paras_lst[m], ["Wk∞", Wk∞_bound[m]])
                end
            else
                # recursive mode: highest two bounds given
                @assert paras[i][1] == "Wk∞" "paras[i][1] must be 'freq' or 'Wk∞' when provided"
                rec_flag = true
                result[end-1: end] = paras[i][2]
            end
        end
    end

    # backward recursion from order k to 0
    for m in k+1:-1:1
        if rec_flag
            pushfirst!(paras_lst[m], ["Wk∞", result[m+2]])
        end
        result[m] = estimate_Wk∞_norm(f, h, m-1; paras=paras_lst[m])
    end

    return result[1:end-2]
end

"""
Estimate an L∞ upper bound for the k-th derivative from sampled data.

The function first computes a finite-difference approximation of the k-th
derivative and takes its maximum absolute value. It then closes this estimate
using up to two auxiliary correction terms supplied through `paras`.

# Input
- `f`: sampled data array. The first dimension is the grid direction.
- `h`: either a scalar step size `h0`, or a length-2 vector `[h0, h1]`.
- `k`: derivative order to estimate.
- `paras`: a length-2 vector `paras = [p1, p2]`,
  where each entry is either `nothing`, `["Wk∞", B]`, or `["freq", M]`.

# Meaning of `paras`
The two entries correspond to two correction layers:
- `p1` uses the effective spacing hk = h0 * (k + 1)
- `p2` uses the spacing `h1`

At the abstract level, the estimate is based on a bound of the form
    ‖∂^k f‖∞ ≤ max(abs(finite_difference(f, h0, k)))
              + ‖∂β²(∂^k f)‖∞ * hβ^2 / 8 + ‖∂θ²(∂^k f)‖∞ * hθ^2 / 8

The two auxiliary entries in `paras` are used to control the two remainder
terms in one of the following ways:
1. Explicit higher-derivative bound: `["Wk∞", B]`
    If a direct upper bound `B` is available for the corresponding remainder term,
    it is inserted additively:
        result += B * h^2 / 8
2. Frequency-based closure: `["freq", M]`
    If a frequency bound `M` is available instead, the corresponding remainder term
    is absorbed into the left-hand side through frequency closure, producing a
    factor (1 - (h M)^2 / 8) in the denominator.

# Return
- A scalar upper bound for the L∞ norm of the k-th derivative.

# Note
This function performs only a single-order estimate. It is used internally by
`get_Wk∞_norm_lst` to build the full list [‖f‖∞, ‖f'‖∞, ..., ‖f^(k)‖∞].
"""
function estimate_Wk∞_norm(f::AbstractArray{dtype}, h::Union{dtype, Vector{dtype}}, k::Int; 
    paras::Vector=[nothing, nothing])
    N = size(f, 1)
    @assert N ≥ 10 * (k + 1) "N too small for this stencil"
    if isa(h, Vector{dtype})
        h0, h1 = h
    else
        h0, h1 = h, zero(dtype)
    end
    Dk_f = finite_difference(f, h0, k)
    result = maximum(abs, Dk_f)

    hk = h0 * dt(k + 1)
    L∞_fac = one(dtype)

    if paras[1] !== nothing
        @assert isa(paras[1], Vector) "paras[1] must be a vector when provided"
        if paras[1][1] == "Wk∞"
            # use known bound on higher derivative to enlarge estimate (Taylor remainder control)
            Wk∞_bound = paras[1][2]
            result += Wk∞_bound * hk^2 / dt(8)
        else
            # assume band-limited signal with max frequency M0 to sharpen estimate
            @assert paras[1][1] == "freq" "paras[1][1] must be 'freq' or 'Wk∞' when provided"
            M0 = paras[1][2]
            L∞_fac -= (hk * M0)^2 / dt(8)
        end
    end

    if paras[2] !== nothing
        @assert isa(paras[2], Vector) "paras[2] must be a vector when provided"
        if paras[2][1] == "Wk∞"
            Wk∞_bound = paras[2][2]
            result += Wk∞_bound * h1^2 / dt(8)
        else
            @assert paras[2][1] == "freq" "paras[2][1] must be 'freq' or 'Wk∞' when provided"
            M1 = paras[2][2]
            L∞_fac -= (h1 * M1)^2 / dt(8)
        end
    end

    @assert L∞_fac > 0 "L∞_fac must be positive"
    return result /L∞_fac
end

"""
Plot a matrix as an image.
# Input:
- A: The matrix to plot.
- title_str: The title of the plot.
# No return value.
"""
function plot_matrix(A::AbstractMatrix{Float64}; title_str::String="")
    figure()
    imshow(A)
    colorbar()
    title(title_str)
    display(gcf())
end

"""
Add symmetric interval error to an interval matrix.

Return hull(mat - err_mat, mat + err_mat) elementwise.
# Input:
- mat: interval matrix
- err_mat: interval error matrix (same size as mat)
# Return:
- Interval matrix with enlarged bounds
"""
function add_error(mat::AbstractMatrix{Interval{Float64}}, err_mat::AbstractMatrix{Interval{Float64}})
    return hull.(mat .- err_mat, mat .+ err_mat)
end


"""
Save a dictionary to a .mat file, splitting interval arrays into lo/hi parts.

For each interval array X, saves X_lo = inf.(X) and X_hi = sup.(X).
# Input:
- path: output file path
- dict: dictionary to be saved
"""
function save_matrix(path::String, dict::Dict{String, <:Any})
    @assert endswith(path, ".mat") "File must be a .mat file"
    new_dict = Dict{String, Any}()
    for key in keys(dict)
        if isa(dict[key], AbstractArray{Interval{Float64}})
            # split interval array into lo/hi parts and save separately
            lo = inf.(dict[key])
            hi = sup.(dict[key])
            new_dict["$(key)_lo"] = lo
            new_dict["$(key)_hi"] = hi
        else
            new_dict[key] = dict[key]
        end
    end
    matwrite(path, new_dict)
end

"""
Construct an interval matrix from elementwise lower and upper bounds.

# Input:
- lo: array of lower bounds
- hi: array of upper bounds (same size as lo)
- check: if true, verify lo ≤ hi up to floating-point tolerance
# Return:
- An array of intervals with the same shape as lo/hi
"""
function interval_matrix(lo::AbstractArray, hi::AbstractArray; check=true)
    size(lo) == size(hi) || throw(DimensionMismatch("lo and hi size mismatch"))

    if check
        # allow floating-point tolerance: lo ≤ hi + eps(hi)
        bad = findall(lo .> hi .+ eps.(hi))
        isempty(bad) || error("Found lo > hi at $(first(bad)) (and $(length(bad))-1 more).")
    end

    # elementwise interval construction
    return interval.(lo, hi) 
end

"""
Read matrices from a .mat file, reconstructing interval arrays if needed.

If key exists, return its value; otherwise expect key_lo and key_hi and
reconstruct via interval_matrix.
# Input:
- path: .mat file path
- key_lst: list of variable names to read
# Return:
- Single object if one key is given, otherwise a list of objects
"""
function read_matrix(path, key_lst)
    @assert endswith(path, ".mat") "File must be a .mat file"
    d = matread(path)
    results = []
    for key in key_lst
        if key in keys(d)
            push!(results, dt.(d[key]))
        else
            # If key not found, expect key_lo and key_hi and reconstruct interval matrix
            @assert ("$(key)_lo" in keys(d)) && ("$(key)_hi" in keys(d)) "Key $key not found in file."
            lo, hi = d["$(key)_lo"], d["$(key)_hi"]
            push!(results, interval_matrix(lo, hi))
        end
    end
    if length(results) == 1
        return results[1]
    else
        return results
    end
end

"""
    verify_spd_by_ldlt(A)

Strictly verifies SPD for a *symmetric interval matrix* `A` using an interval LDLᵀ
decomposition (unit diagonal `L`). Sufficient condition: all pivots satisfy `inf(D[k]) > 0`.

# Input
- `A`: symmetric interval matrix to verify (assumed symmetric)

# Return
- `true` if verified SPD; `false` otherwise

# Notes
- Failure does NOT imply "not SPD"; it means this verification attempt did not certify SPD.
"""
function verify_spd_by_ldlt(A::AbstractMatrix{dtype})
    n, m = size(A)
    @assert n == m "A must be square"

    # Allocate
    L = Matrix{dtype}(undef, n, n)
    D = Vector{dtype}(undef, n)

    # Initialize L to identity (as intervals)
    @inbounds for i in 1:n, j in 1:n
        L[i,j] = (i == j) ? one(dtype) : zero(dtype)
    end

    # LDLᵀ factorization (no pivoting)
    @inbounds for k in 1:n
        # Compute D[k] = A[k,k] - sum_{j<k} L[k,j]^2 * D[j]
        s = zero(dtype)
        for j in 1:k-1
            s += (L[k,j] * L[k,j]) * D[j]
        end
        Dk = A[k,k] - s
        D[k] = Dk

        if inf(Dk) <= 0
            println("Failed at k=$k: pivot interval D[$k] is not strictly positive. inf(D[$k])=$(inf(Dk)).")
            return false
        end

        # Compute L[i,k] for i>k:
        # L[i,k] = (A[i,k] - sum_{j<k} L[i,j]*L[k,j]*D[j]) / D[k]
        for i in k+1:n
            t = zero(dtype)
            for j in 1:k-1
                t += (L[i,j] * L[k,j]) * D[j]
            end
            num = A[i,k] - t
            L[i,k] = num / Dk
        end
    end

    return true
end

"""
    verify_general_eig(A, B, eig_vec, eig_val, thr)
Strictly verifies the generalized eigenvalue problem A * x = λ * B * x
for the given eigenvalues and eigenvectors, within a threshold thr.
"""
function verify_general_eig(A::AbstractMatrix{dtype}, B::AbstractMatrix{dtype}, eig_vec::AbstractMatrix{dtype}, eig_val::AbstractVector{dtype}, thr::dtype)
    d = Diagonal(inv.(eig_val));
    C = A * eig_vec;
    S = thr * B - A + C * d * C'
    return verify_spd_by_ldlt(S)
end

"""
    eig_sym3(a, b, c, d, e, f)

Eigenvalues (or interval enclosures) of the symmetric 3×3 matrix
    [ a  b  c
      b  d  e
      c  e  f ].

Uses the cubic/trigonometric formula with
    q = tr(A)/3,  p = sqrt(‖A - qI‖_F^2 / 6),  r = det(A - qI)/(2p^3),
then ϕ = acos(r)/(3π) and λ = q + 2p*cos(·).

- If `is_zero(p)` is true (scalar/degenerate case), returns `[q, q, q]`.
- Otherwise clamps `r` to `[-1,1]` via `intersect_interval(r, ACOS_INT)` to keep `acos` in-domain.

This algorithm is safe even if p is very small.

Returns a length-3 vector `[λ1, λ2, λ3]` (ordering may overlap for intervals).
"""
const ACOS_INT = @interval(-1, 1)
function eig_sym3(a,b,c,d,e,f)
    q = (a + d + f) / dt(3)
    aq, dq, fq = a - q, d - q, f - q
    p  = sqrt((aq*aq + dq*dq + fq*fq + dt(2)*(b*b + c*c + e*e)) / dt(6))
    # Avoid to use power function on intervals for better performance

    # If p is zero, the eigenvalues are [q, q, q]
    if is_zero(p)
        return [q, q, q]
    end

    detAqI =
        aq * (dq*fq - e*e) -
        b  * (b*fq - c*e) +
        c  * (b*e  - c*dq)

    two_p = dt(2)*p
    r = detAqI / (two_p*p*p)
    r = intersect_interval(r, ACOS_INT)  # clamp to [-1, 1] to avoid domain error in acos due to numerical issues
    ϕ = acos(r) / dt(3) / Pi

    λ1 = q + two_p * cospi(ϕ)
    λ2 = q + two_p * cospi(ϕ + dt(2)/dt(3))
    λ3 = q + two_p * cospi(ϕ + dt(4)/dt(3))

    return [λ1, λ2, λ3]
end

"""
Convert a matrix of symmetric components into 3×3 symmetric matrices.
Each row of bound_lst is interpreted as (a11, a12, a13, a22, a23, a33).

# Input:
- bound_lst: matrix of length-6 iterables
# Return:
- A vector of 3×3 symmetric matrices
"""
function vec_to_mat(bound_lst::Matrix{dtype})
    result = []
    for each in eachrow(bound_lst)
        a11, a12, a13, a22, a23, a33 = each
        mat = [
            a11  a12  a13
            a12  a22  a23
            a13  a23  a33
        ]
        push!(result, mat)
    end
    return result
end