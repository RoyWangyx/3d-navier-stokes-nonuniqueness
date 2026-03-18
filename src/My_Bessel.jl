const DEFAULT_PREC = 128

"""
Supremum bound for |d²/dr² J_ν(r)| over r ∈ [a,b], for half-integer orders ν = l + 1/2:
|J_{l+1/2}''(r)|
≤ sqrt(2 / (π(2l+1))) * ( 1/(4 a^(3/2)) + 1/(sqrt(3) a^(1/2)) + b^(1/2)/sqrt(5) )

# Input:
- ν: Bessel order (must be half-integer ν = l + 1/2, l ≥ 0)
- a, b : Arb values with 0 < a ≤ b
- prec: working precision in bits

# Return:
- Arb value representing the upper bound
"""
function bessel_W2_bound_interval(ν::Real, a::Arb, b::Arb; prec::Int = DEFAULT_PREC)
    # interval must satisfy 0 < a ≤ b
    @assert Arblib.is_positive(a) "need a > 0"
    @assert Arblib.is_positive(b) "need b > 0"
    @assert a <= b

    # ν = l + 1/2
    l = round(Int, ν - 0.5)
    @assert ν == l + 0.5 && l ≥ 0

    # π
    pi_arb = Arb(prec=prec)
    Arblib.const_pi!(pi_arb)

    # sup over [a,b]:
    # decreasing terms → evaluate at a
    # increasing term  → evaluate at b
    return sqrt(Arb(2)/(pi_arb*(2l+1))) *
           (inv(Arb(4)*a^(Arb(3)/2)) + inv(sqrt(Arb(3))*a^(Arb(1)/2)) + b^(Arb(1)/2)/sqrt(Arb(5)))
end

"""
Evaluate the Bessel function of the first kind J_ν(r).

Three input types are supported:
- r::Interval{Float64}: evaluated using Arb with rigorous enclosure
- r::Arb: evaluated directly by Arblib
- r::Float64: evaluated using Julia's built-in besselj

# Input:
- ν: Bessel order
- r: argument (Interval{Float64}, Arb, or Float64)
- prec: working precision in bits (only for Interval/Arb versions)

# Return:
- Interval{Float64}, Arb, or Float64 value of J_ν(r), depending on input type
"""
function my_besselj(ν::Real, r::Interval{Float64}; prec::Int = DEFAULT_PREC)
    # If r = 0, return 0
    if is_zero(r)
        return @interval(0)
    end
    @assert inf(r) > 0 "Interval must be either [0,0] or strictly positive"

    # Convert Float64 interval endpoints to Arb (rigorous enclosures at higher precision)
    lo = Arb(inf(r); prec = prec)
    hi = Arb(sup(r); prec = prec)

    # Evaluate J_ν at both endpoints with Arb
    val_lo = my_besselj(ν, lo; prec = prec)
    val_hi = my_besselj(ν, hi; prec = prec)

    # Upper bound on sup_{x ∈ [lo, hi]} |d²/dr² J_ν(x)|.
    bound = bessel_W2_bound_interval(ν, lo, hi; prec = prec)

    # Remainder bound for enclosing J_ν on [lo, hi] using endpoints:
    #   max deviation from the chord ≤ (sup |J''|) * (hi-lo)^2 / 8
    err = bound * ((hi - lo) ^ 2) / Arb(8)

    # Enlarge endpoint enclosures by the remainder to cover interior points
    Arblib.add_error!(val_lo, err)
    Arblib.add_error!(val_hi, err)

    # Convert Arb enclosures back to Float64 endpoints
    val_lo = Interval{Float64}(val_lo)
    val_hi = Interval{Float64}(val_hi)

    # Final Float64 interval enclosure of J_ν([inf(r), sup(r)])
    return hull(val_lo, val_hi)
end

function my_besselj(ν::Real, r::Arb; prec::Int = DEFAULT_PREC)
    # Convert ν to Arb
    ν_arb = Arb(ν; prec = prec)

    # Compute Bessel function value
    val_arb = Arb(prec = prec)
    Arblib.hypgeom_bessel_j!(val_arb, ν_arb, r; prec = prec)
    return val_arb
end

function my_besselj(ν::Real, r::Float64)
    # Compute Bessel function value directly for Float64 input
    return besselj(ν, r)
end

"""
Compute the function f_Z(l, x) derived from the spherical Bessel equation:
    f_l(x) = x * J_{l-1/2}(x) - (l+1) * J_{l+1/2}(x),
which can be rewritten as
    f_Z(l, x) = x * J_{l-1/2}(x) / J_{l+1/2}(x) - (l+1).
# Inputs:
- `l::Int`   : spherical-harmonic degree (so the Bessel orders are `l ± 1/2`).
- `x::dtype` : evaluation point.
# Returns:
- `dtype` : value of `f_Z(l, x)` in the same numeric type as `x`. 
"""
function func_Z(l::Int, x::dtype)
    # For integer l, the expression (l - 0.5) is exact (no floating-point rounding).
    return x * my_besselj(l - 0.5, x) / my_besselj(l + 0.5, x) - dt(l + 1)
end

"""
Compute the function f_Y(l, x) derived from the spherical Bessel equation:
    f_l(x) = x * J_{l-1/2}(x) - (l+2) * J_{l+1/2}(x)
             + ((x^2 - (l-1)l)(x^2 - (l-1)(l+2))) / (x^2 - (l-1)l(l+2)) * J_{l+1/2}(x),
which can be rewritten as
    For l ≥ 2:
        f_Y(l, x) = x * J_{l-1/2}(x) / J_{l+1/2}(x) - (l+2)
                    + ((x^2 - (l-1)l) * (x^2 - (l-1)(l+2)))
                      / (x^2 - (l-1)l(l+2))
    For l = 1:
        f_Y(1, x) = x * J_{1/2}(x) / J_{3/2}(x) - 3 + x^2
# Inputs:
- `l::Int` : spherical-harmonic degree (so the Bessel orders are `l ± 1/2`).
- `x::dtype` : evaluation point.
# Returns:
- `dtype` : value of `f_Y(l, x)` in the same numeric type as `x`. 
"""
function func_Y(l::Int, x::dtype)
    # For integer l, the expression (l - 0.5) is exact (no floating-point rounding).
    if l >= 2
        return x * my_besselj(l - 0.5, x) / my_besselj(l + 0.5, x) - dt(l + 2) + 
               (x^2 - dt((l - 1) * l)) * (x^2 - dt((l - 1) * (l + 2))) / (x^2 - dt((l - 1) * l * (l + 2)))
    else
        return x * my_besselj(l - 0.5, x) / my_besselj(l + 0.5, x) - dt(l + 2) + x^2
    end
end

# Return true if the two interval values have opposite signs (guaranteed sign change).
@inline function sign_change(fa::Interval{Float64}, fb::Interval{Float64})
    return (inf(fa) > 0 && sup(fb) < 0) || (sup(fa) < 0 && inf(fb) > 0)
end

"""
Find a rigorous interval containing a root of a given function using bisection.
This routine starts from an interval [x0 - δ, x0 + δ] and then applies bisection
until the interval width is less than `tol`. The method is based on rigorous 
interval arithmetic with Arb.
# Arguments
- `l::Real` : parameter passed to the function `func`.
- `x0::Arb` : approximate root, given as an Arb value.
- `func::Function` : function of two variables `(l, x)` returning an Arb value.
- `δ::Real=1e-14` : initial half-width of the search interval.
- `prec::Int=200` : precision (bits) for Arb calculations.
- `tol::Real=1e-20` : stopping tolerance for bisection (interval width).
# Returns
- `Interval{Float64}` : a floating-point interval enclosing the root.
# Throws AssertionError if no sign change is detected in the initial interval.
"""
function rigorous_root_interval(l::Real, x0::Interval{Float64}, func::Function; δ=dt("1e-10"), prec=200, tol=1e-20)
    if is_zero(x0)
        return zero(dtype) # Return 0 if x0 is exactly zero, as it's a root of the function.
    end

    # Check for sign change in the initial interval
    a = dt(inf(x0)) - δ
    b = dt(sup(x0)) + δ
    fa = func(l, a)
    fb = func(l, b)
    @assert sign_change(fa, fb) "No sign change in initial interval! Increase δ."

    # Bisection method
    while inf(abs(b - a)) >= tol
        m = (a + b) / dt(2)
        fm = func(l, m)
        if sign_change(fa, fm)
            b, fb = m, fm
        else
            a, fa = m, fm
        end
    end

    # Return the final interval enclosure
    m = hull(a, b)
    return m
end

"""
Verify that a given value `x0` is a valid root candidate of `func(l, ·)`.
# Arguments
- `l::Real`       : parameter passed to the function `func`.
- `x0::Float64`   : candidate root.
- `func::Function`: function of two variables `(l, x)` returning a real value.
- `δ::Real=1e-13` : relative perturbation size.
# Returns
- `x0::Float64` if the sign change test passes.
# Throws AssertionError if no sign change is detected around `x0`.
"""
function verify_root(l::Real, x0::Float64, func::Function; δ::Real=1e-13)
    if x0 > 0
        left  = func(l, x0 * (1 - δ))
        right = func(l, x0 * (1 + δ))
        @assert left * right < 0 "Sign change expected around l=$l, x0=$x0; left=$left, right=$right"
    end
    return x0
end

"""
Convert a 2D matrix of candidate values into a nested list of verified roots. In 
`get_bessel_zeros.ipynb`, a nested list of roots was converted into a matrix. This 
function reverses that step: it takes the matrix representation and restores the 
nested list structure, verifying each entry along the way.
# Arguments
- `mat::AbstractMatrix`    : input matrix of candidate root values.
- `verify_func::Function`  : function `(l, value, eq_func) -> verified_value`.
- `eq_func::Function`      : equation function passed to `verify_func`.
# Returns
- `Vector{Vector{Any}}` : nested list of verified roots, one list per row of `mat`.
# Throws AssertionError if any row produces an empty list.
"""
function mat_to_lst(mat, verify_func, eq_func)
    result = []
    for l in axes(mat, 1)
        lst = []
        for j in axes(mat, 2)
            # Early termination if zero entry is found
            if(j > 1 && is_zero(mat[l, j]))
                break
            end

            # Verify and refine each candidate root.
            zero_l = verify_func(l, dt(mat[l, j]), eq_func)

            # push a list of roots since there are other parameters related to
            # each root to be added into the list later
            push!(lst, [zero_l])
        end
        @assert !isempty(lst) "empty list when l=$l"
        push!(result, lst)
    end
    return result
end

"""
Augment parameter lists for Y / Z bases.
- For `vari == "Y"`: for each entry `paras[l][j]` with first element `α`,
  append two scalars `(A,B)` where
    A = - (α^2 - (l+2)(l-1)) * J_{l+1/2}(α) / α^{3/2},
    B =   (α^2 - l(l-1)(l+2)) / α^{3/2}.
  If `α == 0`, append `(1, 1)`. They are the coefficients of the eigenfunctions.
  Refer to the paper for details.
- Otherwise: append a single trailing `1`.
# Inputs
- `paras` : nested list/array; each `paras[l][j][1]` is `α`.
- `vari`  : `"Y"` or "Z".
# Returns
- The mutated `paras` (also modified in place).
"""
function preprocess_para!(paras::Vector, vari::String)
    if vari == "Y"
        for l in eachindex(paras), j in eachindex(paras[l])
            α = paras[l][j][1]

            # Calculate the coefficients for X- and Y-type bases.
            if is_zero(α)
                A, B = one(dtype), one(dtype)
            else
                A = -(α^2 - dt((l+2)*(l-1))) * my_besselj(l + 0.5, α) / α^dt(3/2)
                B = (α^2 - dt(l*(l-1)*(l+2))) / α^dt(3/2)
            end

            # Add the coefficients to the list
            append!(paras[l][j], [A, B])
            if is_zero(α)
                continue
            end
        end
    else
        for l in eachindex(paras), j in eachindex(paras[l])
            push!(paras[l][j], one(dtype)) # For Z, we set the coefficient to be 1
        end
    end
    return paras
end

"""
Load Bessel zeros from a `.mat` file and convert them into parameter lists.
# Inputs
- `file_path::String` : path to `.mat` file with `"Y_zeros"` and `"Z_zeros"`.
- `refine::Bool=false`: if true, refine roots with `rigorous_root_interval`,
  else just verify with `verify_root`.
# Returns
- `(Y_paras, Z_paras)` : nested lists of processed parameters.
"""
function load_besselj_zeros(file_path::String; refine::Bool=false)
    # Load Bessel zeros from a file
    matfile = matopen(file_path)
    Y_zeros = read(matfile, "Y_zeros")
    Z_zeros = read(matfile, "Z_zeros")
    close(matfile)

    # Refine or just verify the roots
    if refine
        verify_func = rigorous_root_interval
    else
        verify_func = verify_root
    end

    # Convert the matrix to the nested lists
    Y_paras = mat_to_lst(Y_zeros, verify_func, func_Y)
    Z_paras = mat_to_lst(Z_zeros, verify_func, func_Z)

    # # Preprocessing
    Y_paras = preprocess_para!(Y_paras, "Y")
    Z_paras = preprocess_para!(Z_paras, "Z")
    return Y_paras, Z_paras
end

"""
Compute the radial function UX, UY, UZ at given r. This routine supports odd/even 
extension in `r`: it evaluates the formula using `abs.(r)` and then multiplies 
by a parity sign so that the resulting function has the intended symmetry.
# Inputs
- `l::Real`: degree index.
- `paras`: parameters; for "Z" → (α, A), for "X"/"Y" → (α, A, B). In practice, 
    "X" typically reuses the same `(α,A,B)` layout as "Y".
- `r`: vector of r where U_r is evaluated. r can be any real grid used for odd/even 
    extension.
- `vari::String` : one of "X", "Y", "Z".
# Returns
- `result::Vector{dtype}` : values of U_r at each r, with regularization at r=0.
"""
const EPS = dt("1e-16")
function Ur_func(l::Int, paras::Vector, r::Vector{dtype}, vari::String)
    # handle sign of r for odd/even extensions
    # For "Z": U_r is even if l is even, odd if l is odd. For "X","Y": the opposite.
    n = (vari == "Z") ? l : (l - 1)
    sign_r = isodd(n) ? dt.(sign.(r .+ EPS)) : ones(dtype, length(r))
    r = abs.(r)

    # check variant and unpack parameters
    @assert vari in ["X", "Y", "Z"]
    if vari == "Z"
        α, A = paras
        @assert isa(α, dtype) && isa(A, dtype)
    else
        # paras for vari="X" shares the same parameter layout as "Y" (α, A, B), 
        # and is typically provided via Y_paras.
        α, A, B = paras
        @assert isa(α, dtype) && isa(A, dtype) && isa(B, dtype)
    end

    # evaluate formula depending on variant
    if l == 1 && is_zero(α)
        return ones(dtype, length(r)) .* sign_r
    elseif vari == "X"
        result = A * (r.^dt(l-1)) + (B/dt(l*(l+1))) * (α*my_besselj.(l - 0.5, α*r)
                - dt(l) * my_besselj.(l + 0.5, α*r)./r) ./ sqrt.(r)
    elseif vari == "Y"
        result = dt(l) * A * (r.^dt(l-1)) + B * my_besselj.(l + 0.5, α*r) ./ (r.^dt(3/2))
    else
        result = A * my_besselj.(l + 0.5, α*r) ./ sqrt.(r)
    end

    # Handle r=0 case. The value at `r = 0` is set by a special-case limit:
    # only the `(l == 1, vari in ("X","Y"))` branch yields a nonzero value;
    # all other cases are set to zero at `r = 0`.
    mask = is_zero.(r)
    if l == 1 && vari in ["X", "Y"]
        result[mask] .= A + sqrt(dt(2)/Pi)/dt(3)*B*α^dt(3/2)
    else
        result[mask] .= zero(dtype)
    end
    return result .* sign_r
end

"""
Evaluate radial functions U_r for a list of parameters.
# Inputs
- `r::Vector{dtype}` : radial grid.
- `paras_lst`        : nested list of parameters for each l and zero.
- `vari::String`     : one of "X","Y","Z".
# Returns
- `Ur_lst::Vector{Matrix{dtype}}` : list of matrices; each matrix has size (length(r), number of zeros for that l)).
"""
function eval_Ur(r::Vector{dtype}, paras_lst::Vector, vari::String)
    Ur_lst = []
    N_r = length(r)
    for l in eachindex(paras_lst)
        lst = zeros(dtype, N_r, length(paras_lst[l]))
        for j in eachindex(paras_lst[l])
            lst[:, j] = Ur_func(l, paras_lst[l][j], r, vari)
        end
        push!(Ur_lst, lst)
    end
    return Ur_lst
end

"""
Precompute θ-dependent spherical-harmonic basis rows up to degree `Lmax` for the four
θ-boundary-condition types.

# Inputs
- `θ::Vector{dtype}` : θ grid points.
- `Lmax::Int`        : maximum spherical degree l to be supported (l ≥ 1).

# Returns
- `res_lst::Vector{Matrix{dtype}}` : length-4 list corresponding to boundary codes
  `["00","01","10","11"]`. For each boundary code `bdry`, `res_lst[i]` is a matrix
  `U` of size `(lh, N_θ)`, where `N_θ = length(θ)` and `lh = (Lmax + 1) ÷ 2`.

  Each row of `U` stores θ-basis values on the grid `θ` for one parity family
  (even/odd with respect to θ = π/2). Degrees are therefore not indexed by `1:Lmax`
  directly; instead, they are grouped by parity and accessed via

      row(l) = (l + 1) ÷ 2,

  so the θ-basis vector for degree `l` is obtained by

      Uθ_l = res_lst[i][row(l), :].
"""
function eval_Uθ(θ::Vector{dtype}, l::Int)
    bdry_lst = ["00", "01", "10", "11"]
    res_lst = []

    # Divide the first l spherical harmonic basis functions into even/odd groups 
    # by symmetry at θ = π/2, giving lh = (l+1)/2 functions in each group
    lh = (l+1)÷2
    for bdry in bdry_lst
        conv_mat = conversion_mat(lh, bdry, 1)
        res = conv_mat * trig_func(lh+(bdry=="11"), θ, "I", bdry)'
        push!(res_lst, res)
    end
    return res_lst
end

"""
Evaluate eigenfunctions at (r, θ) from given eigenvectors.
This routine reconstructs the spatial representation of eigenmodes from their
coefficient vectors in the Y/Z basis. It first computes angular parts Uθ,
then combines them with radial parts Ur, and finally assembles the three
Cartesian/vector components.

Mathematically, for an eigenmode with coefficient vector {c_Y, c_X, c_Z}, the
reconstructed field is
    u(r_i, θ_j) = ∑_{l,k} [
        c_{Y,lk} * Ur_Y_{lk}(r_i) * Uθ_Y_{lk}(θ_j)
      + c_{X,lk} * Ur_X_{lk}(r_i) * Uθ_X_{lk}(θ_j)
      + c_{Z,lk} * Ur_Z_{lk}(r_i) * Uθ_Z_{lk}(θ_j)
    ],
where the sum runs over degrees l and their associated basis function index k.

# Inputs
- `eig_vec_odd, eig_vec_even` : coefficient matrices of eigenvectors, split into
    odd/even groups (size N × L, where N is total number of basis functions and
    L is number of eigenmodes in that group).
- `Ur_Y_lst, Ur_X_lst, Ur_Z_lst` : lists of radial basis matrices for Y-, X-, and Z-type
    bases, grouped by spherical degree l.
- `r::Vector{dtype}`  : radial grid points.
- `θ::Vector{dtype}`  : angular grid points (for θ-direction evaluation).

# Returns
- `(res_even, res_odd)` : reconstructed eigenfunctions on the (r, θ) grid.  
    Each result is an array of size (N_r, N_θ, L, 3), where  
    • N_r = number of radial grid points,  
    • N_θ = number of angular grid points,  
    • L   = number of eigenmodes in the corresponding group,  
    • the last dimension (3) stores the three vector components, assembled from
      Y-, X-, and Z-type basis functions.
"""
function eval_Q_eig_function(eig_vec_odd::AbstractMatrix{dtype}, eig_vec_even::AbstractMatrix{dtype},
    Y_paras::Vector, Z_paras::Vector, r::Vector{dtype}, θ::Vector{dtype}, Q::VectorFunc)
    Ur_Y_lst = eval_Ur(r, Y_paras, "Y");
    Ur_X_lst = eval_Ur(r, Y_paras, "X");
    Ur_Z_lst = eval_Ur(r, Z_paras, "Z");

    # Precompute θ-basis tables up to Lmax.
    # NOTE: eval_Uθ returns, for each bdry in ["00","01","10","11"], a matrix of size (lh, N_θ),
    # where lh = (Lmax+1)÷2 and row(l) = (l+1)÷2 selects the θ-basis vector for degree l.
    Uθ_lst = eval_Uθ(θ, max(length(Ur_Y_lst), length(Ur_Z_lst)))

    # Grid sizes
    N_r = length(r)
    N_θ = length(θ)

    # Sizes of coefficient matrices: (num_basis, num_modes)
    N_even, L_even = size(eig_vec_even)
    N_odd, L_odd = size(eig_vec_odd)

    # Store them in a list for convenience: [even, odd]
    eig_vec_lst = [eig_vec_even, eig_vec_odd]

    # Result arrays: (N_r × N_θ × L × 3), last dim = vector components (Y, X, Z)
    rhs_lst = [zeros(dtype, N_r, N_θ, L_even, 3), zeros(dtype, N_r, N_θ, L_odd, 3)]

    # Mapping from (parity-group index 'ind') to which θ-boundary table to use for each component.
    # Uθ_lst is ordered by bdry codes: [ "00", "01", "10", "11" ].
    #
    # For each parity group:
    #   ind_lst[ind][1] -> boundary code index used for Y-component angular basis
    #   ind_lst[ind][2] -> boundary code index used for X-component angular basis
    #   ind_lst[ind][3] -> boundary code index used for Z-component angular basis
    #
    # The choice differs between the two parity groups because the eigenvector coefficients
    # are split into even/odd angular parity blocks (ind = 1 for even, 2 for odd.), and the 
    # Y/X/Z bases have different symmetry.
    ind_lst = [[4, 1, 2], [3, 2, 1]]

    # Current basis counter for each parity group
    cur_l = [0, 0]

    # Y and X contributions
    for l in eachindex(Ur_Y_lst)
        ind = mod(l, 2) + 1 # index convention for Y: 1 → odd, 2 → even
        Ll = size(Ur_Y_lst[l], 2) # number of basis functions at degree l

        # Angular functions for Y and X components
        Uθ_Y = Uθ_lst[ind_lst[ind][1]][(l+1)÷2, :]
        Uθ_X = Uθ_lst[ind_lst[ind][2]][(l+1)÷2, :]

        # Radial part (N_r × Ll) × eigenvector coefficients (Ll × L). Sum over 
        # all basis function indices k under fixed l (from 1 to Ll). Get (N_r × L)
        Ur_Y_mat = Ur_Y_lst[l] * eig_vec_lst[ind][cur_l[ind]+1: cur_l[ind]+Ll, :]
        Ur_X_mat = Ur_X_lst[l] * eig_vec_lst[ind][cur_l[ind]+1: cur_l[ind]+Ll, :]

        # Multiply radial part (N_r × 1 × L) with angular part (1 × N_θ × 1)
        U_Y = reshape(Uθ_Y, 1, :, 1) .* reshape(Ur_Y_mat, N_r, 1, :)
        U_X = reshape(Uθ_X, 1, :, 1) .* reshape(Ur_X_mat, N_r, 1, :)

        # Accumulate into result: component 1=Y, 2=X
        rhs_lst[ind][:, :, :, 1] += U_Y
        rhs_lst[ind][:, :, :, 2] += U_X

        # Advance counter
        cur_l[ind] += Ll
    end

    # Z contributions: similar to above
    for l in eachindex(Ur_Z_lst)
        ind = mod(l+1, 2) + 1 # index convention for Z: 2 → odd, 1 → even
        Ll = size(Ur_Z_lst[l], 2)
        Uθ_Z = Uθ_lst[ind_lst[ind][3]][(l+1)÷2, :]
        Ur_Z_mat = Ur_Z_lst[l] * eig_vec_lst[ind][cur_l[ind]+1: cur_l[ind]+Ll, :]# N_r * L_even
        U_Z = reshape(Uθ_Z, 1, :, 1) .* reshape(Ur_Z_mat, N_r, 1, :)
        rhs_lst[ind][:, :, :, 3] += U_Z
        cur_l[ind] += Ll
    end

    # Check: all basis functions have been consumed
    @assert cur_l[1] == N_even "cur_l[1] = $cur_l[1], N_even = $N_even"
    @assert cur_l[2] == N_odd "cur_l[2] = $cur_l[2], N_odd = $N_odd"

    Q_pts = interpolate(Q, r .* (Pi/dt(2)), θ)

    result_even = zeros(dtype, size(rhs_lst[1]))
    result_odd = zeros(dtype, size(rhs_lst[2]))
    key_mat = [["A11", "A12", "A13"],
               ["A12", "A22", "A23"],
               ["A13", "A23", "A33"]]
    for i in 1:3, j in 1:3
        result_even[:, :, :, i] += Q_pts[key_mat[i][j]].space_func .* rhs_lst[1][:, :, :, j]
        result_odd[:, :, :, i] += Q_pts[key_mat[i][j]].space_func .* rhs_lst[2][:, :, :, j]
    end
    return result_even, result_odd
end
