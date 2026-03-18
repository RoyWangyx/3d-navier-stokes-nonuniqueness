"""
    struct TrigPoly

Container for a single-variable trigonometric polynomial:
- `c::Vector{dtype}`: Cosine coefficients, where `c[k+1]` multiplies `cos(kx)` for `k ≥ 0`.
    `c[1]` corresponds to the coefficient of `cos(0x)`
- `s::Vector{dtype}`: Sine coefficients, where `s[k]` multiplies `sin(kx)` for `k ≥ 1`.
    `s[1]` corresponds to the coefficient of `sin(1x)`
"""
struct TrigPoly
    c::Vector{dtype}  # cos coeffs: c[k+1] for cos(kx), k>=0
    s::Vector{dtype}  # sin coeffs: s[k]   for sin(kx), k>=1
end

############################
# Multiplication
############################
"""
Add a cosine term into the coefficient vector.
Add val * cos(kx) into c, handling symmetry and resizing automatically.

# Input:
- c: cosine coefficient vector
- k: frequency index (negative allowed)
- val: coefficient to add
"""
@inline function _add_cos!(c::Vector{dtype}, k::Int, val::dtype)
    # add val * cos(kx), k>=0
    if k < 0
        k = -k
    end
    idx = k + 1
    if idx > length(c)
        old = length(c)
        resize!(c, idx)
        fill!(view(c, old+1:idx), zero(dtype))
    end
    c[idx] += val
end

"""
Add a sine term into the coefficient vector. Add val * sin(kx) into s, handling
sign symmetry and resizing automatically. k = 0 is ignored since sin(0x) = 0.

# Input:
- s: sine coefficient vector
- k: frequency index (negative allowed)
- val: coefficient to add
"""
@inline function _add_sin!(s::Vector{dtype}, k::Int, val::dtype)
    # add val * sin(kx), allow negative k via sin(-kx)=-sin(kx)
    if k == 0
        return
    elseif k < 0
        k = -k
        val = -val
    end
    idx = k  # s[k] corresponds to sin(kx)
    if idx > length(s)
        old = length(s)
        resize!(s, idx)
        fill!(view(s, old+1:idx), zero(dtype))
    end
    s[idx] += val
end

"""
Multiply two trigonometric polynomials using product-to-sum identities:
- cos(mx)cos(nx) = (cos((m-n)x) + cos((m+n)x))/2
- sin(mx)sin(nx) = (cos((m-n)x) - cos((m+n)x))/2
- sin(mx)cos(nx) = (sin((m+n)x) + sin((m-n)x))/2

# Degree convention (storage-based)
Let
    deg(P) = max( length(P.c)-1, length(P.s) ).
Then `deg(A*B) ≤ deg(A) + deg(B)`, and with full storage allocation we create
arrays up to that maximum degree.

# Input:
- `A::TrigPoly`: first trigonometric polynomial
- `B::TrigPoly`: second trigonometric polynomial

# Return:
- TrigPoly representing A * B
"""
function trig_mul(A::TrigPoly, B::TrigPoly)
    degA = max(length(A.c) - 1, length(A.s))
    degB = max(length(B.c) - 1, length(B.s))
    deg  = degA + degB

    c = zeros(dtype, deg + 1)  # cos(0..deg)
    s = zeros(dtype, deg)      # sin(1..deg)

    # cos*cos
    for m0 in 0:(length(A.c)-1)
        am = A.c[m0+1]
        is_zero(am) && continue
        for n0 in 0:(length(B.c)-1)
            bn = B.c[n0+1]
            is_zero(bn) && continue
            v = am * bn / dt(2)
            _add_cos!(c, abs(m0 - n0), v)
            _add_cos!(c, m0 + n0, v)
        end
    end

    # sin*sin
    for m in 1:length(A.s)
        am = A.s[m]
        is_zero(am) && continue
        for n in 1:length(B.s)
            bn = B.s[n]
            is_zero(bn) && continue
            v = am * bn / dt(2)
            _add_cos!(c, abs(m - n),  v)
            _add_cos!(c, m + n,      -v)
        end
    end

    # sin*cos
    for m in 1:length(A.s)
        am = A.s[m]
        is_zero(am) && continue
        for n0 in 0:(length(B.c)-1)
            bn = B.c[n0+1]
            is_zero(bn) && continue
            v = am * bn / dt(2)
            _add_sin!(s, m + n0, v)
            _add_sin!(s, m - n0, v)
        end
    end

    # cos*sin
    for m0 in 0:(length(A.c)-1)
        am = A.c[m0+1]
        is_zero(am) && continue
        for n in 1:length(B.s)
            bn = B.s[n]
            is_zero(bn) && continue
            v = am * bn / dt(2)
            _add_sin!(s, n + m0, v)
            _add_sin!(s, n - m0, v)
        end
    end

    return TrigPoly(c, s)
end

############################
# Integral on [0, Pi/2]
############################
"""
Compute the exact value of sin(kπ/2) using k mod 4, avoiding floating-point error.
Use: ∫ sin(kx) dx = (1 - cos(kx)) / k   (k ≥ 1).

# Input:
- k: integer index

# Return:
- Integer in {-1, 0, 1}
"""
@inline _sin_kPi_over2(k::Int)::Int =
    (r = mod(k,4); r==0 ? 0 : r==1 ? 1 : r==2 ? 0 : -1)

"""
Compute the exact value of cos(kπ/2) using k mod 4, avoiding floating-point error.
Use: ∫ cos(kx) dx = sin(kx) / k   (k ≥ 1), and ∫ cos(0x) dx = x.

# Input:
- k: integer index

# Return:
- Integer in {-1, 0, 1}
"""
@inline _cos_kPi_over2(k::Int)::Int =
    (r = mod(k,4); r==0 ? 1 : r==1 ? 0 : r==2 ? -1 : 0)

"""
Compute the integral of a trigonometric polynomial over [0, π/2].

Evaluate ∫₀^{π/2} P(x) dx coefficient-wise, using exact values of sin(kπ/2) and
cos(kπ/2) via `_sin_kPi_over2` and `_cos_kPi_over2` to avoid floating-point error.

# Input:
- P: trigonometric polynomial

# Return:
- Integral value
"""
function trig_int_0_pi2(P::TrigPoly)
    res = zero(dtype)

    # cos part
    if length(P.c)  > 0
        res += P.c[1] * (Pi / dt(2))
        for k in 1:(length(P.c)-1)
            ck = P.c[k+1]
            is_zero(ck) && continue
            res += ck * dt(_sin_kPi_over2(k)) / dt(k)
        end
    end

    # sin part
    for k in 1:length(P.s)
        sk = P.s[k]
        is_zero(sk) && continue
        res += sk * dt(1 - _cos_kPi_over2(k)) / dt(k)
    end

    return res
end

"""
Construct cosine and sine basis lists up to order l.
Each element is a TrigPoly with exactly one nonzero basis coefficient equal to 1.

# Input:
- l: nonnegative integer

# Return:
- two lists of TrigPoly:
    cos_list[j+1] = cos(jx) for j = 0..l (so cos_list[1] = 1)
    sin_list[j]   = sin(jx) for j = 1..l
"""
function trig_basis_lists(l::Int)
    @assert l >= 0

    cos_list = Vector{TrigPoly}(undef, l + 1)
    for j in 0:l
        c = zeros(dtype, j+1)  # need slots up to cos(jx)
        c[j + 1] = one(dtype)
        cos_list[j + 1] = TrigPoly(c, dtype[])  # no sin part
    end

    sin_list = Vector{TrigPoly}(undef, l)
    for j in 1:l
        s = zeros(dtype, j)      # need slots up to sin(jx)
        s[j] = one(dtype)
        sin_list[j] = TrigPoly(dtype[], s)  # no cos part
    end
   
    return cos_list, sin_list
end

"""
Construct basis lists of trigonometric polynomials for even order l.

# Input:
- l: nonnegative even integer

# Return:
- two lists of TrigPoly:
    P_list: cosine-only bases (TrigPoly(c, []))
    dP_list: sine-only bases (TrigPoly([], s))
"""
function sph_basis_lists(l::Int)
    @assert l >= 0 && iseven(l)
    dP_list = Vector{TrigPoly}(undef, l)
    P_list = Vector{TrigPoly}(undef, l)
    for j in 1:l
        s = zeros(dtype, j)
        c = zeros(dtype, j + 1)
        dP_list[j] = TrigPoly(dtype[], s)  # no cos part
        P_list[j] = TrigPoly(c, dtype[])  # no sin part
    end

    bdry_lst = ["00", "01"]
    lh = l÷2
    for bdry in bdry_lst
        conv_mat = conversion_mat(lh, bdry, 1)
        id = bdry == "00" ? 2 : 1
        for i in 1:lh
            dP_list[2*i+id-2].s[id:2:end] = conv_mat[i, 1:i]
        end
    end

    bdry_lst = ["10", "11"]
    for bdry in bdry_lst
        conv_mat = conversion_mat(lh, bdry, 1)
        id = bdry == "10" ? 0 : 1
        for i in 1:lh
            P_list[2*i+id-1].c[2-id:2:end] = conv_mat[i, 1:i+id]
        end
    end

    return P_list, dP_list
end

"""
Precompute a θ-basis integral tensor on `[0, π/2]`. This function builds a lookup
table `T` of integrals of triple products of basis functions (two spherical harmonic
bases times one trig basis), with the spherical-coordinate weight `sin(θ)` included:

    T[l, m, n] = ∫_0^{π/2} f_l(θ) f_m(θ) h_n(θ) sin(θ) dθ.

Integration is exact for your TrigPoly representation via `trig_int_0_pi2`. Then 
`contract_QT_blocks` can assemble integrals against 'Q' expanded in the same trig 
basis efficiently without repeated θ quadrature.

# Inputs
- `trig_lists = (cos_list, sin_list)`: Output of `trig_basis_lists(Nmax)`.
  - `sin_list[n]` represents `sin(nθ)` for `n = 1..Nmax1`.
  - `cos_list[k+1]` represents `cos(kθ)` for `k = 0..Nmax2` (so `cos_list[1] = cos(0θ)`).

- `sph_lists = (p_list, dp_list)`: Output of `sph_basis_lists(Lmax)`
  - `p_list[l]` is basis functions from the Legendre P_l(cosθ) family (cosine-only
    TrigPoly), i.e. spherical harmonics.
  - `dp_list[l]` is corresponding θ-derivative family (sine-only TrigPoly), representing 
    dP_l(cosθ) / dθ.

# Indexing convention used in `T`
We pack the two families into one index for convenience.

- Let `Lmax1 = length(dp_list)`, `Lmax2 = length(p_list)`.
  Define
  - for `l = 1..Lmax1`:           `f_l = dp_list[l]`
  - for `l = Lmax1+1..Lmax1+Lmax2`: `f_l = p_list[l - Lmax1]`

- Let `Nmax1 = length(sin_list)`, `Nmax2 = length(cos_list)`.
  Define
  - for `n = 1..Nmax1`:           `h_n = sin_list[n]`
  - for `n = Nmax1+1..Nmax1+Nmax2`: `h_n = cos_list[n - Nmax1]`

# Output
- `T::Array{dtype,3}` with size `(Lmax1+Lmax2, Lmax1+Lmax2, Nmax1+Nmax2)`.
"""
function tensor_int_basis(trig_lists, sph_lists)
    cos_list, sin_list = trig_lists
    p_list, dp_list = sph_lists
    Lmax1 = length(dp_list)
    Lmax2 = length(p_list)
    Nmax1 = length(sin_list)
    Nmax2 = length(cos_list)

    # sin(x) factor
    sin1 = (Nmax1 >= 1) ? sin_list[1] : error("sin_list must have at least sin(x).")

    T = zeros(dtype, Lmax1+Lmax2, Lmax1+Lmax2, Nmax1+Nmax2)

    for l in 1:Lmax1+Lmax2
        f1 = l > Lmax1 ? p_list[l-Lmax1] : dp_list[l]
        f1 = trig_mul(f1, sin1)  # dP_l * sin(x)
        # println("Computing T for l = $l / $(Lmax1+Lmax2)")

        for m in 1:Lmax1+Lmax2
            g = m > Lmax1 ? p_list[m-Lmax1] : dp_list[m]
            f2 = trig_mul(f1, g)

            for n in 1:Nmax1+Nmax2
                h = n > Nmax1 ? cos_list[n-Nmax1] : sin_list[n]
                f3 = trig_mul(f2, h)
                T[l, m, n] = trig_int_0_pi2(f3)
            end
        end
    end

    return T
end

"""
Assemble the θ-integrated coefficient tensor by contracting `Q` with `T`.

For each entry `(i,j)`, let `Q[i,j].space_func[a,n]` be the coefficient of the `n`th
stored θ-mode at the `a`th radial point. This function computes
    out[a, l, m, i, j] = sum_{n=1}^M Q[i,j].space_func[a,n] * T[l+s_i, m+s_j, ν_ij(n)]
where `s_i` and `s_j` select the `dP` or `P` block in the first two axes of `T`, and
`ν_{ij}(n)` is the trig-mode index determined by the boundary type of the `(i,j)` entry.

# How the (dP vs P) family is selected (the `st_i`, `st_j` offsets)
The first two dimensions of `T` stack two spherical families:
- indices `1:Lh`       = dP-family
- indices `Lh+1:2Lh`   = P-family

This code uses:
- P-family only for component index 1, otherwise dP-family. (See vector spherical harmonics)
So:
- if `i == 1` use rows `Lh+1:2Lh`, else use `1:Lh`
- if `j == 1` use cols `Lh+1:2Lh`, else use `1:Lh`

# How θ boundary conditions choose trig parity (the `add_m` / `idm` mapping)
Each `Q[key]` is stored in a θ trig basis whose parity depends on `bdry = Q[key].bdry[2]`:

- "00": sin(2mθ)
- "01": sin((2m+1)θ)
- "10": cos((2m+1)θ)
- "11": cos(2mθ)

`T[:,:,k]` stores integrals against the packed trig basis index `k`.
For a stored mode column `m = 1..M`, we compute:
- `add_m = (bdry[1] == '1') * 2M - (bdry[2] == '1')`
- `idm   = 2m + add_m`
so that `idm` lands in the correct sin/cos and even/odd block inside the 3rd axis of `T`.

# Inputs
- `Q::VectorFunc`: matrix-valued function on the radial grid. Each entry
  `Q[key].space_func` is an `N × M` matrix, where `N` is the number of radial
  points and `M` is the number of stored θ-modes.
- `T::Array{dtype,3}`: precomputed table T; see `tensor_int_basis` for its definition.
  Must have size `(L, L, 4M+1)` where `L` is even and `M` matches `Q[key].space_func`.

# Output
- `Array{dtype,5}` of size `(N, L/2, L/2, 3, 3)`, where `Lh = L/2`.

# Details
- The first two axes of `T` stack two spherical families:
  `1:L/2` for `dP` and `L/2+1:L` for `P`.
- Component index `1` uses the `P` block; component indices `2,3` use the `dP` block.
- For each entry, the boundary flag `Q[key].bdry[2]` determines which trig-mode
  slice `T[:, :, idm]` is used for each column `m`.
"""
function contract_QT_blocks(Q::VectorFunc, T::Array{dtype,3})
    key_mat = ["A11" "A12" "A13";
               "A12" "A22" "A23";
               "A13" "A23" "A33"]
    N, M = size(Q[1].space_func)

    # L: number of spherical-basis terms in θ. It stacks two families (dP and P),
    #    each providing Lh terms up to degree Lh, so total L = 2*Lh.
    # M*4+1: number of trigonometric basis terms in θ used to represent h_k(θ):
    #    sin(nθ) for n = 1..2M and cos(nθ) for n = 0..2M (two parity blocks encoded by idm).
    L = size(T, 1)
    Lh = L ÷ 2
    @assert iseven(L) "L must be even"
    @assert size(T) == (L, L, M*4+1) "Size of T does not match" 

    result = zeros(dtype, N, Lh, Lh, 3, 3)
    for i in 1:3, j in 1:3
        # For the first entry, use P_l, elsewhere use dP_l
        st_i = i==1 ? Lh : 0
        st_j = j==1 ? Lh : 0

        key = key_mat[i,j]
        mat = Q[key].space_func

        # add_m selects the trigonometric subspace determined by boundary conditions.
        # bdry = "ab" with a,b ∈ {'0','1'}:
        #   "00" (Dirichlet–Dirichlet) → sin(2nθ)
        #   "01" (Dirichlet–Neumann)   → sin((2n+1)θ)
        #   "10" (Neumann–Dirichlet)   → cos((2n+1)θ)
        #   "11" (Neumann–Neumann)     → cos(2nθ)
        # The shift add_m maps mode index m to the correct block in T[:,:,idm].
        bdry = Q[key].bdry[2]
        add_m = (bdry[1] == '1') * 2M - (bdry[2] == '1')
        for m in 1:M
            idm = 2m + add_m
            @views result[:, :, :, i, j] .+= reshape(mat[:, m], :, 1, 1) .* 
                reshape(T[st_i+1:st_i+Lh, st_j+1:st_j+Lh, idm], 1, Lh, Lh)
        end
    end
    return result
end

"""
Split tensor T into odd and even angular-parity blocks.
T is indexed in the second and third dimensions by interleaved spherical bases.
This routine separates them into two tensors corresponding to odd and even modes.

# Input:
- T: tensor of size (N, L, L, 3, 3)
- l: total number of angular modes (must be even)

# Return:
- (result_odd, result_even), each of size (N, l/2, l/2, 3, 3)
"""
function split_tensor(T::Array{dtype, 5}, l::Int)
    N = size(T, 1)
    lh = l ÷ 2
    result_even = zeros(dtype, N, lh, lh, 3, 3)
    result_odd = zeros(dtype, N, lh, lh, 3, 3)
    for i in 1:3, j in 1:3
        st_i = i==3 ? 1 : 2
        st_j = j==3 ? 1 : 2
        result_even[:, :, :, i, j] = T[:, st_i:2:l, st_j:2:l, i, j]
        result_odd[:, :, :, i, j] = T[:, (3-st_i):2:l, (3-st_j):2:l, i, j]
    end
    return result_odd, result_even
end

"""
Compute θ-integrated inner-product tensors of Q(r, θ) on a given radial grid.

For each r ∈ r_pt and each matrix entry (i,j), this routine computes
T_{ij}(r, l, m) = ∫₀^{π/2} Q_{ij}(r, θ) Φ_l(θ) Φ_m(θ) sin(θ) dθ,
where Φ_l(θ) is either P_l(cos θ) or dP_l(cos θ)/dθ.

# Input:
- Q: VectorFunc representing a 3×3 matrix function Q(r, θ)
- r_pt: radial grid points where the θ-integrals are evaluated
- Lmax: number of spherical-basis terms (even), used by sph_basis_lists
- Nmax: maximum trigonometric mode order in Q.

# Return:
- (result_odd, result_even): two tensors of size (length(r_pt), Lmax/2, Lmax/2, 3, 3)
"""
function integrate_θ(Q::VectorFunc, r_pt::Vector{dtype}, Lmax::Int, Nmax::Int)
    trig_lists = trig_basis_lists(Nmax)
    sph_lists = sph_basis_lists(Lmax)
    T = tensor_int_basis(trig_lists, sph_lists)
    Q_part = interpolate(Q, r_pt, dtype[]; keys=["I","I"], pt_flag=true)
    # println("Contracting Q with T to compute θ-integrated tensor...")
    contracted_T = contract_QT_blocks(Q_part, T)
    int_θ_lst = split_tensor(contracted_T, Lmax)
    return int_θ_lst
end

"""
Compute `W^{j,∞}` bounds in the θ-direction for each column of `X`.

This function treats the first axis of `X` as the r-direction, flattens all remaining
axes, and applies `get_Wk∞_norm_lst` to each column vector. For each column, it returns
the bounds of orders `0:k` using step sizes `[h, 0]` and parameter package
`[["freq", M*π/2], nothing]`.

# Inputs
- `X::AbstractArray{dtype}`: array whose first axis is the r-direction.
- `h::dtype`: mesh size in the r-direction.
- `k::Int`: highest order of the bound.
- `M::Int`: frequency parameter used to build `paras`.

# Output
- `Y::Array{dtype}`: array of size `(k+1, size(X)[2:end]...)`, where `Y[j+1, ...]`
  is the order-`j` bound returned by `get_Wk∞_norm_lst`.
"""
function int_θ_bound(X::AbstractArray{dtype}, h::dtype, k::Int, M::Int)
    paras = [["freq", dt(M)*Pi/dt(2)] , nothing]
    h = [h, zero(dtype)]
    N = size(X, 1)
    outsz = (k+1, tail(size(X))...)
    Y = Array{dtype}(undef, outsz)

    X2 = reshape(X, N, :)
    Y2 = reshape(Y, k+1, :)

    @threads for col in axes(X2, 2)
        @inbounds begin
            xcol = @view X2[:, col]
            Y2[:, col] .= get_Wk∞_norm_lst(xcol, h, k; paras=paras)
        end
    end

    return Y
end