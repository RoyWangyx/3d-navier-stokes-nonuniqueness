"""
Analytic W^{k,∞} bound for the radial basis function `U_r` of degree `l`.
This routine returns an explicit upper bound for the k-th radial derivative
of the basis `U_r^{(k)}` (in the normalized radial coordinate used throughout
the basis construction).  Refer to the paper for the derivation of these bounds.

# Inputs
- `paras`: parameters for the radial basis.
    • `vari == "Z"`: `(α, A)`
    • `vari ∈ {"X","Y"}`: `(α, A, B)`
- `vari`: basis family label ("X", "Y", or "Z").
- `l`: spherical degree index.
- `k`: derivative order (k ≥ 0).

# Output
- A scalar upper bound (dtype) for `‖∂_r^k U_r‖_{L^∞}` under the analytic estimates
  derived in the paper.
"""
function Ur_bound(paras::Vector, vari::String, l::Int, k::Int)
    if vari == "Z"
        α, A = paras
        @assert isa(α, dtype) && isa(A, dtype)
    else
        α, A, B = paras
        @assert isa(α, dtype) && isa(A, dtype) && isa(B, dtype)
    end

    if α == 0
        return A
    elseif vari == "X"
        return sqrt(dt(2)) * α^dt(3/2 + k) * abs(B) / (sqrt(Pi * dt(2k + 1)) * dt((2l + 1) * l * (l+1))) * 
            (dt(l + 1) / sqrt(dt(2l - 1)) + dt(l) / sqrt(dt(2l + 3)) ) + dt(prod(l-k:l-1)) * abs(A) * (dt("1.1") ^ dt(l-k))
    elseif vari == "Y"
        return sqrt(dt(2)) * α^dt(3/2 + k) * abs(B) / (sqrt(Pi * dt(2k + 1)) * dt(2l + 1)) * 
            (dt(1) / sqrt(dt(2l - 1)) + dt(1) / sqrt(dt(2l + 3))) + dt(prod(l-k:l)) * abs(A) * (dt("1.1") ^ dt(l-k))
    else
        return sqrt(dt(2) / (Pi * dt((2k + 1) * (2l + 1)))) * α^dt(k)
    end
end

"""
Rigorous W^{k,∞} bounds for the angular spherical-harmonic basis on θ ∈ [0, π/2].
This routine returns an explicit upper bound for the W^{k,∞} norms of the angular 
basis functions. Concretely, it produces a table B such that for each degree 
0 ≤ l ≤ l and each derivative order 0 ≤ m ≤ k+1,

    ‖∂_θ^m Φ_l(θ)‖_{L^∞(0,π/2)} ≤ B[l+1, m+1] := = l^m,

where Φ_l denotes the θ-part spherical-harmonic basis used in this code.

# Inputs
- `l::Int` : maximum degree index l to bound (l = 0,1,...,l)
- `k::Int` : derivative order (bounds returned for m = 0,1,...,k+1)

# Returns
- `B::Matrix{dtype}` : a (l+1) × (k+2) matrix of rigorous bounds.
  Row `l+1` corresponds to degree l; column `m+1` corresponds to derivative order m.
"""
function Uθ_bound(l::Int, k::Int)
    l_lst = reshape(collect(0:l), :, 1)
    k_lst = reshape(collect(0:k+1), 1, :)
    return l_lst.^(k_lst)
end

"""
Perform the remaining *radial* integration after the θ-integration step. This 
routine takes the θ-integrated coefficient tensors (returned by `integrate_θ`)
and evaluates the r-integrals on the radial grid `r` using a Newton–Cotes rule,
including the r² weight. It assembles the full Gram matrix by contracting the 
vector components (Y,X,Z) against the 3×3 θ-integrated tensor Q_{ij} (i,j=1..3).
Alongside each quadrature value, we output a guaranteed error bound.

# Inputs

- `int_θ_lst::Vector`: Length-2 list of θ-integrated tensors for the two parity groups.
    Each tensor has size (N_r, Lh, Lh, 3, 3), where:
    - `N_r` is the number of r grid points (same as `length(r)` or compatible with it),
    - `Lh` is the number of spherical θ-basis functions per parity block,
    - the last two indices `(i,j)` correspond to matrix components Q_{ij}.
    Convention: `int_θ_lst[1]` is the “even” group and `int_θ_lst[2]` is the “odd”
    group, consistent with `split_tensor`.

- `Ur_Y_lst::Vector`, `Ur_X_lst::Vector`, `Ur_Z_lst::Vector`:
    Radial basis evaluations grouped by spherical degree index `l`:
    - `Ur_*_lst[l]` is an `N_r × n_l` matrix, each column being one radial basis function.
    - Y and X share the same parameter list structure and hence the same grouping.

- `Y_bound_lst`, `X_bound_lst`, `Z_bound_lst`:
    W^{m,∞} bounds for the corresponding radial bases, in the same block/column layout
    as `Ur_*_lst`. Each `*_bound_lst[l]` is a `(k+1) × n_l` matrix, where row `m+1`
    stores the bound for derivative order `m` (m = 0..k).

- `r::Vector{dtype}`:
    Uniform radial grid. This routine uses `h = r[2] - r[1]` as the grid spacing and
    includes the r² weight via `r_sq = r.^2`.

- `M::Int`:
    Trigonometric frequency parameter used to build W^{k,∞} bounds for the θ-integrated
    tensors via `int_θ_bound`.

# Basis ordering and parity conventions
For each parity group (even / odd), the resulting Gram matrix is ordered as:
1. all (Y,X)-family basis functions first (i.e., columns of `Ur_Y_lst` / `Ur_X_lst`),
2. followed by all Z-family basis functions (columns of `Ur_Z_lst`),
with bases grouped by degree `l` and concatenated in ascending `l`. Within each `l`,
the original column order of `Ur_*_lst[l]` is preserved.

# Outputs

Returns a 4-tuple:
- `res_even`, `res_odd`: Gram matrices for the two parity groups. Each is a square 
    matrix of size (N_Y + N_Z) × (N_Y + N_Z) for that parity, where `N_Y` and `N_Z` 
    are the total numbers of (Y/X)- and Z-basis functions assigned to that parity group.

- `err_even`, `err_odd`: Entrywise rigorous error bounds for `res_even` and `res_odd`, 
    with the same sizes. Each `err_*[p,q]` bounds the quadrature error of `res_*[p,q]`.
"""
function integrate_r(int_θ_lst::Vector, Ur_Y_lst::Vector, Ur_X_lst::Vector, Ur_Z_lst::Vector, 
    Y_bound_lst::Vector, X_bound_lst::Vector, Z_bound_lst::Vector, r::Vector{dtype}, M::Int)
    k = 8
    h = r[2] - r[1]
    r_sq = r .^ 2
    tmp = similar(r_sq)

    # Wk∞ bound for r^2 on [-10*h, 1+10*h]
    r_sq_bound = zeros(dtype, k + 1)
    r_sq_bound[1] = dt("1.1")
    r_sq_bound[2] = dt("2.2")
    r_sq_bound[3] = dt("2.2")

    # count total Y/Z-block sizes by parity
    NY_lst = [0, 0]   # counters for Y-block sizes: [even, odd] (this order is used throughout)
    NZ_lst = [0, 0]
    LY = length(Ur_Y_lst)
    LZ = length(Ur_Z_lst)
    for i in 1:LY
        ind = mod(i, 2) + 1  # index convention for Y: 1 → index 2 → odd, 2 → index 1 → even (this convention is used throughout)
        NY_lst[ind] += size(Ur_Y_lst[i], 2)
    end
    for i in 1:LZ
        ind = mod(i+1, 2) + 1 # index convention for Z: 2 → index 2 → odd, 1 → index 1 → even (this convention is used throughout)
        # See the comments in get_lap_matrix for why the parity grouping is swapped (Y_even pairs with Z_odd, and vice versa).
        NZ_lst[ind] += size(Ur_Z_lst[i], 2)
    end

    # allocate result: two blocks (even, odd)
    result_lst = [zeros(dtype, NY_lst[1] + NZ_lst[1], NY_lst[1] + NZ_lst[1]),
                  zeros(dtype, NY_lst[2] + NZ_lst[2], NY_lst[2] + NZ_lst[2])]
    err_lst = [zeros(dtype, NY_lst[1] + NZ_lst[1], NY_lst[1] + NZ_lst[1]),
                  zeros(dtype, NY_lst[2] + NZ_lst[2], NY_lst[2] + NZ_lst[2])]
    Q_bound_lst = int_θ_bound.(int_θ_lst, Ref(h), Ref(k), Ref(M))

    # track current row/column offsets when filling blocks (used consistently throughout)
    cur_i_lst = [0, 0]  

    # part1: (Y,X)–(Y,X) contributions
    for i in 1:LY
        ind = mod(i, 2) + 1
        cur_i = cur_i_lst[ind]
        Li = size(Ur_Y_lst[i], 2) # number of basis functions in block i

        # corresponding index for θ-integration Gram matrix; since θ-integration results
        # are grouped by boundary condition, this index is only half of the Ur
        ih = (i+1)÷2
        cur_mat = int_θ_lst[ind] # θ-integrated Gram matrix for this parity (even/odd selected by ind)
        cur_bound = Q_bound_lst[ind]
        cur_j = cur_i

        # loop over blocks with the same parity as i (step 2 ensures parity match), starting from j=i
        for j in i:2:LY 
            Lj = size(Ur_Y_lst[j], 2)

            # corresponding index in θ-integration tensor, same logic as ih
            jh = (j+1)÷2 
            # println(size(cur_mat), size(r_sq))
            r_sq_Q11 = r_sq .* cur_mat[k+1:end-k, ih, jh, 1, 1] # line 50
            r_sq_Q12 = r_sq .* cur_mat[k+1:end-k, ih, jh, 1, 2]
            r_sq_Q21 = r_sq .* cur_mat[k+1:end-k, ih, jh, 2, 1]
            r_sq_Q22 = r_sq .* cur_mat[k+1:end-k, ih, jh, 2, 2]
            println("Integrating r-blocks (Y,X) $i and $j")

            # assemble block (i,j) contributions: (Y,X)–(Y,X)
            for ii in 1:Li, jj in 1:Lj
                @views begin
                    yi = Ur_Y_lst[i][:, ii]
                    xi = Ur_X_lst[i][:, ii]
                    yj = Ur_Y_lst[j][:, jj]
                    xj = Ur_X_lst[j][:, jj]

                    b_yi = Y_bound_lst[i][:, ii]
                    b_xi = X_bound_lst[i][:, ii]
                    b_yj = Y_bound_lst[j][:, jj]
                    b_xj = X_bound_lst[j][:, jj]
                end

                # pt1
                @. tmp = yi * yj * r_sq_Q11
                pt1, err1 = newton_cotes_with_bound(tmp, multinomial_sum(k, r_sq_bound, b_yi, cur_bound[:, ih, jh, 1, 1], b_yj))

                # pt2
                @. tmp = xi * yj * r_sq_Q21
                pt2, err2 = newton_cotes_with_bound(tmp, multinomial_sum(k, r_sq_bound, b_xi, cur_bound[:, ih, jh, 2, 1], b_yj))

                # pt3
                @. tmp = yi * xj * r_sq_Q12
                pt3, err3 = newton_cotes_with_bound(tmp, multinomial_sum(k, r_sq_bound, b_yi, cur_bound[:, ih, jh, 1, 2], b_xj))

                # pt4
                @. tmp = xi * xj * r_sq_Q22
                pt4, err4 = newton_cotes_with_bound(tmp, multinomial_sum(k, r_sq_bound, b_xi, cur_bound[:, ih, jh, 2, 2], b_xj))

                result_lst[ind][cur_i+ii, cur_j+jj] = pt1 + pt2 + pt3 + pt4
                err_lst[ind][cur_i+ii, cur_j+jj] = err1 + err2 + err3 + err4
            end

            # fill symmetric block (j,i) by transpose
            if j > i
                result_lst[ind][cur_j+1: cur_j+Lj, cur_i+1: cur_i+Li] =
                    result_lst[ind][cur_i+1: cur_i+Li, cur_j+1: cur_j+Lj]'
            end

            # advance column offset by the size of block j
            cur_j += Lj 
        end
        cur_i_lst[ind] += Li
    end

    # Check: all basis functions have been consumed
    @assert cur_i_lst == NY_lst "cur_i_lst = $cur_i_lst, NY_lst = $NY_lst"

    cur_i_lst = [0, 0]

    # part2: Z–Z contributions, similar as above
    for i in 1:LZ
        ind = mod(i+1, 2) + 1
        cur_i = cur_i_lst[ind]
        Li = size(Ur_Z_lst[i], 2)
        ih = (i+1)÷2
        cur_mat = int_θ_lst[ind]
        cur_bound = Q_bound_lst[ind]
        cur_j = cur_i
        NY = NY_lst[ind]
        for j in i:2:LZ
            Lj = size(Ur_Z_lst[j], 2)
            jh = (j+1)÷2
            r_sq_Q33 = r_sq .* cur_mat[k+1:end-k, ih, jh, 3, 3]
            # assemble block (i,j) contributions: Z–Z
            for ii in 1:Li, jj in 1:Lj
                @views @. tmp = Ur_Z_lst[i][:, ii] * Ur_Z_lst[j][:, jj] * r_sq_Q33
                pt, err = newton_cotes_with_bound(tmp,
                    multinomial_sum(k, r_sq_bound, Z_bound_lst[i][:, ii], cur_bound[:, ih, jh, 3, 3], Z_bound_lst[j][:, jj]))
                result_lst[ind][NY+cur_i+ii, NY+cur_j+jj] = pt
                err_lst[ind][NY+cur_i+ii, NY+cur_j+jj] = err
            end
            if j > i
                result_lst[ind][NY+ cur_j+1: NY+ cur_j+Lj, NY+ cur_i+1: NY+ cur_i+Li] =
                    result_lst[ind][NY+ cur_i+1: NY+ cur_i+Li, NY+ cur_j+1: NY+ cur_j+Lj]'
            end
            cur_j += Lj
        end
        cur_i_lst[ind] += Li
    end
    @assert cur_i_lst == NZ_lst "cur_i_lst = $cur_i_lst, NZ_lst = $NZ_lst"

    # part3: (Y, X)–Z cross terms, similar as above
    cur_i_lst = [0, 0]
    for i in 1:LY
        ind = mod(i, 2) + 1
        cur_i = cur_i_lst[ind]
        Li = size(Ur_Y_lst[i], 2)
        ih = (i+1)÷2
        cur_mat = int_θ_lst[ind]
        cur_bound = Q_bound_lst[ind]
        cur_j = 0
        NY = NY_lst[ind]
        for j in ind:2:LZ
            Lj = size(Ur_Z_lst[j], 2)
            jh = (j+1)÷2
            r_sq_Q13 = r_sq .* cur_mat[k+1:end-k, ih, jh, 1, 3]
            r_sq_Q23 = r_sq .* cur_mat[k+1:end-k, ih, jh, 2, 3]
            # assemble block (i,j) contributions: (Y,X)–Z
            for ii in 1:Li, jj in 1:Lj
                @views @. tmp = Ur_Y_lst[i][:, ii] * Ur_Z_lst[j][:, jj] * r_sq_Q13
                pt1, err1 = newton_cotes_with_bound(tmp,
                    multinomial_sum(k, r_sq_bound, Y_bound_lst[i][:, ii], cur_bound[:, ih, jh, 1, 3], Z_bound_lst[j][:, jj]))

                @views @. tmp = Ur_X_lst[i][:, ii] * Ur_Z_lst[j][:, jj] * r_sq_Q23
                pt2, err2 = newton_cotes_with_bound(tmp,
                    multinomial_sum(k, r_sq_bound, X_bound_lst[i][:, ii], cur_bound[:, ih, jh, 2, 3], Z_bound_lst[j][:, jj]))

                result_lst[ind][cur_i+ii, NY+cur_j+jj] = pt1 + pt2
                err_lst[ind][cur_i+ii, NY+cur_j+jj] = err1 + err2
            end
            result_lst[ind][NY+cur_j+1: NY+cur_j+Lj, cur_i+1: cur_i+Li] =
                result_lst[ind][cur_i+1: cur_i+Li, NY+cur_j+1: NY+cur_j+Lj]'
            cur_j += Lj
        end
        cur_i_lst[ind] += Li
    end
    @assert cur_i_lst == NY_lst "cur_i_lst = $cur_i_lst, NY_lst = $NY_lst"
    return result_lst[1], result_lst[2], err_lst[1], err_lst[2]
end

"""
Evaluate diagonal L² inner products (with r² weight) of radial bases for orthogonality checks.
For each degree l, this routine computes per-column integrals using Newton Cotes quadrature
with rigorous bounds:
- Y-block: (l(l+1)⟨X_i,X_i⟩ + ⟨Y_i,Y_i⟩) / (2l+1)
- Z-block:  l(l+1)⟨Z_i,Z_i⟩ / (2l+1)

# Input:
- Ur_Y_lst, Ur_X_lst, Ur_Z_lst: radial basis values grouped by l (each is N_r × n(l))
- Y_bound_lst, X_bound_lst, Z_bound_lst: corresponding bounds for newton_cotes_with_bound
- r: radial grid (r² weight is included)

# Return:
- Y_int_lst, Z_int_lst: vectors of per-column diagonal inner products for each l
- Y_err_lst, Z_err_lst: corresponding error bounds
"""
function eval_l2_inpd(Ur_Y_lst::Vector, Ur_X_lst::Vector, Ur_Z_lst::Vector, 
    Y_bound_lst::Vector, X_bound_lst::Vector, Z_bound_lst::Vector, r::Vector{dtype})
    Y_int_lst = []
    Z_int_lst = []
    Y_err_lst = []
    Z_err_lst = []
    r_sq = r .^ 2

    for l in eachindex(Ur_Y_lst)
        # compute only diagonal entries: (⟨Xᵢ,Xᵢ⟩·l(l+1) + ⟨Yᵢ,Yᵢ⟩) / (2l+1)
        fX = Ur_X_lst[l] .* Ur_X_lst[l] .* r_sq
        fY = Ur_Y_lst[l] .* Ur_Y_lst[l] .* r_sq

        res_Y = newton_cotes_with_bound.(eachcol(fX), X_bound_lst[l])
        Y_int1 = first.(res_Y)
        Y_err1 = last.(res_Y)

        res_Y = newton_cotes_with_bound.(eachcol(fY), Y_bound_lst[l])
        Y_int2 = first.(res_Y)
        Y_err2 = last.(res_Y)

        Y_int = (Y_int1 .* dt(l*(l+1)) .+ Y_int2) ./ dt(2l+1)
        Y_err = (Y_err1 .* dt(l*(l+1)) .+ Y_err2) ./ dt(2l+1)

        push!(Y_int_lst, Y_int)
        push!(Y_err_lst, Y_err)
    end
    for l in eachindex(Ur_Z_lst)
        fZ = Ur_Z_lst[l] .* Ur_Z_lst[l] .* r_sq

        res_Z = newton_cotes_with_bound.(eachcol(fZ), Z_bound_lst[l])
        Z_int = first.(res_Z)
        Z_err = last.(res_Z)

        Z_int .= Z_int .* dt(l*(l+1)) ./ dt(2l+1)
        Z_err .= Z_err .* dt(l*(l+1)) ./ dt(2l+1)

        push!(Z_int_lst, Z_int)
        push!(Z_err_lst, Z_err)
    end
    return Y_int_lst, Z_int_lst, Y_err_lst, Z_err_lst
end

"""
Convert per-mode L² norms into H¹-type norms and split by parity (l even/odd).
For each mode (l,j), `paras[l][j][1]` stores the radial spectral parameter α.
Using the eigenvalue relation in the paper, the H¹ factor is
    (α² / R² + β),
where `R` is the physical radius scaling and `β` is the constant zeroth-order
term appearing in the target H¹/Laplace inner product.

Modes are split into two vectors by the parity of `l`:
- `res_even`: all modes with even `l` concatenated in increasing `l`;
- `res_odd` : all modes with odd  `l` concatenated similarly.

# Inputs
- `norm_vec`: vector-of-vectors of L² norms, grouped by `l`.
- `paras`: parameter lists, grouped by `l`, with α stored at index 1.
- `R`, `β`: scalars used in the H¹ factor.

# Returns
- `(res_even, res_odd)` as two flat vectors aligned with the parity grouping.
"""
function eval_h1_norm(norm_vec::Vector, paras::Vector, R::dtype, β::dtype)
    # count total sizes by parity
    N_lst = [0, 0] # 1 for even, 2 for odd
    L = length(paras)
    for l in 1:L
        N_lst[mod(l, 2) + 1] += length(paras[l])
    end
    result_lst = [zeros(dtype, N_lst[1]), zeros(dtype, N_lst[2])]
    cur_l_lst = [0, 0]

    for l in 1:L
        ind = mod(l, 2) + 1
        for j in 1:length(norm_vec[l])
            # Multiplies the L² norm with the eigenvalue + β
            result_lst[ind][cur_l_lst[ind]+j] = norm_vec[l][j] * (paras[l][j][1]^2/R^2 + β)
        end

        # advance column offset by the size of block l
        cur_l_lst[ind] += length(norm_vec[l]) 
    end
    return result_lst[1], result_lst[2]
end

"""
Build W^{k,∞} product bounds for the weighted square `r² · U(r)²` from bounds of `U(r)`.
Given a per-degree list of W^{k,∞} bounds for the radial basis `U` (each column
corresponds to one basis function), this helper constructs bounds suitable for
integrands of the form `r² * U_i(r) * U_i(r)` by combining:
- a fixed bound table for `r²`, and
- the W^{k,∞} bounds for `U`.

This is used to feed `multinomial_sum` / `newton_cotes_with_bound` in subsequent
radial quadrature routines.

# Inputs
- `U_bound_lst`: list of bound tables grouped by degree.
- `k`: maximum derivative order used by the multinomial product bound.

# Returns
- A list (grouped by degree) of per-column bounds for `r² · U(r)²`.
"""
function Ur_sq_Wk∞_bound_lists(bound_lst::Vector; k::Int=8)
    # Wk∞ bound for r^2 on [-10*h, 1+10*h]
    r_sq_bound = zeros(dtype, k + 1)
    r_sq_bound[1] = dt("1.1")
    r_sq_bound[2] = dt("2.2")
    r_sq_bound[3] = dt("2.2")

    sq_bound_lst = []
    for each_lst in bound_lst
        tmp_lst = zeros(dtype, size(each_lst, 2))
        @views for j in axes(each_lst, 2)
            tmp_lst[j] = multinomial_sum(k, r_sq_bound, each_lst[:, j], each_lst[:, j])
        end
        push!(sq_bound_lst, tmp_lst)
    end
    return sq_bound_lst
end

"""
Assemble diagonal Gram matrices for the Laplace/H¹ inner product.
This routine evaluates per-mode H¹ norms of the Y/X/Z bases and packs them into
two diagonal matrices according to parity grouping.

# Input:
- Ur_Y_lst, Ur_X_lst, Ur_Z_lst: radial basis values for Y-, X-, and Z-types
- Y_bound_lst, X_bound_lst, Z_bound_lst: corresponding W^{k,∞} bounds for error control
- Y_paras, Z_paras: parameters where paras[l][j][1] provides α for mode (l,j)
- r: radial grid
- R, β: parameters in the H¹ factor (α²/R² + β)

# Return:
- B_even, B_odd: diagonal Gram matrices for even/odd groups
- B_even_err, B_odd_err: diagonal matrices of corresponding error bounds
"""
function get_lap_matrix(Ur_Y_lst::Vector, Ur_X_lst::Vector, Ur_Z_lst::Vector,
    Y_bound_lst::Vector, X_bound_lst::Vector, Z_bound_lst::Vector,
    Y_paras::Vector, Z_paras::Vector, r::Vector{dtype}, R::dtype, β::dtype)
    # Evaluate L² norms of radial basis functions.
    Y_bound_lst = Ur_sq_Wk∞_bound_lists(Y_bound_lst)
    X_bound_lst = Ur_sq_Wk∞_bound_lists(X_bound_lst)
    Z_bound_lst = Ur_sq_Wk∞_bound_lists(Z_bound_lst);
    Y_int_lst, Z_int_lst, Y_err_lst, Z_err_lst = eval_l2_inpd(Ur_Y_lst, Ur_X_lst, Ur_Z_lst, Y_bound_lst, X_bound_lst, Z_bound_lst, r)
    println(Y_int_lst)
    println(Z_int_lst)
    println(Y_err_lst)
    println(Z_err_lst)

    # Compute H¹ norms of radial basis functions.
    Y_norm_even, Y_norm_odd = eval_h1_norm(Y_int_lst, Y_paras, R, β)
    Z_norm_even, Z_norm_odd = eval_h1_norm(Z_int_lst, Z_paras, R, β)
    Y_err_even, Y_err_odd = eval_h1_norm(Y_err_lst, Y_paras, R, β)
    Z_err_even, Z_err_odd = eval_h1_norm(Z_err_lst, Z_paras, R, β)

    # Assemble combined norm vectors and form diagonal matrices (Gram matrices under Laplace inner product).
    # Parity convention (field-level):
    # Here "even/odd" refers to the parity of the *vector field* w.r.t. θ = π/2 (imposed by bdry).
    # Due to the different θ-boundary conditions used by Y- and Z-type bases, their coefficient
    # blocks are parity-shifted: Y-even pairs with Z-odd to form the field-even subspace, and
    # Y-odd pairs with Z-even to form the field-odd subspace. Therefore we pack as (Y_even, Z_odd)
    # and (Y_odd, Z_even) consistently throughout the codebase.
    norm_vec_even = [Y_norm_even; Z_norm_odd]
    norm_vec_odd = [Y_norm_odd; Z_norm_even]
    norm_err_even = [Y_err_even; Z_err_odd]
    norm_err_odd = [Y_err_odd; Z_err_even]

    B_even = Matrix(Diagonal(norm_vec_even))
    B_odd = Matrix(Diagonal(norm_vec_odd))
    B_even_err = Matrix(Diagonal(norm_err_even))
    B_odd_err = Matrix(Diagonal(norm_err_odd))
    return B_even, B_odd, B_even_err, B_odd_err
end

"""
Generate W^{k,∞}-style radial bound tables for Y/X/Z bases from parameter lists.
For each degree index i and each basis j within that degree, this routine evaluates
Ur_bound(paras, vari, i, m) for m = 0..k and stores the results column-wise.
# Input:
- Y_paras: parameter lists for Y/X-type bases, grouped by degree i
- Z_paras: parameter lists for Z-type bases, grouped by degree i
- k: maximum derivative/order index (computes m = 0..k)
# Return:
- Y_bd_lst, X_bd_lst, Z_bd_lst: lists of matrices (k+1)×n_basis(i), grouped by degree
"""
function Ur_bound_lst(Y_paras::Vector, Z_paras::Vector, k::Int)
    LY = length(Y_paras)
    LZ = length(Z_paras)
    Y_bd_lst = []
    X_bd_lst = []
    Z_bd_lst = []

    # part1: (Y, X) contributions
    for i in 1:LY
        Li = length(Y_paras[i]) # number of basis functions in block i
        cur_r_bd_Y = zeros(dtype, k+1, Li)
        cur_r_bd_X = zeros(dtype, k+1, Li)
        for j in 1:Li
            for m in 0:k
                cur_r_bd_Y[m+1, j] = Ur_bound(Y_paras[i][j], "Y", i, m)
                cur_r_bd_X[m+1, j] = Ur_bound(Y_paras[i][j], "X", i, m)
            end
        end
        push!(Y_bd_lst, cur_r_bd_Y)
        push!(X_bd_lst, cur_r_bd_X)
    end

    # part2: Z contributions, similar as above
    for i in 1:LZ
        Li = length(Z_paras[i])
        cur_r_bd = zeros(dtype, k+1, Li)
        for j in 1:Li
            for m in 0:k
                cur_r_bd[m+1, j] = Ur_bound(Z_paras[i][j], "Z", i, m)
            end
        end
        push!(Z_bd_lst, cur_r_bd)
    end
    return Y_bd_lst, X_bd_lst, Z_bd_lst
end

"""
Pack per-degree radial W^{k,∞} bound tables into parity-grouped tensors. It takes 
the per-degree bound blocks produced by `Ur_bound_lst` and concatenates them into
two tensors corresponding to the two parity groups.

# Input layout
- `Y_bound_lst[i]`, `X_bound_lst[i]`, `Z_bound_lst[i]` are (k+1)×n_i matrices,
  where row `m+1` stores the W^{m,∞} bound (m = 0..k) for each basis column,
  and `i` indexes the spherical degree block.

# Output layout
Returns `(U_even, U_odd)` where each has shape (k+1, N_parity, 3).

- First dimension (rows): derivative order m = 0..k.
- Second dimension (columns): basis functions concatenated across degrees `i`
  within the corresponding parity group, in the same order expected by the
  eigenvector coefficient matrix passed to `eig_fun_bound`.
- Third dimension: three channels storing bounds for the three radial families:
    1. channel 1: Y-family radial bounds;
    2. channel 2: X-family radial bounds, multiplied by a bound for Uθ 
        (derivative of spherical harmonics);
    3. channel 3: Z-family radial bounds, multiplied by the same bound above.
"""
function lst_to_mat(Y_bound_lst, X_bound_lst, Z_bound_lst)
    # count total Y/Z-block sizes by parity
    NY_lst = [0, 0]   # counters for Y-block sizes: [even, odd] (this order is used throughout)
    NZ_lst = [0, 0]
    LY = length(Y_bound_lst)
    LZ = length(Z_bound_lst)
    k = size(Y_bound_lst[1], 1)
    for i in 1:LY
        ind = mod(i, 2) + 1  # index convention for Y: 1 → odd, 2 → even (this convention is used throughout)
        NY_lst[ind] += size(Y_bound_lst[i], 2)
    end
    for i in 1:LZ
        ind = mod(i+1, 2) + 1 # index convention for Z: 2 → odd, 1 → even (this convention is used throughout)
        # See the comments in get_lap_matrix for why the parity grouping is swapped (Y_even pairs with Z_odd, and vice versa).
        NZ_lst[ind] += size(Z_bound_lst[i], 2)
    end

    # allocate result: two blocks (even, odd)
    result_lst = [zeros(dtype, k, NY_lst[1] + NZ_lst[1], 3),
                  zeros(dtype, k, NY_lst[2] + NZ_lst[2], 3)]

    # track current row/column offsets when filling blocks (used consistently throughout)
    cur_i_lst = [0, 0]  

    l = max(length(Y_bound_lst), length(Z_bound_lst))
    Uθ_bound_lst = dt.(Uθ_bound(l, k))

    # part1: (Y, X) contributions
    for i in 1:LY
        ind = mod(i, 2) + 1
        cur_i = cur_i_lst[ind]
        Li = size(Y_bound_lst[i], 2) # number of basis functions in block i

        result_lst[ind][:, cur_i+1: cur_i+Li, 1] = Y_bound_lst[i]
        result_lst[ind][:, cur_i+1: cur_i+Li, 2] = X_bound_lst[i] * Uθ_bound_lst[i, 2]
        cur_i_lst[ind] += Li
    end

    # Check: all basis functions have been consumed
    @assert cur_i_lst == NY_lst "cur_i_lst = $cur_i_lst, NY_lst = $NY_lst"

    cur_i_lst = [0, 0]

    # part2: Z contributions, similar as above
    for i in 1:LZ
        ind = mod(i+1, 2) + 1
        cur_i = cur_i_lst[ind]
        Li = size(Z_bound_lst[i], 2)

        result_lst[ind][:, cur_i+1: cur_i+Li, 3] = Z_bound_lst[i] * Uθ_bound_lst[i, 2]
        cur_i_lst[ind] += Li
    end
    @assert cur_i_lst == NZ_lst "cur_i_lst = $cur_i_lst, NZ_lst = $NZ_lst"
    return result_lst[1], result_lst[2]
end