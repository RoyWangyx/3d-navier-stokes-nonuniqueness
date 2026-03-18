"""
Constructs a separated-variable operator of the form
`op = Operator(U_left, U_right')`, with an optional alternative left matrix `L1`
for the first column block when two radial boundary conditions are provided.

- The radial part `U_left` is built as a linear combination of basis functions
  `trig_func(M0, r, key_r, bd_r)` with coefficients `α` from `key_r_lst` (where
  `α` is a **scaling factor** multiplying each radial basis term).
- The angular part `U_right` is built by `trig_func(M1, θ, key_θ, bd_θ)` and then
  transposed for right multiplication.
- If the radial boundary specification includes two parts (`bd_r1` and `bd_r`),
  an additional `L1` operator is constructed (using `key_r_lst1` or, if not
  provided, `key_r_lst`) and set with `set_L1!`.

Inputs:
- `M0::Int`: number of radial modes (columns).
- `M1::Int`: number of angular modes (columns).
- `r::Vector{dtype}`: radial grid (must be a column vector, checked by
  `@assert size(r,2)==1`).
- `θ::Vector{dtype}`: angular grid (must be a column vector).
- `bdry::Vector`: boundary condition specification `[bd0, bd_θ]`.
  - `bd0`: either a `String` (single radial boundary) or a 2-element
    `Vector{String}` giving `[bd_r1, bd_r]`.
  - `bd_θ::String`: angular boundary condition.
- `key_r_lst::Vector`: list of `(key_r::String, α::Union{dtype,Int})` pairs,
  where `α` is a **scaling factor** for the radial basis function specified by `key_r`.
- `key_θ::String`: key specifying the angular basis.
- `key_r_lst1` (keyword, optional): alternative list of `(key_r, α)` pairs used
  to construct `L1` when two radial boundaries are given. Defaults to `key_r_lst`.

Returns:
- `op::Operator`: operator object containing left multiplication matrix `U_left`,
  right multiplication matrix `U_right'`, and optional `L1`.
"""
function build_operator(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype}, bdry::Vector, key_r_lst::Vector,
    key_θ::String; key_r_lst1::Union{Vector, Nothing}=nothing)
    @assert length(bdry)==2 "bdry must be length 2"
    bd0, bd_θ = bdry
    bd_r1, bd_r = (isa(bd0, String) ? (nothing, bd0) : bd0)
    @assert size(r,2)==1 && size(θ,2)==1 "r and θ must be column vectors"

    # construct left multiplication matrix
    U_left = zeros(dtype, length(r), M0)
    for (key_r, alpha) in key_r_lst
        @assert isa(alpha, Union{dtype, Int}) "alpha must be a scalar"
        @assert isa(key_r, String) "key_r must be a String"
        U_left .+= alpha * trig_func(M0, r, key_r, bd_r)
    end

    # construct right multiplication matrix
    U_right = trig_func(M1, θ, key_θ, bd_θ)
    op = Operator(U_left, U_right')

    # construct L1 if two radial boundaries are given
    if bd_r1 !== nothing # If bd_r1 is nothing, even if key_r_lst1 is provided, it is ignored
        # use key_r_lst if key_r_lst1 is not provided
        if key_r_lst1 === nothing
            key_r_lst1 = key_r_lst
        end
        # construct and set L1
        U_left1 = zeros(dtype, length(r), M0)
        for (key_r1, alpha1) in key_r_lst1
            @assert isa(alpha1, Union{dtype, Int}) "alpha1 must be a scalar"
            @assert isa(key_r1, String) "key_r1 must be a String"
            U_left1 .+= alpha1 * trig_func(M0, r, key_r1, bd_r1)
        end
        set_L1!(op, U_left1)
    end

    return op
end

"""
Construct linear operator matrix for the system:
    L(U, P) = - (1/2) U - (1/2) x · ∇U - α ΔU - β ∇P,
where
- U is the velocity-like vector field,
- P is the scalar pressure-like field,
- α, β = scl_fac^(-2), scl_fac^(-1) are the diffusion/pressure scaling factors, and weighted 
by sin(r)/cos(r)^2 due to change of variable. Refer to the paper for the derivation.

# Arguments
- `M0::Int`       : Maximum degree of the trigonometric polynomial basis in the radial variable r.
- `M1::Int`       : Maximum degree of the trigonometric polynomial basis in the angular variable θ.
- `r::Vector{dtype}` : Radial grid points (must be a column vector).
- `θ::Vector{dtype}` : Angular grid points (must be a column vector).
- `scl_fac::dtype`   : Scaling factor, used to compute α = scl_fac^(-2), β = scl_fac^(-1).
- `bdry_index::Int`  : Index to select boundary condition set from `BDRY_LST_SET`.
- `transpose::Bool=false`: if `true`, switch the drift term to an adjoint-like form (using `Scale_T`)
    and omit the pressure blocks (columns for `P`).

# Returns
- `op_mat::Matrix{Vector}` : A 3×4 block operator matrix (3 velocity-like outputs, 3 velocity-like inputs
    + 1 pressure input). Each block `(i,j)` stores a list of `Operator`s corresponding to terms of L(U,P).
"""
function get_linear(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype},
                    scl_fac::dtype, bdry_index::Int; transpose::Bool=false)
    op_mat  = Operator_matrix(3,4)
    bdry_r, bdry_θ, bdry_φ, bdry_p = BDRY_LST_SET[bdry_index]

    # Δ is self-adjoint. The exact adjoint of -(1/2)U -(1/2)x·∇U
    # is U + (1/2)x·∇U. For convenience we use(1/2)U + (1/2)x·∇U,
    # and later projection ⟂U makes the difference irrelevant.
    scale_key = transpose ? "Scale_T" : "Scale"

    alpha = scl_fac^dt(-2) # diffusion scaling factor. order -2 because of two derivatives
    M1p     = (bdry_index > 1) ? M1+1 : M1 
    # For bdry_index 2 or 3, u1 has one more (constant) mode in θ due to Neumann BC on both ends

    # Build operators in parallel for efficiency, as they can be time-consuming to construct
    t11a = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["Cot", alpha]], "Laplacian_r_θ";
                                 key_r_lst1=[["zero", zero(dtype)]])

    t11b = @spawn build_operator(M0,   M1p, r, θ, bdry_r,
                                 [["Laplacian_r_r", alpha], [scale_key, one(dtype)]], "I";
                                 key_r_lst1=[["Laplacian_r_odd", alpha], [scale_key, one(dtype)]])

    t12a = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["Cot", alpha]], "Laplacian_θ_r_r";
                                 key_r_lst1=[["zero", zero(dtype)]])

    t22a = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["Cot", alpha]], "Laplacian_θ_θ";
                                 key_r_lst1=[["zero", zero(dtype)]])

    t22b = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ,
                                 [["Laplacian_θ_r", alpha], [scale_key, one(dtype)]], "I";
                                 key_r_lst1=[["Laplacian_θ_odd", alpha], [scale_key, one(dtype)]])

    t21a = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["Cot", dt(2)*alpha]], "D";
                                 key_r_lst1=[["Laplacian_inter_odd", dt(2)*alpha]])

    t33a = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["Cot", alpha]], "Laplacian_ϕ_θ")

    t33b = @spawn build_operator(M0,   M1,  r, θ, bdry_φ,
                                 [["Laplacian_ϕ_r", alpha], [scale_key, one(dtype)]], "I")
    
    # Pressure terms (discarded if transpose=true)
    if !transpose
        beta = inv(scl_fac)# pressure scaling factor. order -1 because of only one derivative

        t14a = @spawn build_operator(M0, M1, r, θ, bdry_p, [["D_p_r", beta]], "I")

        t24a = @spawn build_operator(M0, M1, r, θ, bdry_p, [["I", beta]], "D")

        op_mat[1,4] = [fetch(t14a)]
        op_mat[2,4] = [fetch(t24a)]
    end

    # assemble
    op_mat[1,1] = [fetch(t11a), fetch(t11b)]
    op_mat[1,2] = [fetch(t12a)]

    op_mat[2,2] = [fetch(t22a), fetch(t22b)]
    op_mat[2,1] = [fetch(t21a)]

    op_mat[3,3] = [fetch(t33a), fetch(t33b)]

    return op_mat
end

"""
Build the identity operator matrix in spherical coordinates.
# Inputs
- `M0::Int, M1::Int` : degrees of triangular polynomials in r and θ
- `r::Vector{dtype}` : radial grid (column vector)
- `θ::Vector{dtype}` : angular grid (column vector)
- `bdry_index::Int`  : index selecting boundary condition set
- `weight_flag::Bool=true` : chooses weighting:
    • false → use "Cos", canceling the built-in 1/cos(r) weight
    • true  → use "Tan", corresponding to the weight sin(r)/cos²(r)
    Refers to the paper for details.
# Returns
- `op_mat` : a 3×4 `Operator_matrix` with diagonal blocks set to identity
  operators in r, θ, and φ directions, adjusted for boundary conditions and
  the chosen weight.
"""
function get_identity(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype},
                      bdry_index::Int; weight_flag::Bool=true)
    @assert size(r, 2) == 1 "r must be a column vector"
    @assert size(θ, 2) == 1 "θ must be a column vector"

    op_mat = Operator_matrix(3, 4)
    bdry_r, bdry_θ, bdry_φ, _ = BDRY_LST_SET[bdry_index]
    # For angular boundary condition "11", u₁ includes an extra constant mode in θ
    M1p = (bdry_index > 1) ? M1+1 : M1

    key_r = weight_flag ? "Tan" : "Cos"
    t11 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [[key_r, one(dtype)]], "I")
    t22 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [[key_r, one(dtype)]], "I")
    t33 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [[key_r, one(dtype)]], "I")

    op_mat[1,1] = [fetch(t11)]
    op_mat[2,2] = [fetch(t22)]
    op_mat[3,3] = [fetch(t33)]

    return op_mat
end

"""
Build the symmetric gradient operator matrix in spherical coordinates.
# Inputs
- `M0::Int, M1::Int`  : degrees of triangular polynomials in r and θ
- `r::Vector{dtype}`  : radial grid (column vector)
- `θ::Vector{dtype}`  : angular grid (column vector)
- `scl_fac::dtype`    : scaling factor (α = 1/scl_fac)
- `bdry_index::Int`   : selects boundary-condition set

# Returns
- `op_mat` : 6×4 `Operator_matrix` for the symmetric gradient (strain tensor).  
  There are 6 rows because a 3×3 symmetric tensor has 6 independent components.
"""
function get_gradient_sym(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype},
                          scl_fac::dtype, bdry_index::Int)
    op_mat = Operator_matrix(6, 4)
    bdry_r, bdry_θ, bdry_φ, _ = BDRY_LST_SET[bdry_index]

    # Scaling factor
    alpha  = inv(scl_fac)
    # For angular boundary condition "11", u₁ includes an extra constant mode in θ
    M1p    = (bdry_index > 1) ? M1+1 : M1

    t11 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D11_r_sym", alpha]], "I")

    t21 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D12_r_sym", alpha]], "D")

    t22 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D12_θ_sym", alpha]], "I")

    t33 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D13_ϕ_sym", alpha]], "I")

    t41 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D22_r_sym", alpha]], "I")

    t42 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D22_θ_sym", alpha]], "D")

    t53 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D23_ϕ_sym", alpha]], "D23_ϕ_θ_sym")

    t61 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D33_r_sym", alpha]], "I")

    t62 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D33_θ_sym", alpha]], "Cot")

    op_mat[1,1] = [fetch(t11)]
    op_mat[2,1] = [fetch(t21)]

    op_mat[2,2] = [fetch(t22)]

    op_mat[3,3] = [fetch(t33)]

    op_mat[4,1] = [fetch(t41)]
    op_mat[4,2] = [fetch(t42)]

    op_mat[5,3] = [fetch(t53)]

    op_mat[6,1] = [fetch(t61)]
    op_mat[6,2] = [fetch(t62)]

    return op_mat
end

"""
Build the full gradient operator matrix in spherical coordinates.

# Inputs
- `M0::Int, M1::Int`  : degrees of triangular polynomials in r and θ
- `r::Vector{dtype}`  : radial grid (column vector)
- `θ::Vector{dtype}`  : angular grid (column vector)
- `scl_fac::dtype`    : scaling factor (α = 1/scl_fac)
- `bdry_index::Int`   : selects boundary-condition set
- `weight_flag::Bool=false` : if true, append "_w" to operator keys,
  corresponding to the weight tan(r). We use the gradient with
  weight to calculate the L2 norm of v⋅∇U. Here, v has the weight 1/cos(r).
  Thus, we need ∇U to have a weight tan(r).

# Returns
- `op_mat` : a 9×4 `Operator_matrix` encoding all nine components of the
  3×3 gradient tensor.
"""
function get_gradient(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype}, 
    scl_fac::dtype, bdry_index::Int; weight_flag::Bool=false)
    op_mat = Operator_matrix(9, 4)
    bdry_r, bdry_θ, bdry_φ, _ = BDRY_LST_SET[bdry_index]

    # Scaling factor
    alpha = inv(scl_fac)
    # For angular boundary condition "11", u₁ includes an extra constant mode in θ
    M1p = (bdry_index > 1) ? M1+1 : M1
    # Append "_w" to operator keys if weighted form is requested
    suffix = weight_flag ? "_w" : ""
    k1 = (bdry_index > 1) ? nothing : [["D11_r" * suffix, -alpha/dt(2)]]
    k0 = (bdry_index > 1) ? nothing : [["zero", zero(dtype)]]

    t11 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D11_r" * suffix,  alpha]], "I")

    t22 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D21_θ" * suffix,  alpha]], "I")

    t33 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D31_ϕ" * suffix,  alpha]], "I")

    t41 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D12_r" * suffix,  alpha]], "D"; key_r_lst1 = k1)

    t42 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D12_θ" * suffix, -alpha]], "I"; key_r_lst1 = k0)

    t51 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D22_r" * suffix,  alpha]], "I"; key_r_lst1 = k1)

    t52 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D22_θ" * suffix,  alpha]], "D"; key_r_lst1 = k0)

    t63 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D32_ϕ" * suffix,  alpha]], "D")

    t73 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D13_ϕ" * suffix, -alpha]], "I")

    t83 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D23_ϕ" * suffix, -alpha]], "Cot")

    t91 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D33_r" * suffix,  alpha]], "I";   key_r_lst1 = k1)

    t92 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D33_θ" * suffix,  alpha]], "Cot"; key_r_lst1 = k0)

    op_mat[1,1] = [fetch(t11)]
    op_mat[2,2] = [fetch(t22)]
    op_mat[3,3] = [fetch(t33)]

    op_mat[4,1] = [fetch(t41)]
    op_mat[4,2] = [fetch(t42)]

    op_mat[5,1] = [fetch(t51)]
    op_mat[5,2] = [fetch(t52)]

    op_mat[6,3] = [fetch(t63)]
    op_mat[7,3] = [fetch(t73)]
    op_mat[8,3] = [fetch(t83)]

    op_mat[9,1] = [fetch(t91)]
    op_mat[9,2] = [fetch(t92)]

    return op_mat
end

"""
Construct the operator matrix associated with a given VectorFunc. The operator 
is selected by key ("linear", "identity", "gradient_sym", "gradient") and built
on the grid (r, θ). Boundary conditions and scaling are taken from sol. And the
conversion matrix of sol is composed on the right of the corresponding operators,
so that the resulting operator is compatible with the internal basis used by sol.

# Requirements
- `sol["u3"].freq_func` must be set.
- `sol.scl_fac` and `sol.bdry_index` must be valid.

# Input:
- sol: VectorFunc providing size, scaling factor, and boundary information
- r: radial grid
- θ: angular grid
- `key::String`: operator family:
  - `"linear"`        → `get_linear(...; transpose=...)`
  - `"identity"`      → `get_identity(...; weight_flag=...)`
  - `"gradient_sym"`  → `get_gradient_sym(...)`
  - `"gradient"`      → `get_gradient(...; weight_flag=...)`
- kwargs: Passed through to the underlying constructor:
  - For `"linear"`: `transpose::Bool`
  - For `"identity"`: `weight_flag::Bool`
  - For `"gradient"`: `weight_flag::Bool`

# Return:
- `op_mat::Operator_matrix`: operator grid ready to be used by `apply(op_mat, vec)`.
"""
function get_operator_matrix(sol::VectorFunc, r::Vector{dtype}, θ::Vector{dtype}, key::String; kwargs...)
    M0, M1 = size(sol["u3"].freq_func)
    scl_fac = sol.scl_fac
    bdry_index = sol.bdry_index
    if key == "linear"
        transpose = get(kwargs, :transpose, false)
        op_mat = get_linear(M0, M1, r, θ, scl_fac, bdry_index; transpose=transpose)
    elseif key == "identity"
        weight_flag = get(kwargs, :weight_flag, true)
        op_mat = get_identity(M0, M1, r, θ, bdry_index; weight_flag=weight_flag)
    elseif key == "gradient_sym"
        op_mat = get_gradient_sym(M0, M1, r, θ, scl_fac, bdry_index)
    elseif key == "gradient"
        weight_flag = get(kwargs, :weight_flag, false)
        op_mat = get_gradient(M0, M1, r, θ, scl_fac, bdry_index; weight_flag=weight_flag)
    else
        error("Invalid operator key: $key")
    end
    apply_convmat_right!(op_mat, sol)
    return op_mat
end

"""
Assemble the contraction between A and ∇B in space domain.

# Input:
- A: VectorFunc with keys u1,u2,u3
- grad_B: VectorFunc with keys A11,...,A33
- transpose: use (∇B)ᵀ if true

# Return:
- VectorFunc with keys e1,e2,e3
"""
function mul_A_gradB_local(A::VectorFunc, grad_B::VectorFunc; transpose::Bool=false)
    if transpose
        e1 = A["u1"].space_func .* grad_B["A11"].space_func + 
             A["u2"].space_func .* grad_B["A12"].space_func + 
             A["u3"].space_func .* grad_B["A13"].space_func
        e2 = A["u1"].space_func .* grad_B["A21"].space_func + 
             A["u2"].space_func .* grad_B["A22"].space_func + 
             A["u3"].space_func .* grad_B["A23"].space_func
        e3 = A["u1"].space_func .* grad_B["A31"].space_func + 
             A["u2"].space_func .* grad_B["A32"].space_func + 
             A["u3"].space_func .* grad_B["A33"].space_func
    else
        e1 = A["u1"].space_func .* grad_B["A11"].space_func + 
             A["u2"].space_func .* grad_B["A21"].space_func + 
             A["u3"].space_func .* grad_B["A31"].space_func
        e2 = A["u1"].space_func .* grad_B["A12"].space_func + 
             A["u2"].space_func .* grad_B["A22"].space_func + 
             A["u3"].space_func .* grad_B["A32"].space_func
        e3 = A["u1"].space_func .* grad_B["A13"].space_func + 
             A["u2"].space_func .* grad_B["A23"].space_func + 
             A["u3"].space_func .* grad_B["A33"].space_func
    end
    result = VectorFunc(A.scl_fac, "space"; e1=e1, e2=e2, e3=e3)
    return result
end

"""
Evaluate A · (∇B) or A · (∇B)ᵀ on the grid (r, θ).
This function interpolates A and B onto the grid, computes ∇B, and assembles
the contraction. Supports both single VectorFunc and lists.

# Input:
- A_lst: VectorFunc or list of VectorFunc (keys u1,u2,u3)
- B_lst: VectorFunc or list of VectorFunc (keys u1,u2,u3)
- r: radial grid
- θ: angular grid
- `transpose::Bool=false`:
  - If `false`, compute `A · (∇B)`.
  - If `true`, this option is only supported when both inputs are single `VectorFunc`s. 
    In that case, it returns two results: `[A·(∇B), A·(∇B)ᵀ]` (in this order).

# Returns
- If `A_lst` and `B_lst` are single `VectorFunc` and `transpose=false`:
  - `VectorFunc` with keys `"e1","e2","e3"`.
- If `A_lst` and/or `B_lst` are vectors and `transpose=false`:
  - a matrix `Array{VectorFunc,2}` where entry `(i,j)` corresponds to `A_lst[i] · ∇B_lst[j]`.
- If `transpose=true` (single-single case only):
  - `Vector{VectorFunc}` of length 2, `[A·(∇B), A·(∇B)ᵀ]`.

# Throws
- `AssertionError` if component keys are missing, scaling factors mismatch, boundary
  indices mismatch, or if `transpose=true` is requested for non-single inputs.
"""
function mul_A_gradB(A_lst::Union{VectorFunc, Vector}, B_lst::Union{VectorFunc, Vector}, r::Vector{dtype}, θ::Vector{dtype}; transpose::Bool=false)
    only_flag = isa(A_lst, VectorFunc) && isa(B_lst, VectorFunc)
    if isa(A_lst, VectorFunc)
        A_lst = [A_lst]
    end
    if isa(B_lst, VectorFunc)
        B_lst = [B_lst]
    end

    # Validate A, B list: same scaling factor and correct component keys
    for A in A_lst
        @assert A.scl_fac == A_lst[1].scl_fac "All VectorFuncs must have the same scaling factor"
        @assert all(u -> u in A.keys, ["u1", "u2", "u3"]) "Each VectorFunc in A_lst must have keys u1, u2, u3"
    end
    for B in B_lst
        @assert B.scl_fac == A_lst[1].scl_fac "All VectorFuncs must have the same scaling factor"
        @assert all(u -> u in B.keys, ["u1", "u2", "u3"]) "Each VectorFunc in B_lst must have keys u1, u2, u3"
        @assert B.bdry_index == B_lst[1].bdry_index "All VectorFuncs must have the same boundary condition index"
    end

    A_lst = [interpolate(A, r, θ) for A in A_lst]
    grad_mat = get_operator_matrix(B_lst[1], r, θ, "gradient"; weight_flag=true)
    grad_B_lst = [apply(grad_mat, B; new_keys=["A11", "A12", "A13", "A21", "A22", "A23", "A31", "A32", "A33"]) for B in B_lst]
    if transpose
        @assert only_flag "Transpose option is only supported for single VectorFunc input"
        return [mul_A_gradB_local(A_lst[1], grad_B_lst[1]), mul_A_gradB_local(A_lst[1], grad_B_lst[1]; transpose=true)]
    else
        result = [mul_A_gradB_local(A, grad_B) for A in A_lst, grad_B in grad_B_lst]
        if only_flag
            result = only(result)
        end
        return result
    end
end