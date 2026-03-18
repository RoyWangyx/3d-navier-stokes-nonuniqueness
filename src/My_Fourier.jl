# dt: A helper function to convert a number to the specified dtype.
if dtype == Float64
    @inline dt(x) = x isa String ? parse(Float64, x) : Float64(x)
elseif dtype == Interval{Float64}
    
    @inline dt(x) = @interval(x)
else
    error("Unsupported dtype: $dtype")
end

# is_zero: zero test used to skip negligible coefficients.
@inline is_zero(x::Float64)::Bool = abs(x) < 1e-14
@inline is_zero(x::Interval{Float64})::Bool = isequal_interval(x, @interval(0))

# Define Pi according to dtype
const Pi = dt(π)
@inline mysin(x) = sinpi(x / Pi)
@inline mycos(x) = cospi(x / Pi)

"""
Compute the ratio sequence for double factorials:
double_fact(n) = \\prod_{k=1}^{m} (1 - 1/(2k)), where n = 2m.
# Input:
- n: non-negative even integer
# Return:
- A vector of length max(1, n/2), where the k-th entry is double_fact(2k-2).
  If n == 0, return [1].
"""
function double_fact(n::Int)
    @assert n ≥ 0 && iseven(n) "n must be a non-negative even integer"
    result = [one(dtype)]
    b = one(dtype)
    for i in 2:2:(n-2)
        val = one(dtype) - inv(dt(i))
        b *= val
        push!(result, b)
    end
    return result
end

"""
Compute the ratio gamma(n-1.5)/gamma(n) stably for integer or half-integer n.
# Input:
- n: non-negative integer or half-integer
# Return:
- A vector of length floor(n), where the k-th entry is gamma(k-1.5)/gamma(k).
"""
function gamma_ratio(n::Real)
    frac = n - floor(n)
    @assert n ≥ 1 && frac in (0.0,0.5) "n must be a non-negative integer or half-integer"
    if n == 1
        return [gamma(dt(n - 1.5))/gamma(dt(n))]
    elseif n == 1.5
        return [Inf]
    end
    result = dtype[]
    if frac == 0.0
        push!(result, gamma(dt(-0.5))/gamma(dt(1.0)))
        b = gamma(dt(0.5))/gamma(dt(2.0))
    else
        push!(result, Inf)
        b = gamma(dt(1.0))/gamma(dt(2.5))
    end
    push!(result, b)
    for i in (2+frac):1:(n-1)
        b *= dt(i - 1.5)/dt(i)
        push!(result, b)
    end
    return result
end

"""
Convert Chebyshev to Legendre basis. Please look up the formula in the Appendix of the paper.
# Input:
- n: highest polynomial degree
- flag: true for even (w.r.t. z in Cartesian coordinates) basis, false for odd basis
# Return:
- The conversion matrix of shape (n+1, n+1).
"""
function che2len(n::Int; flag::Bool=false)
    A = zeros(dtype, n+1, n+1)
    b_lst = double_fact(4*(n+1))
    for i in 0:n, j in 0:i
        if flag
            A[i+1,j+1] = dt(2)*b_lst[i+j+2]*b_lst[i-j+1]
        else
            A[i+1,j+1] = dt(2)*b_lst[i+j+1]*b_lst[i-j+1]
            if j == 0
                A[i+1,j+1] /= dt(2)
            end
        end
    end
    return A
end

"""
Convert Legendre to Chebyshev basis. Please look up the formula in the Appendix of the paper.
# Input:
- n: highest polynomial degree
- flag: true for even (w.r.t. z in Cartesian coordinates) basis, false for odd basis
# Return:
- The conversion matrix of shape (n+1, n+1).
"""
function len2che(n::Int; flag::Bool=false)
    A = zeros(dtype, n+1, n+1)
    offset = flag ? 2.5 : 1.5
    ratio_lst1 = gamma_ratio(2 * n + offset)
    ratio_lst2 = gamma_ratio(n + 1)
    for i in 0:n, j in 0:i
        if flag
            factor = dt((4j + 3)*(i + 0.5))
            arg1 = i + j + 1
        else
            factor = dt((4j + 1)*i)
            arg1 = i + j
        end
        arg2 = i - j
        A[i+1,j+1] = - (factor * ratio_lst1[Int(arg1)+1] * ratio_lst2[Int(arg2)+1]) / dt(4)
    end
    if !flag
        A[1,1] = one(dtype) # Fix the first entry. Otherwise it becomes NaN due to the formula above.
    end
    return A
end

"""
Convert the coefficients of triangular basis to those of the spherical harmonic basis or vice versa.
# Input:
- N: highest polynomial degree
- bdry: boundary condition, a string in ("11","00","10","01"). The two characters in the string
    represent the boundary conditions at 0 and pi/2 respectively. "0" means Dirichlet while "1"
    means Neumann.
- direction: 0 for triangular->spherical harmonic, 1 for spherical harmonic->triangular
# Return:
- The conversion matrix of shape (N, N) (the shape might change if bdry="11").
"""
function conversion_mat(N::Int, bdry::String, direction::Int)
    if bdry == "11" && direction == 0
        # Note: The first spherical harmonic basis is constant. We impose the condition that the coefficient 
        # is zero, so the first column is removed. Refer to the section 5 of the paper for details.
        A = len2che(N-1)
        return A[:,2:end]
    elseif bdry == "11" && direction == 1
        # Note: Similarly as above, the first row is removed.
        A = che2len(N)
        return A[2:end,:]
    elseif bdry == "00" && direction == 0
        # Note: When the boundary condition is Dirichlet at 0, we use the sine basis. In spherical
        # harmonic basis, it is the derivative of P(cos(θ)), where P is the Legendre polynomial.
        # Thus our strategy is to apply anti-derivative first, then convert to Legendre basis. After
        # anti-derivative, the first cosine basis, which is constant, is not recovered. Thus the first
        # column and row of A are removed.
        n = dt.(1:N)
        A = transpose(len2che(N))
        B = Diagonal(dt(0.5) ./ n)# Change sign
        A = A[2:end,2:end]
        return transpose(A * B)
    elseif bdry == "00" && direction == 1
        n = dt.(1:N)
        A = transpose(che2len(N))
        B = Diagonal(dt(2) .* n)
        A = A[2:end,2:end]
        return transpose(B * A)
    elseif bdry == "10" && direction == 0
        return len2che(N-1, flag=true)
    elseif bdry == "10" && direction == 1
        return che2len(N-1, flag=true)
    elseif bdry == "01" && direction == 0
        # Note: Similar to the "00" case, we apply anti-derivative first. However, since the first
        # cosine basis is not constant, we do not remove the first column and row of A.
        n = dt.(1:N)
        A = transpose(len2che(N-1, flag=true))
        B = Diagonal(inv.(2 .* n .- 1))
        return transpose(A * B)
    elseif bdry == "01" && direction == 1
        n = dt.(1:N)
        A = transpose(che2len(N-1, flag=true))
        B = Diagonal(dt(2) .* n .- dt(1))
        return transpose(B * A)
    else
        error("Wrong flag: \$bdry and direction: \$direction")
    end
end

"""
Build the Discrete Fourier Transform matrix of shape (N+1, N+1) under boundary condition bdry.
# Input:
- N: highest polynomial degree
- bdry: boundary condition, same as in conversion_mat.
# Return:
- The Fourier transform matrix of shape approximately (N, N). (The shape might change according
    to the boundary condition.)
"""
function Fourier_mat(N::Int, bdry::String)
    if bdry == "11"
        # Note: The first row is divided by 2 because the integral of 1 is twice that of cos nx. 
        # The first and last columns are divided by 2 because they are on the boundary. The weight
        # of numerical integration is half for boundary points.
        n = reshape(dt.(0:N), :, 1)
        A = (dt(2)/dt(N)) .* mycos.(Pi .* (n * transpose(n)) ./ dt(N))
        A[1,:] ./= dt(2)
        A[:,[1,N+1]] ./= dt(2)
    elseif bdry == "00"
        # Note: We need to add two zero columns because the sine basis is zero at the boundary points.
        # The weights of numerical integration are 0 for boundary points.
        n = reshape(dt.(1:N-1), :, 1)
        A = (dt(2)/dt(N)) .* mysin.(Pi .* (n * transpose(n)) ./ dt(N))
        zeros_col = zeros(dtype, N-1, 1)
        A = hcat(zeros_col, A, zeros_col)
    elseif bdry == "01"
        # Note: Similar to the "00" case, we need to add one zero column before the first column.
        # Similar to the "11" case, the last column is divided by 2 because it is on the boundary.
        n = reshape(dt.(1:N), :, 1)
        A = (dt(2)/dt(N)) .* mysin.(Pi .* (((n .- dt(0.5)) * transpose(n)) ./ dt(N)))
        A[:,end] ./= dt(2)
        zeros_col = zeros(dtype, N, 1)
        A = hcat(zeros_col, A)
    elseif bdry == "10"
        n = reshape(dt.(0:N-1), :, 1)
        A = (dt(2)/dt(N)) .* mycos.(Pi .* (((n .+ dt(0.5)) * transpose(n)) ./ dt(N)))
        A[:,1] ./= dt(2)
        zeros_col = zeros(dtype, N, 1)
        A = hcat(A, zeros_col)
    else
        error("Wrong flag")
    end
    return A
end

"""
When U is divergence-free, the coefficients of U_r and U_θ needs to satisfy certain relation.
This function builds the matrix D such that B_θ = D * B_r can ensure the divergence-free condition,
where B_r and B_θ are the coefficients of U_r and U_θ under the chosen basis respectively.
Refer to the section 5 of the paper for details.
# Input:
- M: highest degree of triangular polynomial.
- bdry: boundary condition, the same as in conversion_mat.
# Return:
- The divergence matrix D of shape (M, M).
"""
function div_mat(M::Int, bdry::String)
    @assert bdry in ("00","01","10") "Invalid boundary condition"
    I_mat = vcat(diagm(0 => ones(dtype, M)), zeros(dtype, 1, M))
    diag = (dt.(1:M) .+ dt(0.5)) ./ dt(2)
    if bdry != "00" 
        diag .-= dt(0.25)
    end
    L = diagm(-1 => diag) - diagm(1 => diag)
    L = L[:,1:end-1]
    return dt(1.5) .* I_mat .+ L
end

"""
Given U_r, compute U_θ such that the vector field U is divergence-free.
# Input:
- U: Matrix representing the radial component of the vector field.
- bdry: list of boundary condition. If there is only one boundary condition (even), it is
    applied to all columns of U. If there are two boundary conditions (odd), the second
    one is applied to the first column of U, and the first one is applied to the rest
    of columns.
# Return:
- The projected polar component U_θ.
"""
function calc_Utheta_divfree(U::AbstractMatrix{dtype}, bdry::Vector{String})
    M0, M1 = size(U)
    DL = div_mat(M0, bdry[1])# Get the D matrix for the coefficients
    lbd = dt.(2 .* (1:M1))# index of spherical harmonics
    if length(bdry) > 1
        @assert length(bdry) == 2 "At most two boundary conditions allowed"
        DL2 = div_mat(M0, bdry[2])# Get the D matrix for the coefficients of the first column
        lbd .-= one(dtype)# Adjust the indices since we use the odd spherical harmonic basis
    end

    lbd .*= (lbd .+ one(dtype))# This is the eigenvalues of spherical harmonics
    DR = -Diagonal(inv.(lbd))# Construct the diagonal matrix

    # For the formula of U_θ, please refer to the section 5 of the paper.
    Uθ = DL * U 
    if length(bdry) > 1
        Uθ[:,1] = DL2 * U[:,1]  # second boundary condition for first column
    end
    Uθ = Uθ * DR
    
    if length(bdry) == 1
        return DL, DR, Uθ
    else
        return DL, DL2, DR, Uθ
    end
end


"""
Evaluate a named differential/operator expression applied to the trigonometric basis functions
used under boundary condition `bdry`. For the derivation of these formulas, see NS spherical.nb
in the project root.

# Input:
- `N`    : number of modes (highest index of the trigonometric basis).
- `x`    : sampling points (a 1D vector). Typically `x ⊂ [0, π/2]` or `x ⊂ [0, 2π]`.
- `key`  : name of the operator/expression to apply (see below).
- `bdry` : boundary condition string in {"11","00","01","10"} (same convention as `conversion_mat`).

# Return:
Returns a matrix `Y` of size `(length(x), N)` such that `Y[i, j]` equals the value at `x[i]`
of the operator specified by `key` applied to the `j`-th basis function.

# Basic:
- "I"       : phi
- "D"       : d/dx phi
- "D2"      : d^2/dx^2 phi
- "Cos"     : cos(x) * phi
- "Tan"     : tan(x) * phi  (implemented as phi * sin(x)/cos(x) with pointwise endpoint fixes)
- "Cot"     : cot(x) * phi  (implemented via cos(x) * (phi/sin(x)))
- "Scale"   : (sin(x)/2) * (cos(x)*phi + sin(x)*(d/dx phi))
- "Scale_T" : -(sin(x)/2) * (cos(x)*phi + sin(x)*(d/dx phi))
- "zero"    : zero matrix

Laplacian-related:
- "Laplacian_r_r", "Laplacian_r_θ", "Laplacian_θ_r_r"
- "Laplacian_θ_r", "Laplacian_ϕ_r"
- "Laplacian_θ_θ", "Laplacian_ϕ_θ"
- "Laplacian_r_odd", "Laplacian_θ_odd", "Laplacian_inter_odd"

D-operators (same grouping as in code):
- D_lst1: "D11_r_w","D21_θ_w","D31_ϕ_w"
- D_lst2: "D11_r_sym","D11_r","D21_θ","D31_ϕ"
- D_lst3: "D12_r_sym","D23_ϕ_sym"
- D_lst4: "D12_θ_sym","D13_ϕ_sym"
- D_lst5: "D22_r_sym","D22_θ_sym","D33_r_sym","D33_θ_sym",
          "D12_r","D22_θ","D32_ϕ","D22_r","D33_r",
          "D33_θ","D12_θ","D13_ϕ","D23_ϕ"
- D_lst6: "D12_r_w","D22_θ_w","D32_ϕ_w","D22_r_w","D33_r_w",
          "D33_θ_w","D12_θ_w","D13_ϕ_w","D23_ϕ_w"
- "D_p_r"
- "D23_ϕ_θ_sym"
"""
function trig_func(N::Int, x::Vector{dtype}, key::String, bdry::String)
    @assert size(x,2)==1
    # detect special lines (0, π, 2π)
    zero_lines = (abs.(x .- 0.0) .< 1e-14) .| (abs.(x .- 2*Pi) .< 1e-14)
    pi_lines = abs.(x .- Pi) .< 1e-14
    nrow = dt.(2 .* (1:N))'
    if bdry == "11"
        nrow .-= dt(2)
    elseif bdry in ("01","10")
        nrow .-= dt(1)
    elseif bdry == "00"
    else
        error("Wrong boundary condition: $bdry")
    end

    Lap_r_lst = ("Laplacian_θ_r","Laplacian_ϕ_r")
    Lap_θ_lst = ("Laplacian_θ_θ","Laplacian_ϕ_θ")
    D_lst1 = ("D11_r_w","D21_θ_w","D31_ϕ_w")
    D_lst2 = ("D11_r_sym","D11_r","D21_θ","D31_ϕ")
    D_lst3 = ("D12_r_sym", "D23_ϕ_sym")
    D_lst4 = ("D12_θ_sym","D13_ϕ_sym")
    D_lst5 = ("D22_r_sym","D22_θ_sym","D33_r_sym","D33_θ_sym",
                "D12_r","D22_θ","D32_ϕ","D22_r","D33_r",
                "D33_θ","D12_θ","D13_ϕ","D23_ϕ")
    D_lst6 = ("D12_r_w","D22_θ_w","D32_ϕ_w","D22_r_w","D33_r_w",
                "D33_θ_w","D12_θ_w","D13_ϕ_w","D23_ϕ_w")

    # 1) Determine which quantities are needed
    need_f_sin_x = key in ("Cot", "Laplacian_r_r", "Laplacian_θ_r_r", "D23_ϕ_θ_sym") || 
                key in D_lst3 || key in D_lst4 || key in D_lst5

    need_df_sin_x = (key == "Laplacian_r_θ")

    need_d2f = key in ("D2", "Laplacian_r_r", "Laplacian_r_θ", "Laplacian_θ_r", 
                    "Laplacian_ϕ_r", "Laplacian_θ_θ", "Laplacian_ϕ_θ", 
                    "Laplacian_r_odd", "Laplacian_θ_odd")

    need_f = key in ("I", "Cos", "Tan", "Scale", "Scale_T", "Laplacian_θ_r", "Laplacian_ϕ_r", 
                    "Laplacian_θ_θ", "Laplacian_ϕ_θ", "Laplacian_r_odd", "Laplacian_θ_odd", 
                    "Laplacian_inter_odd", "D_p_r") || 
            key in D_lst1 || key in D_lst2 || key in D_lst6 || need_f_sin_x || need_d2f

    need_df = key in ("D", "Scale", "Scale_T", "Laplacian_r_r", "Laplacian_θ_r_r", 
                    "Laplacian_θ_r", "Laplacian_ϕ_r", "Laplacian_θ_θ", "Laplacian_ϕ_θ", 
                    "Laplacian_r_odd", "Laplacian_θ_odd", "Laplacian_inter_odd", "D_p_r", 
                    "D23_ϕ_θ_sym") || key in D_lst1 || key in D_lst2 || key in D_lst4 || need_df_sin_x

    need_sin_x = key in ("Tan", "Scale", "Scale_T", "Laplacian_θ_θ", "Laplacian_ϕ_θ", "D_p_r") || 
                        key in D_lst1 || key in D_lst2 || need_f_sin_x || need_df_sin_x

    need_cos_x = key in ("Cot", "Cos", "Tan", "Scale", "Scale_T", "Laplacian_r_r", 
                        "Laplacian_r_θ", "Laplacian_θ_r_r", "Laplacian_θ_r", "Laplacian_ϕ_r", 
                        "Laplacian_θ_θ", "Laplacian_ϕ_θ", "D_p_r", "D23_ϕ_θ_sym") || key in D_lst1 ||
                        key in D_lst2 || key in D_lst3 || key in D_lst4 || key in D_lst5 || key in D_lst6

    need_sin_2x = key in ("Laplacian_r_r", "Laplacian_θ_r", "Laplacian_ϕ_r", "Laplacian_r_odd", 
                        "Laplacian_θ_odd", "Laplacian_inter_odd") || key in D_lst1

    need_cos_2x = key in ("Laplacian_r_r", "Laplacian_θ_r", "Laplacian_ϕ_r", "Laplacian_r_odd", 
                        "Laplacian_θ_odd", "Laplacian_inter_odd") || key in D_lst1 || key in D_lst4

    need_sin_4x = key in ("Laplacian_r_r", "Laplacian_r_odd", "Laplacian_θ_odd")

    need_cos_4x = key in ("Laplacian_r_r", "Laplacian_r_odd", "Laplacian_θ_odd")

    # 2) Calculate sin(x), cos(x), sin(2x), cos(2x), sin(4x), cos(4x) as needed
    sin_x = need_sin_x ? mysin.(x) : nothing
    cos_x = need_cos_x ? mycos.(x) : nothing

    sin_2x = need_sin_2x ? mysin.(dt(2) .* x) : nothing
    cos_2x = need_cos_2x ? mycos.(dt(2) .* x) : nothing

    sin_4x = need_sin_4x ? mysin.(dt(4) .* x) : nothing
    cos_4x = need_cos_4x ? mycos.(dt(4) .* x) : nothing

    # 3) Calculate sin(n x)/cos(n x) and f/df/d2f as needed
    # Note: f depends on bdry in ("11","10") => cos(n x)  else sin(n x)
    use_cos_basis = (bdry[1] == '1')

    f  = nothing
    df = nothing
    d2f = nothing

    if need_f || need_df
        if use_cos_basis
            # f = cos(n x)
            if need_f;  f  = mycos.(x .* nrow); end
            if need_df; df = .-(nrow) .* mysin.(x .* nrow); end
        else
            # f = sin(n x)
            if need_f;  f  = mysin.(x .* nrow); end
            if need_df; df = (nrow) .* mycos.(x .* nrow); end
        end
        if need_d2f; d2f = .-(nrow .^ 2) .* f; end
    end

    # 4) Calculate f/sin x and df/sin x as needed
    f_sin_x  = nothing
    df_sin_x = nothing

    if need_f_sin_x
        @assert !use_cos_basis "$key only defined for bdry[1]=='0'"
        f_sin_x = f ./ sin_x   # L×N
        f_sin_x[zero_lines, :] .= nrow # endpoint fix: sin(n x)/sin(x) → n at x≈0,2π
        f_sin_x[pi_lines, :] .= (bdry == "00" ? -nrow : nrow) # endpoint fix at x≈π (sign depends on bdry)
    end

    if need_df_sin_x
        @assert use_cos_basis "$key only defined for bdry[1]=='1'"
        df_sin_x = df ./ sin_x
        df_sin_x[zero_lines, :] .= -nrow.^2 # endpoint fix: (-n sin(n x))/sin(x) → -n^2 at x≈0,2π
        df_sin_x[pi_lines, :]   .= (bdry=="11" ?  nrow.^2 : -nrow.^2) # endpoint fix at x≈π (sign depends on bdry)
    end
    
    # 5) Select per key
    result = nothing
    if key == "I"
        result = f

    elseif key == "D"
        result = df

    elseif key == "D2"
        result = d2f

    elseif key == "Cot"
        result = cos_x .* f_sin_x

    elseif key == "Cos"
        result = cos_x .* f

    elseif key == "Tan"
        result = f .* sin_x ./ cos_x
        # detect special lines (π/2, 3π/2)
        pihalf_lines = abs.(x .- Pi/dt(2)) .< 1e-14
        threepihalf_lines = abs.(x .- 3*Pi/dt(2)) .< 1e-14
        result[pihalf_lines,:] .= nrow # endpoint fix at x≈π/2
        result[pihalf_lines,2:2:end] .*= -one(dtype) # parity sign flip for even modes
        result[threepihalf_lines,:] .= -nrow # endpoint fix at x≈3π/2
        result[threepihalf_lines,2:2:end] .*= -one(dtype) # parity sign flip for even modes

    elseif key == "Scale"
        result = (sin_x/dt(2)) .* (cos_x .* f .+ sin_x .* df)
    
    elseif key == "Scale_T"
        result = -(sin_x/dt(2)) .* (cos_x .* f .+ sin_x .* df)

    elseif key == "Laplacian_r_r"
        term1 = (-dt(19) .+ dt(3)*cos_4x) .* f_sin_x
        term2 = dt(4) .* cos_x .* (dt(4)*cos_2x .* df .+ sin_2x .* d2f)
        result = (cos_x .* (term1 .+ term2)) ./ dt(8)

    elseif key == "Laplacian_r_θ"
        result = df_sin_x .* cos_x .+ d2f

    elseif key == "Laplacian_θ_r_r"
        result = -dt(2) .* (f_sin_x .* cos_x .+ df)

    elseif key in Lap_r_lst
        term1 = dt(3) .* sin_2x .* f
        term2 = dt(4) .* cos_2x .* df
        term3 = sin_2x .* d2f
        result = (cos_x.^2) .* (.-term1 .+ term2 .+ term3) ./ dt(2)

    elseif key in Lap_θ_lst
        result = df .* (cos_x ./ sin_x) .- f ./ sin_x.^2 .+ d2f
        result[zero_lines,:] .= dt(0)
        result[pi_lines,:] .= dt(0)

    elseif key == "Laplacian_r_odd"
        result = ((dt(2)*sin_2x .+ sin_4x).*d2f)./dt(8) .+ ((dt(3) .+ dt(4)*cos_2x .+ cos_4x).*df)./dt(2) .- ((dt(14)*sin_2x .+ dt(3)*sin_4x).*f)./dt(8)

    elseif key == "Laplacian_θ_odd"
        result = ((dt(2)*sin_2x .+ sin_4x).*d2f)./dt(8) .+ ((dt(1) .+ dt(2)*cos_2x .+ cos_4x).*df)./dt(2) .- ((dt(6)*sin_2x .+ dt(3)*sin_4x).*f)./dt(8)

    elseif key == "Laplacian_inter_odd"
        result = (sin_2x./dt(4)) .* f .- ((dt(1) .+ cos_2x)./dt(4)) .* df

    elseif key == "D_p_r"
        result = sin_x .* (.-sin_x .* f .+ cos_x .* df)
    
    elseif key in D_lst1
        result = (sin_2x ./ dt(2)) .* (.-sin_x .* f .+ cos_x .* df)

    elseif key in D_lst2
        result = cos_x.^2 .* (.-sin_x .* f .+ cos_x .* df)

    elseif key in D_lst3
        result = (cos_x.^2 ./ dt(2)) .* f_sin_x

    elseif key in D_lst4
        result = (cos_x.^2 ./ dt(4)) .* ((-dt(3) .+ cos_2x).*f_sin_x .+ dt(2)*cos_x.*df)

    elseif key in D_lst5
        result = cos_x.^2 .* f_sin_x
    
    elseif key in D_lst6
        result = cos_x .* f

    elseif key == "D23_ϕ_θ_sym"
        result = .-f_sin_x .* cos_x .+ df

    elseif key == "zero"
        result = zeros(dtype, size(x,1), N)

    else
        error("Invalid key: $key")
    end
    return result
end