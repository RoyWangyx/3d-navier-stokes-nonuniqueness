"""
Compute L∞ bounds for all partial derivatives (up to total order `n`) of a product
of `k` factors via the multinomial Leibniz rule.

Let x ∈ ℝ^d and define
    F(x) = ∏_{j=1}^k f_j(x).
For a multi-index α ∈ ℕ^d with |α| = α₁+⋯+α_d, assume we have bounds for each factor:
    ‖∂^α f_j‖_∞ ≤ b_j(α),    for all |α| ≤ n.
Then the product satisfies the multinomial Leibniz bound
    ‖∂^α F‖_∞
    ≤  ∑_{α¹+⋯+α^k = α}  (α! / (α¹!⋯α^k!))  ∏_{j=1}^k b_j(α^j),
where the sum is over all decompositions of the multi-index α across the factors,
and the multi-index factorials are
    α!  = ∏_{i=1}^d α_i!,   α^j! = ∏_{i=1}^d (α^j_i)!.

In this implementation, bounds are provided for each factor in a “tensor-product”
layout over derivative order and coordinate index:
- The first dimension corresponds to derivative order `m = 0..n` (so size ≥ n+1).
- The second dimension corresponds to coordinate/variable index `l = 1..d`.
For each factor `j`, the entry `a[j][m+1, l]` stores an upper bound for the
m-th derivative with respect to the l-th variable (or the corresponding block bound,
see below). These per-(m,l) bounds are combined by factorial scaling and repeated
convolution in `m`, mimicking the multinomial coefficients.

Note: This routine uses the per-variable derivative layout above; it does not
explicitly enumerate all multi-indices α with |α|≤n. Therefore it should be used
when your bound representation factorizes/aggregates across variables in a way
compatible with this convolution scheme.

# Value shapes (scalar vs matrix chain)

- `a[1]` (first factor) is scalar-valued: `a[1][m+1, l]` is a scalar bound.
- For `j ≥ 2`, each `a[j][m+1, l]` may be matrix- or vector-valued (a block bound).
  The factors are interpreted as a matrix product chain per variable l, and
  dimensions are checked for consistency across the chain.

# Inputs
- `n::Int`:
    Maximum derivative order (bounds for orders 0..n must be provided).
- `a...`:
    Bound containers for each factor.
    Requirements:
    - `size(a[j], 1) ≥ n+1` for all `j`.
    - `size(a[j], 2) == d` is the number of variables.
    - For `j ≥ 2`, blocks `a[j][m+1, l]` must be multiplicatively compatible.
- `last_flag::Bool=true`:
    If `true`, return only the order-`n` bound; otherwise return bounds for all
    orders 0..n.

# Returns
- If `last_flag=true`:
    The order-`n` bound(s) for the product, organized over variables l=1..d, with
    the same block shape as the final factor (per l). If `d==1` and/or blocks are
    1×1, the output may be simplified to a single block or scalar.
- If `last_flag=false`:
    Bounds for all orders 0..n, returned as an `(n+1)×d` array of blocks/scalars.

"""
function multinomial_sum(n::Int, a...; last_flag::Bool=true)
    # n * dim * r * c
    k = length(a)
    @assert k ≥ 2
    for j in 1:k
        @assert size(a[j], 1) ≥ n + 1
    end

    # fact_lst[m+1] = m!
    fact_lst = ones(dtype, n + 1)
    @inbounds for m in 1:n
        fact_lst[m + 1] = fact_lst[m] * dt(m)
    end

    # b1[m+1] = a1[m+1] / m!
    d = size(a[1], 2)
    dp = zeros(dtype, n + 1, d)
    @inbounds for m in 0:n
        dp[m + 1, :] = a[1][m + 1, :] / fact_lst[m + 1]
    end

    r = size(a[2][1,1], 1)
    c = r
    c_next = 0

    # Now convolve with remaining matrix polynomials
    for j in 2:k
        @assert size(a[j], 2) == d "dimension mismatch a j=$j: A_j depth=$(size(a[j],4)), expected $d"
        @assert size(a[j][1,1], 1) == c "dimension mismatch a j=$j: A_j rows=$(size(a[j],2)), expected $c"
        c_next = size(a[j][1,1], 2)

        new_dp = [zeros(dtype, r, c_next) for _ in 1:(n + 1), _ in 1:d]
        @views for p in 0:n, l in 1:d
            Ajp = a[j][p + 1, l]
            αp  = inv(fact_lst[p + 1])
            @inbounds for m in 0:(n - p)
                new_dp[m + p + 1, l] .+= (dp[m + 1, l] * αp) * Ajp
            end
        end
        dp = new_dp
        c  = c_next
    end

    # multiply back by n! 
    if last_flag
        out = dp[n + 1, :] .* fact_lst[n + 1]      # multiply back by n!
    else
        out = [dp[m, j] .* fact_lst[m] for m in 1:n+1, j in 1:d]
    end

    # simplify output shape if possible
    if r == 1 && c == 1
        out = [only(each) for each in out]
    end
    if last_flag && d == 1
        out = only(out)
    end
    return out
end

"""
Differentiate a polynomial represented by its coefficient vector.
Inputs:
- c : Vector{BigInt}, polynomial coefficients
Output:
- d : Vector{BigInt}, coefficients of p'(x)
"""
function poly_deriv(c::Vector{BigInt})
    deg = length(c) - 1
    if deg == 0
        return BigInt[0]
    end
    d = Vector{BigInt}(undef, deg)
    for i in 1:deg
        d[i] = BigInt(i) * c[i+1]
    end
    return d
end

"""
Add two polynomials represented by coefficient vectors.
Inputs:
- a, b : Vector{BigInt}, polynomial coefficients
Output:
- out : Vector{BigInt}, coefficients of the sum polynomial
"""
function poly_add(a::Vector{BigInt}, b::Vector{BigInt})
    n = max(length(a), length(b))
    out = fill(BigInt(0), n)

    for i in 1:length(a)
        out[i] += a[i]
    end
    for i in 1:length(b)
        out[i] += b[i]
    end

    # remove trailing zeros
    while length(out) > 1 && out[end] == 0
        pop!(out)
    end
    return out
end

"""
Multiply a polynomial by x.
Inputs:
- c : Vector{BigInt}, polynomial coefficients
Output:
- out : Vector{BigInt}, coefficients of x·p(x)
"""
function poly_mul_x(c::Vector{BigInt})
    out = fill(BigInt(0), length(c) + 1)

    for i in 1:length(c)
        out[i+1] = c[i]
    end

    # remove trailing zeros
    while length(out) > 1 && out[end] == 0
        pop!(out)
    end
    return out
end

"""
Multiply a polynomial by (1 + x²).
Inputs:
- c : Vector{BigInt}, polynomial coefficients
Output:
- out : Vector{BigInt}, coefficients of (1 + x²)p(x)
"""
function poly_mul_1_plus_x2(c::Vector{BigInt})
    deg = length(c) - 1
    out = fill(BigInt(0), deg + 3)  # degree +2
    for i in 0:deg
        out[i+1] += c[i+1]          # *1
        out[i+3] += c[i+1]          # *x^2
    end
    # remove trailing zeros for cleanliness
    while length(out) > 1 && out[end] == 0
        pop!(out)
    end
    return out
end

"""
Compute a safe upper bound for |p(x)| over |x| ≤ R.
Uses the inequality
    sup_{|x|≤R} |∑ c[i+1] x^i| ≤ ∑ |c[i+1]| R^i.
Inputs:
- c : Vector{BigInt}, polynomial coefficients
- R : radius bound
Output:
- s : dtype, upper bound on sup_{|x|≤R} |p(x)|
"""
function poly_abs_sum_bound(c::Vector{BigInt}, R::dtype)
    s = zero(dtype)
    p = one(dtype)
    for i in 0:(length(c)-1)
        s += dt(abs(c[i+1])) * p
        p *= R
    end
    return s
end

"""
Compute rigorous L∞ bounds for all derivatives up to order m of the composite function
    g(β) = f(a * tanβ)
on the interval β ∈ (0, arctan(1/a)), given L∞ bounds on derivatives of f.
The algorithm proceeds in three conceptual steps:
1. Construct bounds on the derivatives of r(β) = a·tanβ using exact polynomial
   representations of dⁿ/dβⁿ(tanβ) and a safe absolute-value estimate on |tanβ| ≤ 1/a.
2. Use Bell polynomials to represent the derivative structure of the composite function
   g(β) = f(r(β)) via the chain rule (Faà di Bruno formula).
3. Combine these structural coefficients with the given bounds M[j] ≥ ‖f^(j-1)‖∞
   to obtain final bounds B[k+1] ≥ ‖g^(k)‖∞ for k = 0..m.

# Inputs
- `M`    : vector of length ≥ m+1, where M[j+1] ≥ ‖f^(j)‖∞ on the r-domain.
- `a`    : scaling parameter in r = a·tanβ, must satisfy 0 < a < 1.
- `m`    : maximum derivative order.

# Output
- `B::Vector{dtype}` of length m+1, where
      B[k+1] ≥ ‖ d^k/dβ^k [ f(a·tanβ) ] ‖∞.
"""
function beta_all_deriv_Linf_bounds(M::AbstractVector{dtype}, a::dtype, m::Integer)
    @assert 0 < a < 1 "Require 0 < a < 1."
    @assert m ≥ 0 "Require m ≥ 0."
    @assert length(M) ≥ m+1 "Need M[j+1] for j=0..m."

    R  = inv(a)      # on β∈(0,atan(1/a)): |tanβ| ≤ 1/a

    # ============================================================
    # Step 1: Bound derivatives of r(β) = a·tanβ
    #
    # For tanβ, its n-th derivative admits the representation
    #   dⁿ/dβⁿ (tanβ) = P_n(tanβ),
    # where P_n(x) are polynomials defined by the recurrence
    #   P_0(x) = x,    P_{n+1}(x) = (1 + x^2) P_n'(x).
    #
    # Since β ∈ (0, arctan(1/a)), we have |tanβ| ≤ R = 1/a, hence
    #   ‖r^(n)‖∞ = a · ‖P_n(tanβ)‖∞ ≤ a · sup_{|x|≤R} |P_n(x)|.
    #
    # These suprema are safely bounded using the coefficient-wise estimate
    #   sup_{|x|≤R} |∑ c_i x^i| ≤ ∑ |c_i| R^i.
    #
    # We store the resulting bounds as:
    #   x[n] = ‖r^(n)‖∞   for n = 1..m.
    # ============================================================

    x = Vector{dtype}(undef, max(m,1))   # x[1]=‖r'‖∞, ..., x[m]=‖r^{(m)}‖∞
    if m ≥ 1
        P = BigInt[0, 1]                 # P_0(x)=x
        for n in 1:m
            P = poly_mul_1_plus_x2(poly_deriv(P))  # P_n = (1+x^2) * P'_{n-1}
            x[n] = a * poly_abs_sum_bound(P, R)
        end
    end

    # ============================================================
    # Step 2: Precompute partial Bell polynomials B_{n,j}
    #
    # Faà di Bruno formula:
    #   g^(n)(β) = Σ_{j=1..n} f^(j)(r(β)) * B_{n,j}(r',...,r^(n-j+1))
    #
    # We evaluate B_{n,j} on the bounds x[i] = ‖r^(i)‖∞
    # using the recurrence:
    #   B_{0,0} = 1
    #   B_{n,j} = Σ_{i=1..n-j+1} binomial(n-1,i-1) * x[i] * B_{n-i,j-1}
    # ============================================================

    Bbell = fill(zero(dtype), (m+1, m+1))
    Bbell[1,1] = one(dtype)  # B_{0,0}

    for n in 1:m, j in 1:n
        s = zero(dtype)
        max_i = n - j + 1
        for i in 1:max_i
            s += dt(binomial(n-1, i-1)) * x[i] * Bbell[n - i + 1, j]  # (n-i, j-1)
        end
        Bbell[n+1, j+1] = s
    end

    # ============================================================
    # Step 3: Combine with bounds on f^(j)
    #
    # Since:
    #   |g^(k)(β)| ≤ Σ_{j=1..k} ‖f^(j)‖∞ * B_{k,j}(‖r'‖∞,...)
    #
    # and M[j+1] ≥ ‖f^(j)‖∞, we obtain final bounds.
    # ============================================================

    out = Vector{dtype}(undef, m+1)
    out[1] = M[1]  # k=0: ‖g‖∞ ≤ ‖f‖∞

    for k in 1:m
        s = zero(dtype)
        for j in 1:k
            s += M[j+1] * Bbell[k+1, j+1]
        end
        out[k+1] = s
    end

    return out
end

"""
Compute L∞ bounds for derivatives of
    f(β) = tan(β)/cos(β) = sec(β)·tan(β)
on β ∈ [0, β0]. Main idea:
1. Express all derivatives of f in the form
   f^(n)(β) = sec(β) · R_n(tanβ), where R_n is a polynomial obtained
   from a simple recurrence.
2. Bound tanβ and secβ on the interval using β0.
3. Use a safe coefficient-based bound for each polynomial R_n.

Inputs:
- β0 : interval endpoint, with 0 < β0 < π/2
- k  : maximum derivative order

Output:
- B::Vector{dtype} of length k+1, where
      B[n+1] ≥ ‖ d^n/dβ^n [ tan(β)/cos(β) ] ‖∞  on β ∈ [0, β0]
  for n = 0..k.
"""
function tan_over_cos_deriv_Linf_bounds(β0::dtype, k::Integer)
    @assert k ≥ 0 "Require k ≥ 0."
    @assert 0 < β0 < Pi/dt(2) "Require 0 < β0 < π/2."

    # Interval bounds:
    #   |tanβ| ≤ T = tan(β0),   secβ ≤ S = sec(β0) = 1 / cos(β0)
    T = tan(β0)
    S = inv(mycos(β0))

    # Output vector:
    #   B[n+1] ≥ ‖f^(n)‖∞   for n = 0..k
    B = Vector{dtype}(undef, k+1)

    # We represent derivatives via polynomials R_n such that
    #   f^(n)(β) = sec(β) · R_n(tanβ).
    #
    # Initialize R_0(x) = x  (corresponding to f(β) = secβ·tanβ)
    Rpoly = BigInt[0, 1]

    # n = 0:
    #   |f(β)| = |secβ · tanβ| ≤ S · T
    B[1] = S * T

    for n in 1:k
        # Polynomial recurrence:
        #   R_{n}(x) = x·R_{n-1}(x) + (1 + x^2)·(d/dx)R_{n-1}(x)
        #
        # This follows from differentiating
        #   f^(n-1)(β) = sec(β)·R_{n-1}(tanβ)
        # using the chain rule and d/dβ tanβ = 1 + tan^2β.
        Rp    = poly_deriv(Rpoly)
        term1 = poly_mul_x(Rpoly)
        term2 = poly_mul_1_plus_x2(Rp)
        Rpoly = poly_add(term1, term2)

        # Bound:
        #   |f^(n)(β)| ≤ S · sup_{|x|≤T} |R_n(x)|
        # and use the coefficient-wise safe bound for the polynomial.
        B[n+1] = S * poly_abs_sum_bound(Rpoly, T)
    end

    return B
end

"""
Transform derivative bounds with respect to r into bounds with respect to β.
Given a list of matrices A, where each A[j] stores L∞ bounds on the (j-1)-th
derivative with respect to r, this function applies the change of variable
    r = a0 · tan(β)
and uses `beta_all_deriv_Linf_bounds` to obtain the corresponding bounds
with respect to β.

Inputs:
- A  : Vector of matrices, A[j][p,q] ≥ ‖ ∂^(j-1) f_{p,q} / ∂r^(j-1) ‖∞
- a0 : parameter in r = a0·tan(β)

Output:
- result2 : Vector of matrices, where result2[j][p,q] ≥
            ‖ ∂^(j-1) f_{p,q} / ∂β^(j-1) ‖∞
"""
function r_to_beta(A, a0::dtype)
    k = length(A)
    m, n = size(A[1])

    # Stack A[j] into a 3D array:
    #   bound_tensor[j, p, q] = A[j][p, q]
    bound_tensor = zeros(dtype, k, m, n)
    for j in 1:k, p in 1:m, q in 1:n
        bound_tensor[j, p, q] = A[j][p, q]
    end

    # For each entry (p, q), treat the vector over derivative order
    # and apply the r → β transformation using beta_all_deriv_Linf_bounds.
    for p in 1:m, q in 1:n
        bound_tensor[:, p, q] =
            beta_all_deriv_Linf_bounds(bound_tensor[:, p, q], a0, k-1)
    end

    # Convert back to a vector of matrices for consistency with input format
    out = [bound_tensor[i, :, :] for i in 1:k]
    return out
end

"""
Combine basis-function bounds with eigenvectors to obtain bounds for eigenfunctions.
# Input:
- bd_mat: bound tensor of size (k, N, 3) or compatible, where k is derivative order
- eig_vec: matrix of eigenvectors (size N × M)
# Return:
- Vector of length k, where each entry is a matrix of size 3 × M giving the
  corresponding bounds for each derivative order
"""
function eig_fun_bound(bd_mat, eig_vec)
    k = size(bd_mat, 1)
    abs_eig_vec = abs.(eig_vec)
    result = [bd_mat[i, :, :]' * abs_eig_vec for i in 1:k]
    return result
end