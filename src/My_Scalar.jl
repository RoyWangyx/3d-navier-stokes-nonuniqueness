# BDRY_LIST: list of valid boundary conditions, 0 for Dirichlet,
#  1 for Neumann, two entries for two boundaries.
const BDRY_LIST = ["00", "01", "10", "11"]

"""
Validate the boundary-condition descriptor `bdry`. Valid boundary-condition
codes are listed in `BDRY_LIST`.

# Input
- bdry::Vector: a length-2 vector of the form `[bdry_r, bdry_θ]`, where
    - `bdry_θ::String` is the angular boundary condition and must be in `BDRY_LIST`;
    - `bdry_r` is the radial boundary condition and is either
        1) `String` in `BDRY_LIST` (same boundary condition used for all columns), or
        2) `Vector{String}` of length 2 with both entries in `BDRY_LIST`
        (two radial boundary conditions; typically the first one is applied to the
        first column and the second one to the remaining columns).

# Throws
- `error(...)` if `bdry` has invalid length, invalid types, or contains codes not
    in `BDRY_LIST`.

# Return
- No return value (used for validation).
"""
function check_bdry(bdry::Vector)
    if length(bdry)==2 && bdry[2] in BDRY_LIST
        # ok
    else
        error("Invalid boundary: $bdry")
    end

    if bdry[1] ∉ BDRY_LIST
        if isa(bdry[1], Vector) && length(bdry[1])==2 && all(x->x in BDRY_LIST, bdry[1])
            # ok
        else
            error("Invalid boundary: $(bdry[1])")
        end
    end
end

"""
    Mutable struct ScalarFunc
A scalar function in space and frequency domains, with associated methods.
# Fields:
- name::String: name of the function (e.g., "u1", "u2", "u3", "p")
- bdry::Union{Vector, Nothing}: boundary conditions
- space_func::Union{AbstractMatrix{dtype},Nothing}: function values on spatial domain. Dimension
    1 is radial direction, dimension 2 is angular direction
- freq_func::Union{AbstractMatrix{dtype},Nothing}: function values on frequency domain. Dimension
    1 is radial direction, dimension 2 is angular direction
- conv_mat::Union{AbstractMatrix{dtype},Nothing}: θ conversion matrix (if applicable) from
    Fourier basis to spherical harmonic basis
- scl_fac::dtype: scaling factor in space domain
"""
mutable struct ScalarFunc
    name::String
    bdry::Union{Vector, Nothing}
    space_func::Union{AbstractMatrix{dtype},Nothing}
    freq_func::Union{AbstractMatrix{dtype},Nothing}
    conv_mat::Union{AbstractMatrix{dtype},Nothing}
    scl_fac::dtype
end

"""
Constructor for ScalarFunc
# Input:
- name::String: name of the function (e.g., "u1", "u2", "u3", "p")
- bdry::Union{Vector, Nothing}: boundary conditions, default nothing
- scl_fac::dtype: scaling factor in space domain, default 1.0
# Return:
- ScalarFunc object with specified name, boundary, and scaling factor.
    space_func, freq_func, and conv_mat are initialized to nothing.
"""
function ScalarFunc(name::String; bdry::Union{Vector, Nothing}=nothing, scl_fac::dtype=one(dtype))
    # Check the validity of boundary conditions
    bdry !== nothing && check_bdry(bdry)
    return ScalarFunc(name, bdry, nothing, nothing, nothing, scl_fac)
end

"""
Set function values in specified domain
# Input:
- self::ScalarFunc: ScalarFunc object
- domain::String: "space" or "frequency"
- func::Array{dtype,2}: 2D array of function values
# Modifies:
- self.space_func or self.freq_func is set to func based on domain
# Throws error if domain is invalid. No return value.
"""
function set_func!(self::ScalarFunc, domain::String, func::Array{dtype,2})
    if domain == "space"
        self.space_func = func
    elseif domain == "frequency"
        self.freq_func = func
    else
        error("Invalid domain: $domain")
    end
end

"""
Set boundary conditions for the ScalarFunc object
# Input:
- self::ScalarFunc: ScalarFunc object
- bdry::Vector: boundary conditions, either [String, String] or [Vector{String}, String]. 
# Modifies:
- self.bdry is set to bdry
# Throws error if self.bdry is already set or if bdry is invalid. No return value.
"""
function set_bdry!(self::ScalarFunc, bdry::Vector)
    @assert self.bdry === nothing "Boundary already set"
    check_bdry(bdry)
    self.bdry = bdry
end

# Set self.conv_mat to the θ conversion matrix based on self.bdry (in-place).
function set_conv_mat!(self::ScalarFunc)
    @assert self.freq_func !== nothing "freq_func not set"
    @assert self.bdry !== nothing "boundary not set"
    self.conv_mat = conversion_mat(size(self.freq_func, 2), self.bdry[2], 1)
end

"""
Discrete Fourier transform: space->frequency
# Input:
- self::ScalarFunc: ScalarFunc object
- conv_flag::Bool: whether to apply conversion matrix, default false
# Return:
- freq_func::Array{dtype,2}: 2D array of function values in frequency domain
# Throws error if self.space_func is not set
"""
function Fourier_transform(self::ScalarFunc; conv_flag::Bool=false)
    # Fourier transform procedure: angular DFT -> conversion to 
    # spherical harmonic basis (if needed) -> radial DFT

    @assert self.space_func !== nothing && self.bdry !== nothing

    # Apply Fourier transform in θ direction
    N0, N1 = size(self.space_func)
    Fθ = Fourier_mat(N1-1, self.bdry[2])
    tmp = self.space_func * Fθ'

    # Apply conversion to spherical harmonic basis if needed
    if conv_flag
        conv_mat = conversion_mat(size(tmp,2), self.bdry[2], 0)
        tmp = tmp * conv_mat
        set_conv_mat!(self)
    end

    # Apply Fourier transform in radial direction
    if isa(self.bdry[1], String)
        Fr = Fourier_mat(N0-1, self.bdry[1])
        return Fr * tmp
    else
        # If there are two boundary conditions, we need to treat
        # the first column and the rest columns separately
        Fr1 = Fourier_mat(N0-1, self.bdry[1][1])
        Fr2 = Fourier_mat(N0-1, self.bdry[1][2])
        freq_func1 = Fr1 * tmp[:,1:1]
        freq_func2 = Fr2 * tmp[:,2:end]
        m1 = size(freq_func1, 1)
        m2 = size(freq_func2, 1)
        if m2 < m1
            freq_func2 = vcat(freq_func2, zeros(dtype, m1 - m2, size(freq_func2,2)))
        end
        return hcat(freq_func1, freq_func2)
    end
end

"""
Interpolate: frequency->space
This function is more general then inverse Fourier transform, as it evaluates
the result of a lot of operators acting on the scalar function at arbitrary points.
# Input:
- self::ScalarFunc: ScalarFunc object
- r_pt::Vector{dtype}: radial points
- θ_pt::Vector{dtype}: angular points
- keys::Vector{String}: list of two strings specifying the type of operators on the
    scalar function to be applied. The two strings are for operators used in radial
    and angular directions respectively. Default is ["I","I"], meaning identity operator.
- pt_flag::Bool: if true, return the intermediate result after radial transform
    and before angular transform. Default false.
# Return:
- u::Array{dtype,2}: 2D array of function values at (r_pt, θ_pt) if pt_flag is false,
    or after radial transform if pt_flag is true.
# Throws error if self.freq_func is not set.
"""
function interpolate(self::ScalarFunc, r_pt::Vector{dtype}, θ_pt::Vector{dtype};
        keys::Vector{String}=["I","I"], pt_flag=false)
    M0 = size(self.freq_func, 1)
    # Interpolation procedure: radial transform -> conversion to
    # Fourier basis (if needed) -> angular transform, which is
    # the reverse order of Fourier transform.

    # Apply radial transform
    if isa(self.bdry[1], String)
        u_r = trig_func(M0, r_pt, keys[1], self.bdry[1])
        tmp = u_r * self.freq_func
    else
        # If there are two boundary conditions, we need to treat
        # the first column and the rest columns separately
        u_r1 = trig_func(M0, r_pt, keys[1], self.bdry[1][1])
        u_r2 = trig_func(M0, r_pt, keys[1], self.bdry[1][2])
        t1 = u_r1 * self.freq_func[:,1:1]
        t2 = u_r2 * self.freq_func[:,2:end]
        tmp = hcat(t1, t2)
    end

    # Apply conversion to Fourier basis if needed
    self.conv_mat !== nothing && (tmp *= self.conv_mat)
    if pt_flag
        return tmp
    else
        u_θ = trig_func(size(tmp,2), θ_pt, keys[2], self.bdry[2])
        return tmp * u_θ'
    end
end

"""
Transform a scalar function to frequency/space domain
# Input:
- self::ScalarFunc: ScalarFunc object
- domain::String: source domain, "space" or "frequency"
- shape::Union{Tuple{Int,Int},Nothing}: shape of the function matrix after transform.
    If not provided, it will be inferred from existing space_func.
- conv_flag::Bool: whether to apply conversion matrix when transforming to frequency
    domain, default false
# Modifies:
- self.space_func or self.freq_func is set based on domain
# Throws error if domain is invalid or if shape cannot be inferred. No return value.
"""
function transform!(self::ScalarFunc, domain::String;
    shape::Union{Tuple{Int,Int},Nothing}=nothing, conv_flag::Bool=false)
    # Determine shape of the function matrix
    if shape === nothing
        @assert self.space_func !== nothing "No space function"
        N0, N1 = size(self.space_func)
    else
        N0, N1 = shape
    end

    # For space->frequency transform, use the Fourier transform
    if domain == "space"
        self.freq_func = Fourier_transform(self; conv_flag=conv_flag)
    # For frequency->space transform, use interpolation
    elseif domain == "frequency"
        # build radial and angular grids
        r_pt = dt.(0:N0-1) .* (Pi/dt(2*(N0-1)))
        θ_pt = dt.(0:N1-1) .* (Pi/dt(2*(N1-1)))
        # invert transform
        self.space_func = interpolate(self, r_pt, θ_pt)
    else
        error("Invalid domain: $domain")
    end
end

"""
Compute the inner product of two ScalarFunc objects on the spatial grid.
The inner product is evaluated by numerical quadrature (Newton–Cotes rule) on the
space_func arrays, with domain sizes given by domain, and scaled by self.scl_fac.
# Input:
- self: first ScalarFunc
- other: second ScalarFunc
- domain: [L0, L1], domain lengths in the two dimensions (default [π/2, π/2])
# Return:
- inner product value (dtype)
"""
function inner_product(self::ScalarFunc, other::ScalarFunc; domain::Vector{dtype}=[Pi/dt(2), Pi/dt(2)])
    N0, N1 = size(self.space_func)
    # requires (N0-1) % 6 == 0 and (N1-1) % 6 == 0
    L0, L1 = domain
    hβ = L0/dt(N0-1)
    hθ = L1/dt(N1-1)

    wβ = newton_cotes_weights(N0-1, hβ)
    wθ = newton_cotes_weights(N1-1, hθ)

    # Calculate inner product efficiently with @simd
    s = zero(dtype)
    @inbounds for j in 1:N1
        tj = wθ[j] * mysin(dt(j-1) * hθ)# weighted by sin(θ)
        @simd for i in 1:N0
            s += self.space_func[i,j] * other.space_func[i,j] * wβ[i] * tj
        end
    end
    
    return s * self.scl_fac # scale by scl_fac
end

"""
Compute the list of W^{k,∞} norm estimates from the space-domain representation.
This routine evaluates the W^{k,∞} bounds along both coordinate directions, returning
a (k+1)×2 array of estimates.

# Input:
- f::ScalarFunc: ScalarFunc object with space_func set
- k: derivative order
- paras_lst: parameter lists passed to estimate_Wk∞_norm (two directions)
- L: domain size [L0, L1]

# Return:
- Array{dtype,2} of size (k+1, 2) containing W^{k,∞} estimates.
  (Exception: if k == 0, this function returns a scalar dtype, not a 1×2 array.)
"""

function get_Wk∞_norm_lst(f::ScalarFunc, k::Int; 
    paras_lst::Vector=[[nothing, nothing], [nothing, nothing]], L::Vector{dtype}=[Pi/dt(2), Pi/dt(2)])
    L0, L1 = L
    A = f.space_func
    N1, N2 = size(A) .- 1
    Ne = N1 - 2 * k
    h = [L0 / dt(Ne), L1 / dt(N2)]
    if k == 0 # no derivative, just return the L∞ norm of the whole array
        return estimate_Wk∞_norm(A, h, k; paras=paras_lst)
    end
    result = zeros(dtype, k+1, 2)
    @views begin # @views can speed up slicing
        # Note that space_func stores k extra grid layers on each side in both directions,
        # used for finite-difference evaluation near the boundary. When computing
        # directional derivative bounds, the redundant layers in the transverse direction
        # are removed before estimation.
        result[:, 1] = get_Wk∞_norm_lst(A[:, k+1:end-k], h, k; paras=paras_lst[1])
        # The second direction is transposed, so reverse h
        result[:, 2] = get_Wk∞_norm_lst(A[k+1:end-k, :]', reverse(h), k; paras=paras_lst[2])
    end 
    return result
end

"""
Trim the border of the space-domain array in-place.

# Input:
- self::ScalarFunc: ScalarFunc object (requires self.space_func to be set)
- num::Int or (Int,Int): number of rows/cols to trim from each side (or separately)

# Modifies:
- self.space_func is replaced by its trimmed subarray

# Throws:
- Error if self.space_func is not set
- Error if the trim size is too large for the current array
"""
function trim_border!(self::ScalarFunc, num::Union{Int, Tuple{Int,Int}})
    @assert self.space_func !== nothing "No space function"
    if isa(num, Int)
        @assert num*2 < size(self.space_func, 1) "Trim size too large"
        @assert num*2 < size(self.space_func, 2) "Trim size too large"
        self.space_func = self.space_func[num+1:end-num, num+1:end-num]
    else
        num1, num2 = num
        @assert num1*2 < size(self.space_func, 1) "Trim size too large"
        @assert num2*2 < size(self.space_func, 2) "Trim size too large"
        self.space_func = self.space_func[num1+1:end-num1, num2+1:end-num2]
    end
end

"""
Plot the space function using imshow
# Input:
- self::ScalarFunc: ScalarFunc object
# Throws error if self.space_func is not set. No return value.
"""
function my_plot(self::ScalarFunc)
    @assert self.space_func !== nothing "No space function"
    plot_matrix(self.space_func; title_str="Space function: $(self.name)")
end

"""
Reload copy for ScalarFunc
# Input:
- self::ScalarFunc: ScalarFunc object
# Return:
- new::ScalarFunc: a deep copy of self
"""
function copy(self::ScalarFunc)
    new_space_func = (self.space_func === nothing) ? nothing : copy(self.space_func)
    new_freq_func = (self.freq_func === nothing) ? nothing : copy(self.freq_func)
    new_conv_mat = (self.conv_mat === nothing) ? nothing : copy(self.conv_mat)
    return ScalarFunc(self.name, self.bdry, new_space_func, new_freq_func,
        new_conv_mat, self.scl_fac)
end

"""
add_equal!: += operator for ScalarFunc, in-place addition
# Input:
- self::ScalarFunc: first ScalarFunc object, modified in-place
- other::ScalarFunc: second ScalarFunc object
# Modifies:
- self.space_func and self.freq_func are modified by adding corresponding fields of other.
    If the shapes of freq_func do not match, they are adjusted accordingly by zero-padding.
# Return:
- self::ScalarFunc: the modified first ScalarFunc object
"""
function add_equal!(self::ScalarFunc, other::ScalarFunc)
    # space_func part
    if self.space_func !== nothing && other.space_func !== nothing
        self.space_func .+= other.space_func
    end

    # freq_func part
    if self.freq_func !== nothing && other.freq_func !== nothing
        m1, n1 = size(self.freq_func)
        m2, n2 = size(other.freq_func)

        if m1 == m2 && n1 == n2
            # same size
            self.freq_func .+= other.freq_func
        elseif m1 > m2 && n1 > n2
            # self bigger: only add into the matching subblock
            self.freq_func[1:m2, 1:n2] .+= other.freq_func
        else
            # need a larger canvas
            m = max(m1, m2)
            n = max(n1, n2)
            result = zeros(dtype, m, n)

            # copy both into result
            result[1:m1, 1:n1] .+= self.freq_func
            result[1:m2, 1:n2] .+= other.freq_func

            # replace
            self.freq_func = result
        end
    end

    return self
end

"""
Add two ScalarFunc objects.
# Input:
- self::ScalarFunc: first ScalarFunc object
- other::ScalarFunc: second ScalarFunc object
# Return:
- A new ScalarFunc object which is the sum of self and other
"""
function +(self::ScalarFunc, other::ScalarFunc)
    result = copy(self); add_equal!(result, other); return result
end

"""
multiply_equal!: *= operator for ScalarFunc, in-place multiplication
# Input:
- self::ScalarFunc: ScalarFunc object, modified in-place
- s::Real: scalar multiplier. 
# Modifies:
- self.space_func and self.freq_func are scaled by s if they are not `nothing`
# Return:
- self::ScalarFunc: the modified ScalarFunc object
"""
function multiply_equal!(self::ScalarFunc, s::Real)
    self.space_func !== nothing && (self.space_func .*= s)
    self.freq_func !== nothing && (self.freq_func .*= s)
    return self
end

"""
*: scalar multiplication for ScalarFunc, returns a new object
# Input:
- self::ScalarFunc: input ScalarFunc object
- s::Real: scalar multiplier
# Return:
- A new ScalarFunc object scaled by s
"""
function *(self::ScalarFunc, s::Real)
    result = copy(self); multiply_equal!(result, s); return result
end

"""
minus_equal!: -= operator for ScalarFunc, in-place subtraction
# Input:
- self::ScalarFunc: first ScalarFunc object, modified in-place
- other::ScalarFunc: second ScalarFunc object
# Modifies:
- self::ScalarFunc by subtracting other
# Return:
- self::ScalarFunc: the modified ScalarFunc object
"""
function minus_equal!(self::ScalarFunc, other::ScalarFunc)
    add_equal!(self, other * dt(-1))
    return self
end

"""
-: subtraction for ScalarFunc
# Input:
- self::ScalarFunc: first ScalarFunc object
- other::ScalarFunc: second ScalarFunc object
# Return:
- A new ScalarFunc object equal to self - other
"""
function -(self::ScalarFunc, other::ScalarFunc)
    return self + other * dt(-1)
end

"""
divide_equal!: /= operator for ScalarFunc, in-place division
# Input:
- self::ScalarFunc: ScalarFunc object, modified in-place
- s::Real: scalar divisor (nonzero)
# Modifies:
- self::ScalarFunc by dividing space_func and freq_func by s
# Return:
- self::ScalarFunc: the modified ScalarFunc object
# Throws:
- AssertionError if s == 0
"""
function divide_equal!(self::ScalarFunc, s::Real)
    @assert s != 0 "Division by zero"
    multiply_equal!(self, inv(dt(s)))
    return self
end

"""
/: scalar division for ScalarFunc
# Input:
- self::ScalarFunc: input ScalarFunc object
- s::Real: scalar divisor (nonzero)
# Return:
- A new ScalarFunc object equal to self / s
# Throws:
- AssertionError if s == 0
"""
function /(self::ScalarFunc, s::Real)
    @assert s != 0 "Division by zero"
    return self * inv(dt(s))
end