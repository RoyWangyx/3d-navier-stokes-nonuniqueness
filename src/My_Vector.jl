# These are all the boundary condition options we need in this project. Every entry has
# four lists, representing the boundary conditions for u1, u2, u3, p respectively. Every
# list has two entries, representing the boundary conditions at r and θ respectively. For
# the eigenvector, due to its odd symmetry, we impose different boundary conditions at r=0
# for the first spherical harmonic mode (l=1) and the rest (l≥2). The first entry in the list
# is for l=1, the second entry is for l≥2.
const BDRY_LST_SET = [
    [[["10", "00"], "10"], [["10", "00"], "01"], ["00", "00"], [["10", "00"], "10"]],
    [["01", "11"], ["01", "00"], ["01", "01"], ["10", "11"]],
    [["00", "11"], ["00", "00"], ["00", "01"], ["10", "11"]]
]

"""
    Mutable struct VectorFunc
A structure to hold multiple ScalarFunc objects, representing a vector-valued function.
- keys: Vector of names (String)
- dict: Dictionary mapping keys to ScalarFunc objects
- scl_fac: Scaling factor (dtype), which is passed to each ScalarFunc
- bdry_index: index of boundary condition in BDRY_LST_SET. If nothing, no boundary conditions are set.
"""
mutable struct VectorFunc
    keys::Vector{String}
    dict::Dict{String, ScalarFunc}
    scl_fac::dtype
    bdry_index::Union{Int, Nothing}
end

"""
Constructor for VectorFunc:
# Input:
- keys: Vector of names
- bdry_index: the index of boundary condition list (optional); default is nothing. If
    bdry_index is provided, each ScalarFunc is initialized with the corresponding boundary
    condition in BDRY_LST_SET[bdry_index].
- scl_fac: scaling factor (dtype), which is passed to each ScalarFunc; default is 1.0.
# Return:
- VectorFunc object with specified keys and boundary conditions.
"""
function VectorFunc(keys::Vector{String}; bdry_index::Union{Int, Nothing}=nothing, scl_fac::dtype=1.0)
    dict = Dict{String, ScalarFunc}()
    if bdry_index !== nothing
        @assert 1 <= bdry_index <= length(BDRY_LST_SET) "Invalid bdry_index"
        @assert length(BDRY_LST_SET[bdry_index]) == length(keys) "Length of bdry_lst does not match number of keys"
    end
    for (i, key) in enumerate(keys)
        bdry = (bdry_index === nothing) ? nothing : BDRY_LST_SET[bdry_index][i]
        dict[key] = ScalarFunc(key; bdry=bdry, scl_fac=scl_fac)
    end
    return VectorFunc(keys, dict, scl_fac, bdry_index)
end

"""
Another constructor for `VectorFunc` where the keys are inferred from keyword arguments.
Each keyword corresponds to a function name, and its value must be an `AbstractMatrix{dtype}`
used to construct the associated `ScalarFunc`. This form is convenient for creating a 
`VectorFunc` with arbitrary keys. Boundary conditions can optionally be assigned.
# Input:
- `scl_fac::dtype` : Scaling factor, passed to each `ScalarFunc`.
- `domain::String` : must be exactly "space" or "frequency" (passed to `set_func!`).
- `bdry_index::Union{Int,Nothing}` : Optional index into `BDRY_LST_SET`; overrides `bdry_lst` if provided.
- `bdry_lst::Union{Vector,Nothing}` : Optional explicit list of boundary conditions (one per key).
- `kwargs...` : Keyword arguments, each key is converted to `String` and each value must be an `AbstractMatrix{dtype}`.
# Returns
- A `VectorFunc` object with keys from `kwargs`, functions set according to `domain`,
  and boundary conditions set if provided.
"""
function VectorFunc(scl_fac::dtype, domain::String; bdry_index::Union{Int, Nothing}=nothing, 
    bdry_lst::Union{Vector, Nothing}=nothing, kwargs...)
    key_lst = string.(collect(keys(kwargs)))
    result = VectorFunc(key_lst; scl_fac=scl_fac)
    for (key, value) in kwargs
        set_func!(result[string(key)], domain, value)
    end
    if bdry_index !== nothing
        result.bdry_index = bdry_index
        bdry_lst = BDRY_LST_SET[bdry_index]
    end
    if bdry_lst !== nothing
        @assert length(bdry_lst) == length(result.keys) "Length of bdry_lst does not match number of keys"
        for (i, key) in enumerate(result.keys)
            set_bdry!(result.dict[key], bdry_lst[i])
        end
    end
    return result
end

"""
Load a VectorFunc from a .mat file. The keys are assumed to be ["u1", "u2", "u3", "p"].
Usually used to load a solution (velocity and pressure).
# Input:
- filename: path to the .mat file
- bdry_index: index of boundary condition in BDRY_LST_SET. Must be provided.
- domain: "space" or "frequency" (default: "frequency")
# Return:
- VectorFunc object with data loaded from the .mat file
"""
function from_mat_file(filename::String, bdry_index::Int; domain::String="frequency")
    u1, u2, u3, p, scl_fac, shape = read_matrix(filename, ["u1", "u2", "u3", "p", "scl_fac", "shape"])
    N0, N1 = dtype == Float64 ? Int64.(shape) : Int64.(inf.(shape))

    up = VectorFunc(scl_fac, domain; bdry_index=bdry_index, u1=u1, u2=u2, u3=u3, p=p)

    # u1 and u2 are the radial and angular components, need to transform to spherical harmonic basis
    set_conv_mat!(up["u1"])
    set_conv_mat!(up["u2"])

    transform!(up, domain; shape=(N0, N1))
    set_div_free!(up)  # impose divergence-free condition

    try
        lambda = read_matrix(filename, ["lambda"])
        return up, lambda[1]
    catch
        return up
    end
end

"""
Load gradient operator Q from a `.mat` file.
The file must contain keys `"A11","A12","A13","A22","A23","A33","shape","scl_fac"`.
# Input
- `file_path::String` : path to a `.mat` file.
# Return
- `Q::VectorFunc` : gradient operator after `transform!`.
"""
function load_grad_U(file_path::String)
    A11, A12, A13, A22, A23, A33, R_max, shape = read_matrix(
        file_path, ["A11","A12","A13","A22","A23","A33","scl_fac","shape"])
    N0, N1 = dtype == Float64 ? Int64.(shape) : Int64.(inf.(shape))
    bdry_lst = [["10", "11"], ["10", "00"], ["00", "01"], ["10", "11"], ["00", "00"], ["00", "11"]]

    Q = VectorFunc(R_max, "frequency"; bdry_lst=bdry_lst,
            A11=A11, A12=A12, A13=A13, A22=A22, A23=A23, A33=A33)
    transform!(Q, "frequency"; shape=(N0, N1))

    return Q
end

"""
Reload getindex for VectorFunc to access ScalarFunc objects by key or index.
# Input:
- self::VectorFunc: the vector function container
- key::Union{Int,String}: position in `self.keys` or the key string
# Return:
- ScalarFunc object corresponding to the given key
# Throws AssertionError if the key is invalid
"""
function getindex(self::VectorFunc, key::Union{Int,String})
    isa(key, Int) && (key = self.keys[key])
    @assert key in self.keys "Invalid key"
    return self.dict[key]
end

"""
Reload setindex! for VectorFunc to assign ScalarFunc objects by key or index.
# Input:
- self::VectorFunc: the vector function container
- value::ScalarFunc: the new function to assign
- key::Union{Int,String}: position in `self.keys` or the key string
# Modifies:
- Updates self.dict[key] with the new ScalarFunc
# Throws AssertionError if the key is invalid
"""
function setindex!(self::VectorFunc, value::ScalarFunc, key::Union{Int,String})
    isa(key, Int) && (key = self.keys[key])
    @assert key in self.keys "Invalid key"
    self.dict[key] = value
end

"""
Reload transform! of ScalarFunc. Transform all ScalarFunc entries in a VectorFunc to the
specified domain.
# Input:
- self::VectorFunc: the vector function container
- domain::String: target domain, "space" or "frequency"
- shape: shape of matrix after transformation (default: nothing). Needed if space_func is 
    not set.
# Modifies:
- Each ScalarFunc in self.dict is transformed in place, with conv_flag=true for keys "u1"
    and "u2"
# Return:
- self::VectorFunc: the transformed container
"""
function transform!(self::VectorFunc, domain::String; shape=nothing)
    for key in self.keys
        # u1 and u2 are the radial and angular components, need to transform
        # to spherical harmonic basis
        conv_flag = key in ["u1", "u2"]
        transform!(self.dict[key], domain; shape=shape, conv_flag=conv_flag)
    end
    return self
end

"""
Interpolate all ScalarFunc entries in a VectorFunc onto the given radial and angular points.
This is a thin wrapper over `ScalarFunc.interpolate`. It constructs a new `VectorFunc` and 
stores the interpolated arrays in each entry's `space_func` field.

# Input:
- self::VectorFunc: the vector function container
- r_pt::Vector{dtype}: interpolation points in the radial direction
- θ_pt::Vector{dtype}: interpolation points in the angular direction
- keys::Vector{String}=["I","I"]: operator keys for [radial, angular] directions
- pt_flag::Bool=false: if true, the stored arrays correspond to the intermediate result after
    the radial transform and before the angular transform (i.e., not the full space-domain
    evaluation). For convenience, they are still written into `space_func`.

# Return:
- result::VectorFunc: a new container with the same keys as `self`, where each
  `result[key].space_func` stores the evaluated/intermediate array on the (r_pt, θ_pt) grid.

# Throws:
- Error if any underlying `ScalarFunc` has no `freq_func` set.
"""
function interpolate(self::VectorFunc, r_pt::Vector{dtype}, θ_pt::Vector{dtype}; keys::Vector{String}=["I","I"], pt_flag=false)
    result = VectorFunc(self.keys; scl_fac=self.scl_fac)
    for key in self.keys
        set_func!(result.dict[key], "space", interpolate(self[key], r_pt, θ_pt; keys=keys, pt_flag=pt_flag))
        if keys == ["I","I"]
            set_bdry!(result.dict[key], self.dict[key].bdry)
        end
    end
    return result
end

"""
Set the VectorFunc to be divergence-free by adjusting the u2 component.
# Input:
- self::VectorFunc: the vector function container to be modified
- print_flag::Bool: whether to print the L∞-norm of the difference between
    original u2 and the corrected u2 (default: true)
# Modifies:
- Updates self["u2"] to enforce the divergence-free condition
# Return:
- result_lst: a list elements are coefficient matrices representing the linear
    relations satisfied by u1 and u2
"""
function set_div_free!(self::VectorFunc; print_flag::Bool=false)
    u1, u2 = self["u1"].freq_func, self["u2"].freq_func
    @assert self["u1"].bdry !== nothing "Boundary conditions must be set to enforce divergence-free condition"
    bdry = self["u1"].bdry[1]
    bdry = isa(bdry, String) ? [bdry] : reverse(bdry)
    result = calc_Utheta_divfree(u1, bdry)
    u2_tld = result[end]# The last entry is the new u2
    set_func!(self["u2"], "frequency", u2_tld)
    print_flag && println("u2 difference: ", norm(vec(u2 - u2_tld), Inf))
    return front(result) # Exclude the last entry, which is u2_tld
end


"""
Compute the inner product of two VectorFunc objects on the spatial grid.
This routine sums component-wise inner products of the underlying `ScalarFunc`s
(using their `.space_func` arrays) and applies the global scaling factor `self.scl_fac`.
Numerical quadrature uses the same Newton–Cotes rule as `ScalarFunc.inner_product`.

Optionally, if `W_norms_lst` is provided, an additional Newton–Cotes quadrature error
bound is computed (with a fixed order used internally) and, when `dtype == Interval{Float64}`,
the returned value is wrapped by an interval hull.

# Input:
- self::VectorFunc: first operand (requires `.space_func` to be set for each component)
- other::VectorFunc: second operand (requires `.space_func` to be set for each component)
- W_norms_lst=nothing: optional data used to bound quadrature error and produce an interval hull
- domain::Vector{dtype}=[Pi/dt(2), Pi/dt(2)]: physical domain lengths [L0, L1] used by quadrature

# Return:
- result::dtype: the inner product value (or an interval hull if `dtype == Interval{Float64}` and
  `W_norms_lst !== nothing`)

# Throws:
- AssertionError if component keys do not match.
- Errors from `ScalarFunc.inner_product` if required fields are missing.
"""
function inner_product(self::VectorFunc, other::VectorFunc; W_norms_lst=nothing, domain::Vector{dtype}=[Pi/dt(2), Pi/dt(2)])
    result = zero(dtype)
    k = 8
    @assert all(self.keys == other.keys) "Keys do not match for space domain"
    for i in 1:length(self.keys)
        result += inner_product(self[i], other[i]; domain=domain)
    end
    if W_norms_lst !== nothing
        w_bds = zeros(dtype, k + 1, 2)
        w_bds[:, 2] .= one(dtype)
        w_bds[1, 1] = one(dtype)
        W_normsT = [W' for W in W_norms_lst[1]]
        total_W_norm = multinomial_sum(k, w_bds, W_normsT, W_norms_lst[2])

        L0, L1 = domain
        N0, N1 = size(self[1].space_func) .- 1
        h0, h1 = L0 / dt(N0), L1 / dt(N1)
        err = (total_W_norm[1] * h0^dt(k) + total_W_norm[2] * h1^dt(k)) * dt(3) / dt(2800) * L0 * L1 * self.scl_fac
        if dtype == Interval{Float64}
            result = hull(result - err, result + err)
        end
    end
    return result  
end

"""
Reload my_plot for VectorFunc. Plot all ScalarFunc entries in a VectorFunc.
# Input:
- self::VectorFunc: the vector function container
# Throws AssertionError if self.keys is empty. No return value.
"""
function my_plot(self::VectorFunc)
    @assert length(self.keys) > 0 "No keys in solution"
    for k in self.keys
        my_plot(self.dict[k])
    end
end

"""
Return the L∞ and L2-norm of each ScalarFunc entry.
# Input:
- self::VectorFunc: the vector function container
# Return:
- The L-∞-norms and L2 norm are returned as a list.
"""
function norm_lst(self::VectorFunc, mode::String; paras=nothing, domain::Vector{dtype}=[Pi/dt(2), Pi/dt(2)])
    if mode == "L2"
        W_norms_lst = paras === nothing ? nothing : [paras, paras]
        return sqrt(inner_product(self, self; W_norms_lst=W_norms_lst, domain=domain))
    elseif mode == "LInf"
        l = length(self.keys)
        results = dtype[]
        if paras === nothing
            paras = [nothing, nothing]
        else
            M0, M1 = paras
            paras = [["freq", M0], ["freq", M1]]
        end
        for i in 1:l
            push!(results, get_Wk∞_norm_lst(self[i], 0; paras_lst=paras, L=domain))
        end
        return results
    else
        @assert mode[1] == 'W' && mode[end-2:end] == "Inf" "Invalid norm mode"
        k = parse(Int, mode[2:end-3])
        l = length(self.keys)

        if isa(paras, Vector{dtype})
            L∞_norms = norm_lst(self, "LInf"; paras=paras, domain=domain)
            return [L∞_norms .* paras[j]^dt(i) for i in 0:k, j in 1:2]
        end

        Wk∞_norms_lst = []
        for i in 1:l
            push!(Wk∞_norms_lst, get_Wk∞_norm_lst(self[i], k; paras_lst=paras[i, :], L=domain))
        end
        results = [zeros(dtype, l) for _ in 1:k+1, _ in 1:2]

        for i in 1:l, j in 1:2, m in 0:k
            results[m+1, j][i] = Wk∞_norms_lst[i][m+1, j]
        end
        return results
    end
end

function trim_border!(self::VectorFunc, num::Union{Int, Tuple{Int,Int}})
    for k in self.keys
        trim_border!(self.dict[k], num)
    end
    return self
end


"""
Save a VectorFunc to a .mat file.
# Input:
- self::VectorFunc: the vector function container
- filename::String: path to the .mat file to be written
- shape::Union{Tuple{Int,Int}, Nothing}: shape to be saved (optional and default 
    is nothing). If not provided, it is inferred from the first entry's space_func.
# Return:
- Dict{String,Any}: dictionary of data written to the .mat file
# Throws AssertionError if shape is not provided and no valid space_func exists
"""
function save(self::VectorFunc, filename::String; shape::Union{Tuple{Int,Int}, Nothing}=nothing, lambda::Union{dtype, Nothing}=nothing)
    data = Dict{String,Any}()
    for k in self.keys
        scalar = self.dict[k]
        data[k] = scalar.freq_func
        scalar.conv_mat !== nothing && (data[k * "_conv_mat"] = scalar.conv_mat)
    end
    data["scl_fac"] = self.scl_fac
    if shape === nothing
        @assert !isempty(self.keys) && self[1].space_func !==
            nothing "No keys in solution or no space function available"
        N0, N1 = size(self[1].space_func)
        data["shape"] = [N0, N1]
    end
    lambda !== nothing && (data["lambda"] = lambda)
    matwrite(filename, data)
    return data
end

"""
Reload copy for VectorFunc. Create a deep copy of a VectorFunc, optionally restricted
to a subset of keys.
# Input:
- self::VectorFunc: the vector function container
- keys: optional subset specification (Int, String, Vector, UnitRange, or nothing).
    If nothing, all keys are copied. Default is nothing.
# Return:
- new: a new VectorFunc containing copies of the selected entries
# Throws AssertionError if a provided key does not exist
"""
function copy(self::VectorFunc; keys=nothing)
    copied_keys = keys === nothing ? copy(self.keys) : begin
        if isa(keys, Int)
            [self.keys[keys]]
        elseif isa(keys, String)
            @assert keys in self.keys "Key not found in solution"
            [keys]
        elseif isa(keys, Vector)
            [isa(k, Int) ? self.keys[k] : k for k in keys]
        elseif isa(keys, UnitRange)
            self.keys[keys]
        else
            error("Invalid keys argument")
        end
    end
    new = VectorFunc(copied_keys; bdry_index=self.bdry_index, scl_fac=self.scl_fac)
    for k in copied_keys
        new[k] = copy(self.dict[k])
    end
    return new
end

"""
Reload add_equal!(+=) for VectorFunc. Perform in-place addition with another VectorFunc.
# Input:
- self::VectorFunc: the vector function container to be modified
- other::VectorFunc: the vector function container to add
# Modifies:
- Updates self by adding ScalarFunc entries from other, creating new keys if needed
# Return:
- self::VectorFunc: the modified container
"""
function add_equal!(self::VectorFunc, other::VectorFunc)
    for k in self.keys
        if haskey(other.dict, k)
            add_equal!(self.dict[k], other.dict[k])
        end
    end
    for k in other.keys
        if !(k in self.keys)
            push!(self.keys, k)
            self.dict[k] = copy(other.dict[k])
        end
    end
    return self
end

"""
Reload + for VectorFunc. Return a new VectorFunc equal to the sum of two VectorFunc objects.
# Input:
- self::VectorFunc: first operand
- other::VectorFunc: second operand
# Return:
- A new VectorFunc representing self + other
"""
function +(self::VectorFunc, other::VectorFunc)
    result = copy(self)
    add_equal!(result, other)
    return result
end

"""
Reload multiply_equal!(*=) for VectorFunc. Perform in-place scalar multiplication.
# Input:
- self::VectorFunc: the vector function container to be modified
- scalar::Union{dtype,Int}: scalar multiplier
# Modifies:
- Scales all ScalarFunc entries in self by scalar
# Return:
- self::VectorFunc: the modified container
"""
function multiply_equal!(self::VectorFunc, scalar::Union{dtype,Int})
    for k in keys(self.dict)
        multiply_equal!(self.dict[k], scalar)
    end
    return self
end

"""
Reload * for VectorFunc. Return a new VectorFunc scaled by a scalar.
# Input:
- self::VectorFunc: the vector function container
- scalar::Union{dtype,Int}: scalar multiplier
# Return:
- A new VectorFunc equal to self * scalar
"""
function *(self::VectorFunc, scalar::Union{dtype,Int})
    result = copy(self)
    multiply_equal!(result, scalar)
    return result
end

"""
Reload minus_equal!(-=) for VectorFunc. Perform in-place subtraction with another VectorFunc.
# Input:
- self::VectorFunc: the vector function container to be modified
- other::VectorFunc: the vector function container to subtract
# Modifies:
- Updates self by subtracting ScalarFunc entries of other
# Return:
- self::VectorFunc: the modified container
"""
function minus_equal!(self::VectorFunc, other::VectorFunc)
    add_equal!(self, other * dt(-1))
    return self
end

"""
Reload - for VectorFunc. Return a new VectorFunc equal to the difference of two VectorFunc objects.
# Input:
- self::VectorFunc: first operand
- other::VectorFunc: second operand
# Return:
- A new VectorFunc representing self - other
"""
function -(self::VectorFunc, other::VectorFunc)
    return self + other * dt(-1)
end

"""
Reload divide_equal!(/=) for VectorFunc. Perform in-place division by a scalar.
# Input:
- self::VectorFunc: the vector function container to be modified
- scalar::Union{dtype,Int}: scalar divisor (nonzero)
# Modifies:
- Scales all ScalarFunc entries in self by 1/scalar
# Return:
- self::VectorFunc: the modified container
# Throws AssertionError if scalar == 0
"""
function divide_equal!(self::VectorFunc, scalar::Union{dtype,Int})
    @assert scalar != 0 "Division by zero"
    multiply_equal!(self, inv(scalar))
    return self
end

"""
Reload / for VectorFunc. Return a new VectorFunc divided by a scalar.
# Input:
- self::VectorFunc: the vector function container
- scalar::Union{dtype,Int}: scalar divisor (nonzero)
# Return:
- A new VectorFunc representing self / scalar
# Throws AssertionError if scalar == 0
"""
function /(self::VectorFunc, scalar::Union{dtype,Int})
    @assert scalar != 0 "Division by zero"
    return self * inv(scalar)
end