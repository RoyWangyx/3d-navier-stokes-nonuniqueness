"""
    mutable struct Operator
Operators on scalar functions.
A scalar function is represented in a Fourier basis. The operator acts two-sided:
`L` on the radial variable and `R` on the angular variable, with an optional
pointwise multiplier `T` in the spatial domain and an alternative left matrix
`L1` applied to the first column due to the different boundary condition.
# Fields:
- L::AbstractMatrix{dtype}: left matrix
- R::AbstractMatrix{dtype}: right matrix
- T::Union{AbstractMatrix{dtype}, Nothing}: optional pointwise multiplier matrix
- L1::Union{AbstractMatrix{dtype}, Nothing}: optional alternative left matrix for first column
"""
mutable struct Operator
    L::AbstractMatrix{dtype}                 
    R::AbstractMatrix{dtype}                 
    T::Union{AbstractMatrix{dtype}, Nothing}          
    L1::Union{AbstractMatrix{dtype}, Nothing}           
end

"""
Outer constructor for Operator with keyword arguments for optional fields.
# Input:
- L::AbstractMatrix{dtype}: left matrix
- R::AbstractMatrix{dtype}: right matrix
- T::Union{AbstractMatrix{dtype}, Nothing}=nothing: optional pointwise multiplier
- L1::Union{AbstractMatrix{dtype}, Nothing}=nothing: optional left matrix for first column
# Return:
- op::Operator: constructed operator
"""
function Operator(L::AbstractMatrix{dtype}, R::AbstractMatrix{dtype}; 
    T::Union{AbstractMatrix{dtype}, Nothing}=nothing, L1::Union{AbstractMatrix{dtype}, Nothing}=nothing)
    op = Operator(L, R, T, L1)
    return op
end

"""
In-place scalar (*=) multiplication on Operator. We only scale the right matrix R.
# Input:
- op::Operator: operator to be modified
- scalar::Union{dtype,Int}: multiplier
# Modifies:
- Scales op.R in place by scalar
# Return:
- op::Operator: the modified operator
"""
function multiply_equal!(op::Operator, scalar::Union{dtype,Int})
    op.R .*= scalar
    return op
end

"""
Scalar multiplication (*) of Operator (returns a new Operator).
# Input:
- op::Operator: operator to scale
- scalar::Union{dtype,Int}: multiplier
# Return:
- result::Operator: a new operator with R scaled by scalar
"""
function *(op::Operator, scalar::Union{dtype,Int})
    result = copy(op)
    multiply_equal!(result, scalar)
    return result
end

"""
Set the T matrix of an Operator.
# Input:
- op::Operator: operator to modify
- T::AbstractMatrix{dtype}: pointwise multiplier matrix
# Modifies:
- Sets op.T = T
# No return value; modifies in place
"""
set_T!(op::Operator, T::AbstractMatrix{dtype}) = (op.T = T; nothing)

"""
Set the L1 matrix of an Operator.
# Input:
- op::Operator: operator to modify
- L1::AbstractMatrix{dtype}: left matrix for the first column block
# Modifies:
- Sets op.L1 = L1
# No return value; modifies in place
"""
set_L1!(op::Operator, L1::AbstractMatrix{dtype}) = (op.L1 = L1; nothing)

"""
Apply an Operator to a matrix by first multiplying on the left with L (replacing the
first column with L1*mat[:,1] if L1 is provided), then multiplying on the right with R,
and finally applying pointwise multiplier with T if present.
# Input:
- op::Operator: operator with L, R, and optional T and L1
- mat::AbstractMatrix{dtype}: input matrix
# Return:
- result::AbstractMatrix{dtype}: the transformed matrix
# Throws AssertionError if dimensions of L and R do not match mat
"""
function apply(op::Operator, mat::AbstractMatrix{dtype})
    @assert size(op.L, 2) == size(mat, 1) "Left matrix size mismatch: $(size(op.L, 2)) vs $(size(mat, 1))"
    @assert size(op.R, 1) == size(mat, 2) "Right matrix size mismatch: $(size(op.R, 1)) vs $(size(mat, 2))"

    result = op.L * mat
    op.L1 !== nothing && (result[:,1] .= op.L1 * mat[:,1])
    result *= op.R
    op.T  !== nothing && (result .*= op.T)

    return result
end

"""
Change variables of an Operator by composing new transformation matrices.
# Input:
- op::Operator: operator to be modified
- L::Union{AbstractMatrix{dtype}, Nothing}=nothing: if provided, post-multiplies the
    existing op.L (op.L = op.L * L)
- R::Union{AbstractMatrix{dtype}, Nothing}=nothing: if provided, pre-multiplies the
    existing op.R (op.R = R * op.R)
- L1::Union{AbstractMatrix{dtype}, Nothing}=nothing: if provided, post-multiplies the
    existing op.L1 (op.L1 = op.L1 * L1)
# Modifies:
- Updates op.L, op.R, and op.L1 in place if corresponding matrices are provided
# Return:
- op::Operator: the modified operator with updated matrices
"""
function change_variable!(op::Operator; L::Union{AbstractMatrix{dtype}, Nothing}=nothing,
    R::Union{AbstractMatrix{dtype}, Nothing}=nothing, L1::Union{AbstractMatrix{dtype}, Nothing}=nothing)
    L  !== nothing && (op.L  *= L)
    L1 !== nothing && (op.L1 *= L1)
    R  !== nothing && (op.R   =  R * op.R)
    return op
end

"""
Create a deep copy of an Operator, including optional fields.
Copies L and R, and if present also copies T and L1.
# Input:
- op::Operator: operator to copy
# Return:
- new::Operator: a new operator with copied fields
"""
function copy(op::Operator)
    new = Operator(copy(op.L), copy(op.R))
    op.T  !== nothing && set_T!(new,  copy(op.T))
    op.L1 !== nothing && set_L1!(new, copy(op.L1))
    return new
end

"""
    mutable struct Operator_matrix
Operator_matrix on vector functions. A block-structured container whose entry (i,j) stores a list of 
`Operator`s mapping the j-th input component to the i-th output component. It supports addition, scaling, 
application, and **change of variables** utilities.
# Fields:
- M::Int: number of rows
- N::Int: number of columns
- op_lst::Matrix{Vector{Operator}}: op_lst[i,j] holds a list of Operators applied from input j to output i
"""
mutable struct Operator_matrix
    M::Int
    N::Int
    op_lst::AbstractMatrix{Vector{Operator}}
end


"""
Construct an Operator_matrix of size M×N with empty operator lists in each entry.
# Input:
- M::Int: number of rows
- N::Int: number of columns
# Return:
- Operator_matrix: initialized with Operator[] in every entry
"""
function Operator_matrix(M::Int, N::Int)
    op_lst = [Operator[] for i in 1:M, j in 1:N]
    return Operator_matrix(M, N, op_lst)
end

"""
Reload getindex for Operator_matrix to slice rows/columns.
# Input:
- mat::Operator_matrix
- rows::Union{Int,UnitRange}: selected row(s)
- cols::Union{Int,UnitRange}: selected column(s)
# Return:
- Operator_matrix: new container with the selected subgrid (shares inner operator lists)
"""
function getindex(mat::Operator_matrix, rows::Union{Int,UnitRange}, cols::Union{Int,UnitRange})
    new_op_lst = mat.op_lst[rows, cols]
    return Operator_matrix(length(rows), length(cols), new_op_lst)
end

"""
Reload setindex! for Operator_matrix to assign the operator list at (i,j).
# Input:
- mat::Operator_matrix
- ops::Vector{Operator}: operators to store
- i::Int: row index
- j::Int: column index
# Modifies:
- Replaces mat.op_lst[i,j] with ops
"""
function setindex!(mat::Operator_matrix, ops::Vector{Operator}, i::Int, j::Int)
    mat.op_lst[i, j] = ops
end

"""
In-place scalar multiplication (*=) of an Operator_matrix.
# Input:
- mat::Operator_matrix
- scalar::Union{dtype,Int}
# Modifies:
- Multiplies every Operator in mat.op_lst by scalar in place
# Return:
- mat::Operator_matrix: the modified container
"""
function multiply_equal!(mat::Operator_matrix, scalar::Union{dtype,Int})
    for i in 1:mat.M, j in 1:mat.N, op in mat.op_lst[i, j]
        multiply_equal!(op, scalar)
    end
    return mat
end

"""
Scalar multiplication (*) of an Operator_matrix (returns new).
# Input:
- mat::Operator_matrix
- scalar::Union{dtype,Int}
# Return:
- Operator_matrix: copy of mat with every Operator scaled by scalar
"""
function *(mat::Operator_matrix, scalar::Union{dtype,Int})
    mat2 = copy(mat)
    multiply_equal!(mat2, scalar)
    return mat2
end

"""
Reload + for Operator_matrix. Combine two operator grids (sizes may differ).
# Input:
- A::Operator_matrix
- B::Operator_matrix
# Return:
- Operator_matrix: size max(A.M,B.M)×max(A.N,B.N), with element-wise concatenation 
    of operator lists
"""
function +(A::Operator_matrix, B::Operator_matrix)
    # determine the new size
    M = max(A.M, B.M)
    N = max(A.N, B.N)
    C = Operator_matrix(M, N)
    # add operators from A and B into C
    for i in 1:M, j in 1:N
        if i <= A.M && j <= A.N
            for op in A.op_lst[i, j]
                push!(C.op_lst[i, j], copy(op))
            end
        end
        if i <= B.M && j <= B.N
            for op in B.op_lst[i, j]
                push!(C.op_lst[i, j], copy(op))
            end
        end
    end
    return C
end

"""
Reload - for Operator_matrix. Subtract two operator grids.
# Input:
- A::Operator_matrix
- B::Operator_matrix
# Return:
- Operator_matrix: result of A + (-1)*B
"""
function -(A::Operator_matrix, B::Operator_matrix)
    return A + B * dt(-1)
end

"""
Add one or more Operators into op_lst[i, j].
# Input:
- mat::Operator_matrix
- i::Int: row index
- j::Int: column index
- ops::Operator...: one or more operators to append
# Modifies:
- Appends ops to mat.op_lst[i,j]
# Return:
- mat::Operator_matrix: the modified container
"""
function add_operator!(mat::Operator_matrix, i::Int, j::Int, ops::Operator...)
    append!(mat.op_lst[i, j], ops)
    return mat
end

"""
Apply an Operator_matrix to a VectorFunc: for each output row i, sum over columns j of
apply(op, vec[j].freq_func) for all op in op_lst[i,j], then set the resulting field in space domain.
# Input:
- mat::Operator_matrix
- vec::VectorFunc: input fields; its length (keys) must equal N
- new_keys::Vector{String}=["e1","e2","e3"]: output keys; its length must equal M
# Return:
- VectorFunc: result with keys=new_keys and space-domain fields set from the accumulations
# Throws AssertionError if sizes of vec or new_keys do not match the operator grid
"""
function apply(mat::Operator_matrix, vec::VectorFunc; new_keys::Vector{String}=["e1","e2","e3"])
    M, N = mat.M, mat.N
    @assert length(vec.keys) == N "Input size mismatch"
    @assert length(new_keys) == M "Output keys mismatch"

    # Determine the output size m and input size n by checking the operators in mat
    # We assume all operators have the same output size m * n.
    m, n = 0, 0
    for i in 1:M, j in 1:N
        for op in mat.op_lst[i, j]
            if op.L !== nothing
                if m == 0
                    m = size(op.L, 1)
                else
                    @assert m == size(op.L, 1) "Inconsistent output size in L across operators"
                end
            end
            if op.L1 !== nothing
                @assert m == size(op.L1, 1) "Inconsistent output size in L1 across operators"
            end
            if op.R !== nothing
                if n == 0
                    n = size(op.R, 2)
                else
                    @assert n == size(op.R, 2) "Inconsistent input size in R across operators"
                end
            end
        end
    end

    result = VectorFunc(new_keys; scl_fac=vec.scl_fac)
    for i in 1:M
        # Initialize cumulative result if nothing was accumulated
        cum = zeros(dtype, m, n)
        for j in 1:N
            for op in mat.op_lst[i, j]
                cum .+= apply(op, vec[j].freq_func)
            end
        end
        set_func!(result[new_keys[i]], "space", cum)
    end
    return result
end

"""
Reload copy for Operator_matrix. Deep-copy all Operators into a new container.
# Input:
- mat::Operator_matrix
# Return:
- new::Operator_matrix: same size as mat with copied operators in each entry
"""
function copy(mat::Operator_matrix)
    new = Operator_matrix(mat.M, mat.N)
    for i in 1:mat.M, j in 1:mat.N
        for op in mat.op_lst[i, j]
            add_operator!(new, i, j, copy(op))
        end
    end
    return new
end

"""
Compose right-side changes using each input field's conv_mat where available.
For every op in op_lst[i,j], if vec[j].conv_mat exists, update op.R ← vec[j].conv_mat * op.R.
# Input:
- mat::Operator_matrix
- vec::VectorFunc: provides conv_mat per input key (column j)
# Modifies:
- Each operator in column j with non-nothing conv_mat is updated in place
# Return:
- mat::Operator_matrix: the modified container
"""
function apply_convmat_right!(mat::Operator_matrix, vec::VectorFunc)
    for j in 1:mat.N, i in 1:mat.M
        if vec[j].conv_mat !== nothing
            for op in mat.op_lst[i, j]
                change_variable!(op; L=nothing, R=vec[j].conv_mat, L1=nothing)
            end
        end
    end
    return mat
end

"""
Compute eigenvalues of a symmetric 3×3 tensor field at each grid point.

Input:
- `eigen_mat::VectorFunc`: keys "A11","A12","A13","A22","A23","A33",
  each provides `.space_func` (N0×N1) as the corresponding tensor component.

Returns:
- `eig_val_mat::Array{dtype,3}` (N0×N1×3): eigenvalues at each grid point
  in the order returned by `eig_sym3` (not necessarily sorted).
"""
function eigen_calc(eigen_mat::VectorFunc)
    A11 = eigen_mat["A11"].space_func
    A12 = eigen_mat["A12"].space_func
    A13 = eigen_mat["A13"].space_func
    A22 = eigen_mat["A22"].space_func
    A23 = eigen_mat["A23"].space_func
    A33 = eigen_mat["A33"].space_func

    N0, N1 = size(A11)
    eig_val_mat = Array{dtype}(undef, N0, N1, 3)

    @threads for j in 1:N1
        @inbounds for i in 1:N0
            λ1, λ2, λ3 = eig_sym3(A11[i,j], A12[i,j], A13[i,j],
                                  A22[i,j], A23[i,j], A33[i,j])
            eig_val_mat[i, j, 1] = λ1
            eig_val_mat[i, j, 2] = λ2
            eig_val_mat[i, j, 3] = λ3
        end
    end

    return eig_val_mat
end