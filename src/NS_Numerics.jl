module NS_Numerics

using LinearAlgebra
using SpecialFunctions
using IntervalArithmetic
using Combinatorics
using SparseArrays
using PyPlot
using MAT
using Base:tail, front
using Base.Threads
using Arblib
import LinearAlgebra: norm # need to reload norm
import Base: +, -, *, /, copy, getindex, setindex! # reload arithmetic operators
import ArbExtras: besselj
import FunctionZeros: besselj_zero
import HypergeometricFunctions:pFq

# Change to Interval{Float64} when doing rigorous verification
const dtype = Interval{Float64}
# const dtype = Float64

println("Using dtype = ", dtype)
println("Number of threads = ", nthreads())

include("Matrix_Func.jl")
include("My_Fourier.jl")
include("My_Scalar.jl")
include("My_Vector.jl")
include("My_Operator.jl")
include("My_Bessel.jl")
include("Trig_Poly.jl")
include("Comb_Bound.jl")
include("My_Assembly.jl")
include("NS_Terms.jl")

export
    dtype, norm,
    
    # My_Fourier
    dt, is_zero, Pi, mysin, mycos, double_fact, gamma_ratio, che2len, len2che, 
    conversion_mat, Fourier_mat, div_mat, calc_Utheta_divfree, trig_func,

    # Matrix_Func
    newton_cotes_weights, newton_cotes_with_bound, finite_difference, Linf_to_paras, 
    get_Wk∞_norm_lst, estimate_Wk∞_norm, plot_matrix, add_error, save_matrix, 
    interval_matrix, read_matrix, verify_spd_by_ldlt, verify_general_eig, eig_sym3, 
    vec_to_mat,

    # My_Scalar
    BDRY_LIST, check_bdry, ScalarFunc, set_func!, set_bdry!, set_conv_mat!,
    Fourier_transform, interpolate, transform!, inner_product, get_Wk∞_norm_lst, 
    trim_border!, my_plot, add_equal!, multiply_equal!, minus_equal!, divide_equal!,

    # My_Vector
    BDRY_LST_SET, VectorFunc, from_mat_file, load_grad_U, transform!, interpolate, 
    set_div_free!, inner_product, my_plot, norm_lst, trim_border!, save, add_equal!,
    multiply_equal!, minus_equal!, divide_equal!,

    # My_Operator
    Operator, set_T!, set_L1!, apply, change_variable!, Operator_matrix, 
    add_operator!, apply_convmat_right!, eigen_calc,

    # My_Bessel
    bessel_W2_bound_interval, my_besselj, func_Z, func_Y, sign_change, 
    rigorous_root_interval, verify_root, mat_to_lst, preprocess_para!, 
    load_besselj_zeros, Ur_func, eval_Ur, eval_Uθ, eval_Q_eig_function,

    # Trig_Poly
    TrigPoly, trig_mul, trig_int_0_pi2, trig_basis_lists, sph_basis_lists, 
    tensor_int_basis, contract_QT_blocks, split_tensor, integrate_θ, int_θ_bound,

    # Comb_Bound
    multinomial_sum, poly_deriv, poly_add, poly_mul_x, poly_mul_1_plus_x2,
    poly_abs_sum_bound, beta_all_deriv_Linf_bounds, tan_over_cos_deriv_Linf_bounds,
    r_to_beta, eig_fun_bound,

    # My_Assembly
    Ur_bound, Uθ_bound, integrate_r, eval_l2_inpd, eval_h1_norm, 
    Ur_sq_Wk∞_bound_lists, get_lap_matrix, Ur_bound_lst, lst_to_mat,

    # NS_Terms
    build_operator, get_linear, get_identity, get_gradient_sym, get_gradient, 
    get_operator_matrix, mul_A_gradB_local, mul_A_gradB
end