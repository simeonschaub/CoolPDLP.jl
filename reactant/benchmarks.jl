# ╔═╡ f30b45e9-4f21-43ba-866a-5acead85ccad
begin
    using Revise
    #eval(:(import Pkg; Pkg.develop("CoolPDLP")))
    using CoolPDLP, MinimumCostFlows
end

# ╔═╡ 03db4d24-f759-4d65-a0b3-b9058ea4091d
using LinearAlgebra, SparseArrays

# ╔═╡ ca7de5ac-4e1b-438f-a41e-91382f244921
using CUDA, cuSPARSE, Adapt

# ╔═╡ 56556686-1ae2-4721-ae85-119b4163eb8c
using KernelAbstractions

# ╔═╡ 32f56903-e7fb-4372-99fe-9baf9c3372e7
using Graphs

# ╔═╡ 38ccebad-dcdb-470a-9acf-abbd31b2ba77
using SIMD

# ╔═╡ cd8f43a4-09d4-4c60-ab60-423249f7c2a0
using Preferences

prob = read_dimacs_mcf("/home/simeon.schaub/min-cost-flow/road/road_flow_07_TX_" .* ('a':'e') .* ".min.gz", NativeMCFProblem{Int32, Int32})

(; g, supply, cap_lo, cap_hi, cost) = prob

using MinimumCostFlows: JDSMatrixPM, JDSMatrix, Matrix2PerRowPM, Matrix2PerRow, construct_constraint_matrix

A_lp, l = construct_constraint_matrix(prob, Float32);

b_test = rand(size(A_lp, 2))
b_test_left = rand(size(A_lp, 1))

A_lpt = Matrix2PerRowPM(A_lp');

using GPUToolbox: i32
using MinimumCostFlows: One, Zero

function two_per_col_spmv!(c::AbstractVector{T}, (; colidx)::Matrix2PerRowPM{I}, b::AbstractVector{T}, α::Number, β::Number) where {T <: Number, I <: Integer}
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    @inbounds @fastmath if i <= length(c) ÷ 2
        ptr = reinterpret(Core.LLVMPtr{NTuple{4, Base.VecElement{I}}, AS.Global}, pointer(colidx))
        j = SIMD.Vec(CUDA.unsafe_cached_load(ptr, i))
        bⱼ = vgathera((pointer(b, 0i32)) + j * Int32(sizeof(T)))

        l = shufflevector(bⱼ, Val((0, 1)))
        sl = sum(SIMD.Vec{2, T}((1, -1)) * l)
        c[i] = α * sl + β * c[i]
        u = shufflevector(bⱼ, Val((2, 3)))
        su = sum(SIMD.Vec{2, T}((1, -1)) * u)
        c[i + 1i32] = α * su + β * c[i + 1i32]
    end
    return nothing
end

function two_per_col_spmv!(c::AbstractVector{T}, (; colidx, vals)::Matrix2PerRow{T, I}, b::AbstractVector{T}, α::Number, β::Number) where {T <: Number, I <: Integer}
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    @inbounds @fastmath if i <= length(c) ÷ 2
        ptr = reinterpret(Core.LLVMPtr{NTuple{4, Base.VecElement{I}}, AS.Global}, pointer(colidx))
        j = SIMD.Vec(CUDA.unsafe_cached_load(ptr, i))
        bⱼ = vgathera((pointer(b, 0i32)) + j * Int32(sizeof(T)))
        val_ptr = reinterpret(Core.LLVMPtr{NTuple{4, Base.VecElement{T}}, AS.Global}, pointer(vals))
        val = SIMD.Vec(CUDA.unsafe_cached_load(val_ptr, i))
        prod = val * bⱼ

        sl = sum(shufflevector(prod, Val((0, 1))))
        c[i] = α * sl + β * c[i]
        su = sum(shufflevector(prod, Val((2, 3))))
        c[i + 1i32] = α * su + β * c[i + 1i32]
    end
    return nothing
end

function LinearAlgebra.mul!(c::CuVector{T}, A::Union{Matrix2PerRow{T, I}, Matrix2PerRowPM{I}}, b::CuVector{T}, α::Number, β::Number) where {T <: Number, I <: Integer}
    α_is_one = isone(α)
    β_is_zero = iszero(β)
    threads = 768
    blocks = cld(length(c) ÷ 2, threads)
    if α_is_one && β_is_zero
        @cuda threads=threads blocks=blocks two_per_col_spmv!(c, A, b, One(), Zero())
    elseif α_is_one
        @cuda threads=threads blocks=blocks two_per_col_spmv!(c, A, b, One(), β)
    elseif β_is_zero
        @cuda threads=threads blocks=blocks two_per_col_spmv!(c, A, b, α, Zero())
    else
        @cuda threads=threads blocks=blocks two_per_col_spmv!(c, A, b, α, β)
    end
    return c
end

using BenchmarkTools

let b = CuArray{Float32}(b_test_left)
    A = adapt(CUDABackend(), A_lpt)
    c = similar(b, length(b_test))
    @benchmark CUDA.@sync mul!($c, $A, $b, 1f0, 0f0)
end

let b = CuArray{Float32}(b_test_left)
    A = adapt(CUDABackend(), Matrix2PerRow{Float32}(A_lpt))
    c = similar(b, length(b_test))
    @benchmark CUDA.@sync mul!($c, $A, $b, 1f0, 0f0)
end

let b = CuArray{Float32}(b_test_left)
    A = CuSparseMatrixCSR(SparseMatrixCSC{Float32}(A_lpt))
    c = similar(b, length(b_test))
    @benchmark CUDA.@sync mul!($c, $A, $b, 1f0, 0f0)
end

let b = CuArray{Float32}(b_test_left[:, 1:1])
    A = CuSparseMatrixCSR(SparseMatrixCSC{Float32}(A_lpt))
    c = similar(b, length(b_test), 1)
    @benchmark CUDA.@sync mul!($c, $A, $b, 1f0, 0f0)
end

let b = CuArray{Float32}(b_test)
    A = adapt(CUDABackend(), A_lp)
    c = similar(b, length(b_test_left))
    @benchmark CUDA.@sync mul!($c, $A, $b, 1f0, 0f0)
end

let b = CuArray{Float32}(b_test)
    b2 = similar(b)
    A = adapt(CUDABackend(), A_lp)
    c = similar(b, length(b_test_left))
    s1, s2 = CUDA.ones.(size(A))
    @benchmark CUDA.@sync begin
        $b2 .= $s2 .* $b
        mul!($c, $A, $b2, 1f0, 0f0)
        $c .*= $s1
    end
end

let b = CuArray{Float32}(b_test)
    A = adapt(CUDABackend(), JDSMatrix{Float32}(A_lp))
    c = similar(b, length(b_test_left))
    @benchmark CUDA.@sync mul!($c, $A, $b, 1f0, 0f0)
end

let b = CuArray{Float32}(b_test)
    A = CuSparseMatrixCSR(SparseMatrixCSC{Float32}(A_lp))
    c = similar(b, length(b_test_left))
    @benchmark CUDA.@sync mul!($c, $A, $b, 1f0, 0f0)
end

let b = CuArray{Float32}(b_test[:, 1:1])
    A = CuSparseMatrixCSR(SparseMatrixCSC{Float32}(A_lp))
    c = similar(b, length(b_test_left), 1)
    @benchmark CUDA.@sync mul!($c, $A, $b, 1f0, 0f0)
end
