# ╔═╡ f30b45e9-4f21-43ba-866a-5acead85ccad
begin
    using Revise
    #eval(:(import Pkg; Pkg.develop("CoolPDLP")))
    using CoolPDLP, MinimumCostFlows
end

# ╔═╡ 03db4d24-f759-4d65-a0b3-b9058ea4091d
using LinearAlgebra, SparseArrays

# ╔═╡ ca7de5ac-4e1b-438f-a41e-91382f244921
using CUDA, cuSPARSE

# ╔═╡ 32f56903-e7fb-4372-99fe-9baf9c3372e7
using Graphs

prob = read_dimacs_mcf("/home/simeon.schaub/min-cost-flow/road/road_flow_07_TX_" .* ('a':'e') .* ".min.gz", NativeMCFProblem{Int32, Int32})

(; g, supply, cap_lo, cap_hi, cost) = prob

using MinimumCostFlows: JDSMatrixPM, JDSMatrix, Matrix2PerRowPM, Matrix2PerRow, construct_constraint_matrix

A_lp, l = construct_constraint_matrix(prob, Float32);

b_test = rand(size(A_lp, 2))
b_test_left = rand(size(A_lp, 1))

A_lpt = Matrix2PerRowPM(A_lp');

Base.get_extension(MinimumCostFlows, :CUDAExt)
which(LinearAlgebra.mul!, (CuVector{Float32}, Matrix2PerRowPM{Int32}, CuVector{Float32}, Float32, Float32))

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
