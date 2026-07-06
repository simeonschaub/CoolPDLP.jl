### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ f30b45e9-4f21-43ba-866a-5acead85ccad
begin
	using Revise
	#eval(:(import Pkg; Pkg.develop("CoolPDLP")))
	using CoolPDLP, MinimumCostFlows
end

# ╔═╡ 4b866d1e-6bd7-11f1-b595-5d614c27232b
using Reactant

# ╔═╡ 03db4d24-f759-4d65-a0b3-b9058ea4091d
using LinearAlgebra, SparseArrays

# ╔═╡ ca7de5ac-4e1b-438f-a41e-91382f244921
using CUDA, cuSPARSE, Adapt

# ╔═╡ eb7c1bea-04cf-45b9-a5ec-fe6a53291372
using PythonCall

# ╔═╡ 56556686-1ae2-4721-ae85-119b4163eb8c
using KernelAbstractions

# ╔═╡ 32f56903-e7fb-4372-99fe-9baf9c3372e7
using Graphs

# ╔═╡ 38ccebad-dcdb-470a-9acf-abbd31b2ba77
using SIMD

# ╔═╡ cd8f43a4-09d4-4c60-ab60-423249f7c2a0
using Preferences

# ╔═╡ 177db2b3-ed9e-4e70-92ed-09c24954f32a
Reactant.set_default_backend("gpu")

# ╔═╡ b6eb660e-873f-46fa-97e9-a50e8ad76e70
b = Reactant.to_rarray(CUDA.rand(10000))

# ╔═╡ a8eccc79-51d1-4034-86cf-a593389007c6
x = Reactant.to_rarray(CuArray{Float32}(undef, 10000))

# ╔═╡ 5440bbb5-d146-4b19-8bbd-976ddbf94499
@pyexec """
global jax, _spmv
import jax
import jax.experimental.sparse

@jax.jit(static_argnames=['m'])
def _spmv(_A, b, m):
	(I, J, V) = _A
	A = jax.experimental.sparse.CSR((V, I, J), shape = (m, len(b)))
	return jax.experimental.sparse.csr_matvec(A, b)

def spmv(_A, _b, m):
	b = jax.numpy.array(_b)
	(_I, _J, _V) = _A
	I = jax.numpy.array(_I) - 1
	J = jax.numpy.array(_J) - 1
	V = jax.numpy.array(_V)
	return _spmv((I, J, V), b, m)
""" => spmv

# ╔═╡ 3d8a5324-3193-4f86-ac4e-f57e189f2b5b
begin
	struct JAXSparseMatrixCSR{T, I, VI <: AbstractVector{I}, VT <: AbstractVector{T}} <: AbstractSparseMatrix{T, I}
		colval::VI
		rowptr::VI
		nzval::VT
		m::Int
		n::Int
	end
	function JAXSparseMatrixCSR(A::GPUSparseMatrixCSR)
	    A′ = Reactant.to_rarray(A)
	    return JAXSparseMatrixCSR(A′.colval, A′.rowptr, A′.nzval, A′.m, A′.n)
	end
	JAXSparseMatrixCSR(A::AbstractSparseMatrix) = JAXSparseMatrixCSR(GPUSparseMatrixCSR(A))
end

# ╔═╡ e098b073-7213-4e34-aa71-17a5ff396ced
Base.size((; m, n)::JAXSparseMatrixCSR) = (m, n)

# ╔═╡ 0bfe2d70-4fd5-43ac-ae01-abbd44f49296
KernelAbstractions.get_backend((; nzval)::JAXSparseMatrixCSR) = get_backend(nzval)

# ╔═╡ ed3f004f-afa4-4fbf-bebd-4c35e50b3fc1
Adapt.adapt_storage(b::CUDABackend, (; colval, rowptr, nzval, m, n)::JAXSparseMatrixCSR) = CuSparseMatrixCSR(adapt(b, rowptr), adapt(b, colval), adapt(b, nzval), (m, n))
Adapt.adapt_storage(b::CPU, (; colval, rowptr, nzval, m, n)::JAXSparseMatrixCSR) = permutedims(SparseMatrixCSC(n, m, adapt(b, rowptr), adapt(b, colval), adapt(b, nzval)))

# ╔═╡ b2e7ada9-a0bc-4fa4-b9cf-98a0a4d6eddf
Reactant.use_overlayed_version(::JAXSparseMatrixCSR) = false

# ╔═╡ f8bf5c2c-7428-4e60-92c3-c242b224e690
function LinearAlgebra.mul!(
        c::AbstractVector,
        A::JAXSparseMatrixCSR,
        b::AbstractVector,
        α::Number,
        β::Number
    )
    return c .= @jit Reactant.TracedLinearAlgebra.overloaded_mul!(c, A, b, α, β)
end

# ╔═╡ 9b9c9631-fe6c-47b5-9106-226ecc94861a
function Reactant.TracedLinearAlgebra.overloaded_mul!(
        c::AbstractVector,
        A::JAXSparseMatrixCSR,
        b::AbstractVector,
        α::Number,
        β::Number
    )
    Reactant.call_with_reactant(c, A, b, α, β) do c, A, b, α, β
        tmp = spmv((A.colval, A.rowptr, A.nzval), b, A.m)

        β_is_zero = !(β isa Reactant.TracedRNumber) && iszero(β)
        α_is_one = !(α isa Reactant.TracedRNumber) && isone(α)

        if α_is_one && β_is_zero
            c .= tmp
        else
            α_res = if α_is_one
                tmp
            else
                α .* tmp
            end
            if β_is_zero
                c .= α_res
            else
                c .= α_res .+ β .* c
            end
        end
    end
    return c
end

# ╔═╡ 33cb78ab-82eb-4b46-ac0e-934f7cfb3430
prob = read_dimacs_mcf("/home/simeon.schaub/test5.dimacs", NativeMCFProblem{Int32, Int32})

(; g, supply, cap_lo, cap_hi, cost) = prob

using MinimumCostFlows: JDSMatrixPM, Matrix2PerRowPM, Matrix2PerRow, construct_constraint_matrix

# ╔═╡ c0b1c4e0-4d2b-47ca-b5a8-e8dc16e0ef22
A = Reactant.to_rarray(CoolPDLP.GPUSparseMatrixCSR(SparseMatrixCSC{Float32, Int32}(sprand(Float32, 10000, 10000, .001))));

# ╔═╡ 714c10e9-dd20-4884-b3c9-983452c0476e
typeof(A)

# ╔═╡ 756ccafd-8485-470c-83a3-991eaf2c79bb
spmv((A.colval, A.rowptr, A.nzval), b, A.m)

# ╔═╡ a1f59ebd-11b7-46ef-8589-ba5ac960821c
f = @compile sync=true spmv((Int32.(A.colval), Int32.(A.rowptr), A.nzval), b, A.m)

# ╔═╡ 55f78666-ca94-4b2c-a452-c3d47b05ffe1
let (A, b, m) = ((Int32.(A.colval), Int32.(A.rowptr), A.nzval), b, A.m)
	@time for _ in 1:100
		f(A, b, m)
	end
end

# ╔═╡ 12733c98-8d7d-44cd-9478-f8ca05cd0cb9
let (A, b, m) = ((Int32.(A.colval), Int32.(A.rowptr), A.nzval), b, A.m)
	f(A, b, m)
end

# ╔═╡ 3d7d97e3-16f8-4e01-8a8f-d9cb4a355a18
A_jax = JAXSparseMatrixCSR(Int32.(A.colval), Int32.(A.rowptr), A.nzval, A.m, A.n);

# ╔═╡ 2b4a7ab7-2fef-4da6-87e5-d4a4a2e28287
function Reactant.TracedLinearAlgebra.overloaded_mul!(
        c::AbstractVector,
        A::GPUSparseMatrixCSR,
        b::AbstractVector,
        α::Number,
        β::Number
    )
    backend = CoolPDLP.common_backend(c, A, b)
    kernel! = CoolPDLP.spmv_csr!(backend)
    kernel!(c, A.rowptr, A.colval, A.nzval, b, α, β; ndrange = size(A, 1))
    return c
end

# ╔═╡ bab6dd21-9fe0-463d-a6c0-9bc35b3f13ce
b_test = rand(500000)

# ╔═╡ b79fd3f9-fdb2-4311-90e5-28950187592c
SparseMatrixCSC{Int8}(JDSMatrixPM([2, -1, 3, -4, 3], Base.OneTo(4), [1, 4, 6], 3))

# ╔═╡ 84d1b460-330d-44fa-8c62-de1df60dc6ee
A_lp, l = construct_constraint_matrix(prob, Float32);

u = copy(l)

# ╔═╡ dc79b95b-9b4e-4d1c-a46b-b4b87bd29273
A_lpt = Matrix2PerRowPM(A_lp');

# ╔═╡ a1930dc4-b211-4827-82c0-8e7f6cf2146e
SparseMatrixCSC{Int8}(A_lp)

# ╔═╡ 8133b28c-50ae-4e44-88a8-fb8a3785e84a


# ╔═╡ b3d3ea67-2206-437a-bedc-a8690b322a73
# ╠═╡ disabled = true
#=╠═╡
milp = MILP(;
	c = Reactant.to_rarray(Float32.(nonzeros(cost))),
	lv = Reactant.to_rarray(Float32.(nonzeros(cap_lo))),
	uv = Reactant.to_rarray(Float32.(nonzeros(cap_hi))),
	A = JAXSparseMatrixCSR(A_lp′.colval, A_lp′.rowptr, A_lp′.nzval, A_lp′.m, A_lp′.n),
	At = JAXSparseMatrixCSR(A_lpt.colval, A_lpt.rowptr, A_lpt.nzval, A_lpt.m, A_lpt.n),
	lc = Reactant.to_rarray(Float32.(l)),
	uc = Reactant.to_rarray(Float32.(u)),
);
  ╠═╡ =#

# ╔═╡ 8b469c37-29b6-4fa2-84e8-4c30292360b4
milp = MILP(;
	c = Float32.(nonzeros(cost)),
	lv = Float32.(nonzeros(cap_lo)),
	uv = Float32.(nonzeros(cap_hi)),
	#A = SparseMatrixCSC{Float32}(A_lp),
	A = A_lp,
	At = A_lpt,
	lc = Float32.(l),
	uc = Float32.(u),
);

# ╔═╡ a8cdac0d-b39a-47cc-a0f7-2a5110f34631
typeof(milp)

# ╔═╡ f22d89ed-9adf-4906-b509-815e460e791c
begin
	struct MyReactantBackend <: KernelAbstractions.GPU end
	Adapt.adapt_storage(::MyReactantBackend, a) = Reactant.to_rarray(a)
	KernelAbstractions.get_backend(::JAXSparseMatrixCSR) = MyReactantBackend()
	KernelAbstractions.allocate(::MyReactantBackend, T::Type, size::Tuple) = ConcreteRArray{T}(undef, size...)
	Reactant.@reactant_overlay KernelAbstractions.allocate(::MyReactantBackend, ::Type{Reactant.TracedRNumber{T}}, size::NTuple{N}) where {T, N} = Reactant.TracedRArray{T, N}((), nothing, size)
end

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

# ╔═╡ f2833b6b-9d13-4068-ae52-beea8e6ffec0
let (A, b) = (adapt(CUDABackend(), A), CuArray(b))
	c = similar(b)
	CUDA.@time for _ in 1:100
		mul!(c, A, b)
	end
	c
end

# ╔═╡ 801dc5e8-515e-4278-8c59-7da6f6cf2dba
let (A, b) = (cuSPARSE.CuSparseMatrixCSR(CuArray(A.rowptr), CuArray(A.colval), CuArray(A.nzval), (Int32(A.m), Int32(A.n))), CuArray(b))
	c = similar(b)
	CUDA.@time for _ in 1:100
		mul!(c, A, b)
	end
	c
end

# ╔═╡ a656443d-e4e8-4f49-84c5-93bcc7fc0836
@jit mul!(similar(b), A_jax, b)

# ╔═╡ 0b94583a-0667-4a91-babe-009036f6e297
let b = b_test[1:100000]
	mul!(similar(b, 500000), A_lpt, b, 1.0, 0.0), mul!(similar(b, 500000), SparseMatrixCSC{Float64}(A_lpt), b, 1.0, 0.0)
end

# ╔═╡ 56069ebf-0372-42b4-a0bf-e3e1a5ed2007
let b = CuArray{Float32}(b_test[1:100000])
	A = adapt(CUDABackend(), A_lpt)
	c1 = similar(b, 500000)
	@device_code_llvm mul!(c1, A, b, 1f0, 0f0)
end

# ╔═╡ 8b1dc72b-2356-423d-9300-ac1491b95703
let b = CuArray{Float32}(b_test[1:100000])
	A = adapt(CUDABackend(), A_lpt)
	c1 = similar(b, 500000)
	CUDA.@time for _ in 1:1000
		mul!(c1, A, b, 1f0, 0f0)
	end
	A2 = CuSparseMatrixCSR(SparseMatrixCSC{Float32}(A_lpt))
	c2 = similar(c1)
	CUDA.@time for _ in 1:1000
		mul!(c2, A2, b, 1f0, 0f0)
	end
	c1, c2
end

let b = CuArray{Float32}(b_test[1:100000])
	A = adapt(CUDABackend(), Matrix2PerRow(A_lpt))
	c1 = similar(b, 500000)
	CUDA.@time for _ in 1:1000
		mul!(c1, A, b, 1f0, 0f0)
	end
	A2 = CuSparseMatrixCSR(SparseMatrixCSC{Float32}(A_lpt))
	c2 = similar(c1)
	CUDA.@time for _ in 1:1000
		mul!(c2, A2, b, 1f0, 0f0)
	end
	c1, c2
end

# ╔═╡ 902133e5-a422-4577-8c09-ecd4550ecc0c
let b = CuArray{Float32}(b_test)
	A = adapt(CUDABackend(), A_lp)
	c1 = similar(b, 100000)
	@device_code_llvm mul!(c1, A, b, 1f0, 0f0)
end

# ╔═╡ bb7aff80-9624-4349-8519-380ddcf31330
let b = CuArray{Float32}(b_test)
	A = adapt(CUDABackend(), A_lp)
	c1 = similar(b, 100000)
	CUDA.@time for _ in 1:1000
		mul!(c1, A, b, 1f0, 0f0)
	end
	A2 = CuSparseMatrixCSR(SparseMatrixCSC{Float32}(A_lp))
	c2 = similar(c1)
	CUDA.@time for _ in 1:1000
		mul!(c2, A2, b, 1f0, 0f0)
	end
	c1, c2
end

# ╔═╡ 2c7288f1-d269-4e4b-996b-1a70c89ce5f4
let b = Reactant.to_rarray(Float32.(b_test))
	A = Reactant.to_rarray(A_lp)
	c1 = similar(b, 100000)
	f = @compile sync = true mul!(c1, A, b, 1f0, 0f0)
	@time f(c1, A, b, 1f0, 0f0)
end

# ╔═╡ 70655d62-3dc6-47f1-885f-79e01ca6d6c5
let b = Reactant.to_rarray(Float32.(b_test))
	A = Reactant.to_rarray(A_lp)
	c1 = similar(b, 100000)
	f = @compile raise = true sync = true mul!(c1, A, b, 1f0, 0f0)
	@time f(c1, A, b, 1f0, 0f0)
end

# ╔═╡ a877a5c1-f238-4df3-bc2c-879ffc641497
let b = b_test
	mul!(similar(b, 100000), A_lp, b, 1.0, 0.0), mul!(similar(b, 100000), SparseMatrixCSC{Float64}(A_lp), b, 1.0, 0.0)
end

# ╔═╡ 7650a412-5113-445e-82ec-84e15b41f3aa
sol, stats = solve(milp, PDLP(
    Float32,  # desired float type
    Int32,  # desired int type
    JAXSparseMatrixCSR,  # GPU sparse matrix type
    backend = MyReactantBackend(),
    time_limit = 100.0,#00.0,
    #max_kkt_passes = 10^6,
    termination_reltol = 1e-6,
))

# ╔═╡ a34ee9db-8ea8-4cea-9d6d-bd9fa1f18acb
begin
	struct Foo{B} <: AbstractMatrix{Float32} end
	function CoolPDLP.set_matrix_type(::Type{Foo{B}}, milp::MILP) where {B}
	    (;
            c, lv, uv, A, At, lc, uc, D1, D2,
            int_var, var_names, dataset, name, path,
        ) = milp
        A_M = adapt(B(), A_lp)
        At_M = adapt(B(), A_lpt)
        #backend = MyReactantBackend()
        backend = B()

        return MILP(;
            c = adapt(backend, c),
            lv = adapt(backend, lv),
            uv = adapt(backend, uv),
            A = A_M,
            At = At_M,
            lc = adapt(backend, Float32.(l)), #lc),
            uc = adapt(backend, Float32.(u)), #uc),
            D1 = adapt(backend, D1),
            D2 = adapt(backend, D2),
            int_var = adapt(backend, int_var),
            var_names,
            dataset,
            name,
            path
        )
    end
end

# ╔═╡ 8cda099c-9629-4fcd-9e15-ea93b8a92571
solve(milp, PDLP(
    Float32,  # desired float type
    Int32,  # desired int type
    Foo{CUDABackend},  # GPU sparse matrix type
    backend = CUDABackend(),
    time_limit = 100.0,#00.0,
    max_kkt_passes = 10^6,
    termination_reltol = 1e-4,
))
solve(milp, PDLP(
    Float32,  # desired float type
    Int32,  # desired int type
    CuSparseMatrixCSR,  # GPU sparse matrix type
    backend = CUDABackend(),
    time_limit = 100.0,#00.0,
    max_kkt_passes = 10^6,
    termination_reltol = 1e-4,
))

begin
	struct Bar{B} <: AbstractMatrix{Float32} end
	function CoolPDLP.set_matrix_type(::Type{Bar{B}}, milp::MILP{T, <:AbstractVector{T}, <:MinimumCostFlows.JDSMatrix, <:Matrix2PerRow}) where {B, T<:Number}
	    (;
            c, lv, uv, A, At, lc, uc, D1, D2,
            int_var, var_names, dataset, name, path,
        ) = milp
        A_M = adapt(B(), A)
        At_M = adapt(B(), At)
        #backend = MyReactantBackend()
        backend = B()

        return MILP(;
            c = adapt(backend, c),
            lv = adapt(backend, lv),
            uv = adapt(backend, uv),
            A = A_M,
            At = At_M,
            lc = adapt(backend, lc),
            uc = adapt(backend, uc),
            D1 = adapt(backend, D1),
            D2 = adapt(backend, D2),
            int_var = adapt(backend, int_var),
            var_names,
            dataset,
            name,
            path
        )
    end
end

solve(milp, PDLP(
    Float32,  # desired float type
    Int32,  # desired int type
    Bar{CUDABackend},  # GPU sparse matrix type
    backend = CUDABackend(),
    time_limit = 100.0,#00.0,
    max_kkt_passes = 10^6,
    termination_reltol = 1e-4,
))

solve(milp, PDLP(
    Float32,  # desired float type
    Int32,  # desired int type
    Bar{MyReactantBackend},  # GPU sparse matrix type
    backend = MyReactantBackend(),
    time_limit = 100.0,#00.0,
    max_kkt_passes = 10^6,
    termination_reltol = 1e-4,
))

# ╔═╡ f0178e50-93c2-487a-a317-89d9f63d8f85
Preferences.set_preferences!(CoolPDLP, "dispatch_doctor_mode" => "disable"; export_prefs = true)

# ╔═╡ 557c5345-197c-441a-9bd4-dd34edc2321f
Preferences.set_preferences!(Revise, "revise_structs" => true; export_prefs = true)

# ╔═╡ 7e56368c-77b1-4f06-be57-ad59efdeb142
# ╠═╡ disabled = true
#=╠═╡
@kernel function two_per_col_spmv!(c::AbstractVector{T}, (; colidx)::Matrix2PerRowPM{I}, b::AbstractVector{T}, α::Number, β::Number) where {T <: Number, I <: Integer}
	i = @index(Global)
	@inbounds #=@fastmath=# begin
		j = vload(SIMD.Vec{2, I}, pointer(colidx, 2i - 1))
		vals = vgather((pointer(b, 0)) + j * sizeof(T))
		s = sum(SIMD.Vec{2, T}((1, -1)) * vals)
		c[i] = α * s + β * c[i]
	end
end
  ╠═╡ =#

# ╔═╡ 9f5cbe3c-6882-43d6-a69e-b8595ff13a3a
@kernel function two_per_col_spmv!(c::AbstractVector{T}, (; colidx)::Matrix2PerRowPM{I}, b::AbstractVector{T}, α::Number, β::Number) where {T <: Number, I <: Integer}
	i = @index(Global)
	@inbounds #=@fastmath=# begin
		s = zero(T)
		j = colidx[1, i]
		if j > 0
			s += b[j]
		end
		j = colidx[2, i]
		if j > 0
			s -= b[j]
		end
		c[i] = α * s + β * c[i]
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
"""

# ╔═╡ Cell order:
# ╠═f30b45e9-4f21-43ba-866a-5acead85ccad
# ╠═4b866d1e-6bd7-11f1-b595-5d614c27232b
# ╠═177db2b3-ed9e-4e70-92ed-09c24954f32a
# ╠═03db4d24-f759-4d65-a0b3-b9058ea4091d
# ╠═ca7de5ac-4e1b-438f-a41e-91382f244921
# ╠═c0b1c4e0-4d2b-47ca-b5a8-e8dc16e0ef22
# ╠═714c10e9-dd20-4884-b3c9-983452c0476e
# ╠═b6eb660e-873f-46fa-97e9-a50e8ad76e70
# ╠═a8eccc79-51d1-4034-86cf-a593389007c6
# ╠═3ad1cb2b-1c8d-4500-882b-852972427926
# ╠═725500c4-ae2d-47fc-a44d-a14e9417d322
# ╠═687ee5af-479a-4976-9ba8-fafd77b33707
# ╠═1f26e113-a8dd-411d-9a79-9ece6e542c16
# ╠═2b4a7ab7-2fef-4da6-87e5-d4a4a2e28287
# ╠═eb7c1bea-04cf-45b9-a5ec-fe6a53291372
# ╠═5440bbb5-d146-4b19-8bbd-976ddbf94499
# ╠═756ccafd-8485-470c-83a3-991eaf2c79bb
# ╠═a1f59ebd-11b7-46ef-8589-ba5ac960821c
# ╠═23247f32-ed39-4c72-b90a-944c30c8e847
# ╠═c3e3ee37-18c4-4065-9408-7a3860af2919
# ╠═f2833b6b-9d13-4068-ae52-beea8e6ffec0
# ╠═801dc5e8-515e-4278-8c59-7da6f6cf2dba
# ╠═55f78666-ca94-4b2c-a452-c3d47b05ffe1
# ╠═01f62c77-5d7e-4d70-8784-57495f0ddf3f
# ╠═12733c98-8d7d-44cd-9478-f8ca05cd0cb9
# ╠═56556686-1ae2-4721-ae85-119b4163eb8c
# ╠═3d8a5324-3193-4f86-ac4e-f57e189f2b5b
# ╠═e098b073-7213-4e34-aa71-17a5ff396ced
# ╠═0bfe2d70-4fd5-43ac-ae01-abbd44f49296
# ╠═ed3f004f-afa4-4fbf-bebd-4c35e50b3fc1
# ╠═b2e7ada9-a0bc-4fa4-b9cf-98a0a4d6eddf
# ╠═9b9c9631-fe6c-47b5-9106-226ecc94861a
# ╠═f8bf5c2c-7428-4e60-92c3-c242b224e690
# ╠═3d7d97e3-16f8-4e01-8a8f-d9cb4a355a18
# ╠═a656443d-e4e8-4f49-84c5-93bcc7fc0836
# ╠═32f56903-e7fb-4372-99fe-9baf9c3372e7
# ╠═3d0c7357-0557-4ee6-82a0-92ce02b3b5e0
# ╠═33cb78ab-82eb-4b46-ac0e-934f7cfb3430
# ╠═186d9701-ee32-427f-b358-f946c5116ab4
# ╠═d4113e89-2dce-458e-b67d-6dd7ae7bb81b
# ╠═cb3f97c7-3ecc-44ca-9efd-a71c951bf891
# ╠═daa38713-fc22-4d8c-b831-4a224c142585
# ╠═a728eee3-1c6e-4763-b0dd-9ee639f3a031
# ╠═8e60db21-c1a5-4ff9-8cc5-8bd769f7ad75
# ╠═38ccebad-dcdb-470a-9acf-abbd31b2ba77
# ╠═7e56368c-77b1-4f06-be57-ad59efdeb142
# ╠═9f5cbe3c-6882-43d6-a69e-b8595ff13a3a
# ╠═9f9c6e6e-ec58-4079-b783-5e05ac12ecbe
# ╠═ebcb36b8-e124-4482-93c1-1584130575dd
# ╠═dc79b95b-9b4e-4d1c-a46b-b4b87bd29273
# ╠═0b94583a-0667-4a91-babe-009036f6e297
# ╠═56069ebf-0372-42b4-a0bf-e3e1a5ed2007
# ╠═8b1dc72b-2356-423d-9300-ac1491b95703
# ╠═bab6dd21-9fe0-463d-a6c0-9bc35b3f13ce
# ╠═902133e5-a422-4577-8c09-ecd4550ecc0c
# ╠═bb7aff80-9624-4349-8519-380ddcf31330
# ╠═2c7288f1-d269-4e4b-996b-1a70c89ce5f4
# ╠═70655d62-3dc6-47f1-885f-79e01ca6d6c5
# ╠═a877a5c1-f238-4df3-bc2c-879ffc641497
# ╠═a1930dc4-b211-4827-82c0-8e7f6cf2146e
# ╠═b79fd3f9-fdb2-4311-90e5-28950187592c
# ╠═84d1b460-330d-44fa-8c62-de1df60dc6ee
# ╠═8133b28c-50ae-4e44-88a8-fb8a3785e84a
# ╠═b3d3ea67-2206-437a-bedc-a8690b322a73
# ╠═8b469c37-29b6-4fa2-84e8-4c30292360b4
# ╠═a8cdac0d-b39a-47cc-a0f7-2a5110f34631
# ╠═f22d89ed-9adf-4906-b509-815e460e791c
# ╠═9a3a3854-5f99-499d-8430-ba47aa9519f3
# ╠═ad350afb-b3d7-4509-a644-9600205d48d9
# ╠═80cc3978-4b8e-40a9-9def-3a1908e5413b
# ╠═396d48cd-c56e-402a-8f48-a06e8ee884c2
# ╠═77a7833e-3ffa-4fd8-ba18-9d3a7ea0ca38
# ╠═eff6c280-117d-4599-a677-d93e9baf5215
# ╠═faa8062b-fc99-4dbb-913e-393586f1629d
# ╠═db7ed6de-f65f-4c80-99ee-b0b447369705
# ╠═239bbe4c-6d32-4d0a-b0f9-628466a7c57a
# ╠═23551a28-66d4-419c-b0c9-3bf9338bec3e
# ╠═7650a412-5113-445e-82ec-84e15b41f3aa
# ╠═a34ee9db-8ea8-4cea-9d6d-bd9fa1f18acb
# ╠═74795c63-5dd5-4722-b559-16bb4da653de
# ╠═8cda099c-9629-4fcd-9e15-ea93b8a92571
# ╠═9bf727d6-5601-4668-814a-d0b7266da1d4
# ╠═cd8f43a4-09d4-4c60-ab60-423249f7c2a0
# ╠═f0178e50-93c2-487a-a317-89d9f63d8f85
# ╠═557c5345-197c-441a-9bd4-dd34edc2321f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
