using Reactant, CoolPDLP, LinearAlgebra, KernelAbstractions, Adapt

struct MyReactantBackend <: KernelAbstractions.GPU end
Adapt.adapt_storage(::MyReactantBackend, a) = Reactant.to_rarray(a)
KernelAbstractions.get_backend(::JAXSparseMatrixCSR) = MyReactantBackend()
KernelAbstractions.allocate(::MyReactantBackend, T::Type, size::Tuple) = ConcreteRArray{T}(undef, size...)
Reactant.@reactant_overlay KernelAbstractions.allocate(::MyReactantBackend, ::Type{Reactant.TracedRNumber{T}}, size::NTuple{N}) where {T, N} = Reactant.TracedRArray{T, N}((), nothing, size)

CoolPDLP.fixed_stepsize(milp::MILP{<:Number, <:ConcreteRArray}, params::CoolPDLP.StepSizeParameters) = CoolPDLP.fixed_stepsize(adapt(CUDABackend(), milp), params)
CoolPDLP.primal_weight_init(milp::MILP{<:Number, <:ConcreteRArray}, params::CoolPDLP.StepSizeParameters) = CoolPDLP.primal_weight_init(adapt(CUDABackend(), milp), params)

function CoolPDLP.termination_check!(
        state::CoolPDLP.AbstractState,
        milp::MILP,
        algo::CoolPDLP.Algorithm,
        kkt_errors!::F,
    ) where {F}
    (; sol, scratch, stats) = state
    stats.time_elapsed = time() - stats.starting_time
    stats.err = @show kkt_errors!(scratch, sol, milp)
    if algo.generic.record_error_history
        push!(stats.error_history, (stats.kkt_passes, stats.err))
    end
    stats.termination_status = CoolPDLP.termination_status(stats, algo.termination)
    return stats.termination_status !== CoolPDLP.STILL_RUNNING
end

function CoolPDLP.restart_check!(
        state::CoolPDLP.PDLPState,
        milp::MILP,
        algo::CoolPDLP.Algorithm{:PDLP},
        kkt_errors!::F,
    ) where {F}
    (;
        sol, sol_last, sol_avg, sol_avg_last, sol_restart,
        step_sizes, scratch, iteration, restart_stats,
    ) = state
    (; ω) = step_sizes

    err = kkt_errors!(scratch, sol, milp)
    err_avg = kkt_errors!(scratch, sol_avg, milp)
    if CoolPDLP.absolute(err, ω) < CoolPDLP.absolute(err_avg, ω)
        restart_stats.restart_from_avg = false
        restart_stats.err_candidate = err
    else
        restart_stats.restart_from_avg = true
        restart_stats.err_candidate = err_avg
    end

    err_last = kkt_errors!(scratch, sol_last, milp)
    err_avg_last = kkt_errors!(scratch, sol_avg_last, milp)
    if CoolPDLP.absolute(err_last, ω) < CoolPDLP.absolute(err_avg_last, ω)
        restart_stats.err_candidate_last = err_last
    else
        restart_stats.err_candidate_last = err_avg_last
    end

    restart_stats.err_restart = kkt_errors!(scratch, sol_restart, milp)

    return CoolPDLP.should_restart(restart_stats, step_sizes, iteration, algo.restart)
end

function CoolPDLP.solve!(
        state::CoolPDLP.PDLPState,
        milp::MILP{<:Number, <:ConcreteRArray},
        algo::CoolPDLP.Algorithm{:PDLP}
    )
    #prog = CoolPDLP.ProgressUnknown(desc = "PDLP iterations:", enabled = algo.generic.show_progress)
    (; η, η_sum, ω) = state.step_sizes
    step! = @compile raise = true _step!(state, milp, Reactant.ConcreteRNumber(η), Reactant.ConcreteRNumber(η_sum), Reactant.ConcreteRNumber(ω))
    kkt_errors! = @compile raise = true CoolPDLP.kkt_errors!(state.scratch, state.sol, milp)
    primal_weight_update! = @compile raise = true CoolPDLP.primal_weight_update!(
        state.scratch, state.step_sizes, state.sol, state.sol_restart, algo.step_size
    )
    while true
        for _ in 1:algo.generic.check_every
            # switch pointers
            state.sol, state.sol_last = state.sol_last, state.sol
            state.step_sizes.η_sum = step!(state, milp, Reactant.ConcreteRNumber(η), Reactant.ConcreteRNumber(η_sum), Reactant.ConcreteRNumber(ω))
            state.stats.kkt_passes += 1
            CoolPDLP.add_inner!(state.iteration)
            #CoolPDLP.next!(prog; showvalues = prog_showvalues(state))
        end
        if CoolPDLP.termination_check!(state, milp, algo, kkt_errors!)
            break
        elseif CoolPDLP.restart_check!(state, milp, algo, kkt_errors!)
            CoolPDLP.restart!(state, algo, primal_weight_update!)
        end
    end
    #CoolPDLP.finish!(prog)
    return state
end

function CoolPDLP.combine(l::Reactant.TracedRNumber, u::Reactant.TracedRNumber)
    ls = ifelse(isfinite(l), abs(l), zero(l))
    us = ifelse(isfinite(u), abs(u), zero(u))
    return max(zero(l), ls, us)
end


function CoolPDLP.restart!(state::CoolPDLP.PDLPState{T}, algo::CoolPDLP.Algorithm{:PDLP}, primal_weight_update!::F) where {T, F}
    (;
        sol, sol_avg, sol_restart,
        step_sizes, iteration, scratch, restart_stats,
    ) = state

    # identify candidate
    if restart_stats.restart_from_avg
        sol_cand = sol_avg
    else
        sol_cand = sol
    end
    # update step sizes (must be done before losing previous restart)
    step_sizes.η_sum = zero(T)
    step_sizes.ω = primal_weight_update!(
        scratch, step_sizes, sol_cand, sol_restart, algo.step_size
    )
    # update solutions
    sol !== sol_cand && copy!(sol, sol_cand)
    CoolPDLP.zero!(sol_avg)
    copy!(sol_restart, sol)
    # update counters
    CoolPDLP.add_outer!(iteration)
    return nothing
end

function CoolPDLP.primal_weight_update!(
        scratch::CoolPDLP.Scratch{<:Reactant.TracedRNumber},
        step_sizes::CoolPDLP.StepSizes,
        sol_cand::CoolPDLP.PrimalDualSolution,
        sol_restart::CoolPDLP.PrimalDualSolution,
        params::CoolPDLP.StepSizeParameters
    )
    (; ω) = step_sizes
    (; primal_weight_damping, zero_tol) = params
    Δx = norm(@. scratch.x = sol_cand.x - sol_restart.x)
    Δy = norm(@. scratch.y = sol_cand.y - sol_restart.y)
    θ = primal_weight_damping
    @trace if Δx > zero_tol && Δy > zero_tol
        res = exp(θ * log(Δy / Δx) + (1 - θ) * log(ω))
    else
        res = ω
    end
    return res
end

function LinearAlgebra.axpby!(
        α::Reactant.TracedRNumber, x::TracedRArray{T}, β::Reactant.TracedRNumber, y::TracedRArray{T}
    ) where {T}
    if length(x) != length(y)
        throw(
            DimensionMismatch(
                lazy"x has length $(length(x)), but y has length $(length(y))"
            ),
        )
    end
    T1 = Reactant.unwrapped_eltype(T)
    α = Reactant.promote_to(Reactant.TracedRNumber{T1}, α)
    β = Reactant.promote_to(Reactant.TracedRNumber{T1}, β)
    ax = @opcall multiply(x, Reactant.broadcast_to_size(α, size(x)))
    by = @opcall multiply(y, Reactant.broadcast_to_size(β, size(y)))

    set_mlir_data!(y, get_mlir_data(@opcall add(ax, by)))
    return y
end
