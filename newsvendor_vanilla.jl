# 定义一个模块，并对外暴露接口
module Newsvendor
export NewsvendorDP, recursion, get_pmf_poisson, State

# using SpecialFunctions   # lgamma
using Statistics
using Distributions


function poisson_pmf(k::Int, λ::Float64)
    if k < 0 || λ < 0
        return 0.0
    elseif k == 0 && λ == 0
        return 1.0
    end
    logp = -λ + k * log(λ) - lgmma(k + 1)
    return exp(logp)
end

function poisson_quantile(p::Float64, λ::Float64)
    low = 0
    high = max(100, Int(3λ))
    while low < high
        mid = (low + high) ÷ 2
        if cdf(Poisson(λ), mid) < p
            low = mid + 1
        else
            high = mid
        end
    end
    return low
end

# --------------------------
# PMF truncation
# --------------------------

function get_pmf_poisson(demands::Vector{Float64}, q::Float64)
    T = length(demands)
    pmf = Vector{Vector{Tuple{Int, Float64}}}(undef, T)

    for t in 1:T
        ub = poisson_quantile(q, demands[t])
        lb = poisson_quantile(1 - q, demands[t])

        support = [(d, pdf(Poisson(demands[t]), d) / (2q - 1))
                   for d in lb:ub]

        pmf[t] = support
    end

    return pmf
end

# --------------------------
# State
# --------------------------

struct State
    period::Int
    inventory::Float64
end

Base.:(==)(a::State, b::State) = a.period == b.period && a.inventory == b.inventory
Base.hash(s::State, h::UInt) = hash((s.period, s.inventory), h)

# --------------------------
# Newsvendor DP
# --------------------------
# mutable 表示这个结构体的字段是可以被修改的
mutable struct NewsvendorDP
    T::Int
    capacity::Float64
    stepSize::Float64
    fix_cost::Float64
    var_cost::Float64
    hold_cost::Float64
    penalty_cost::Float64
    max_I::Float64
    min_I::Float64
    pmf::Vector{Vector{Tuple{Int, Float64}}}
    cache_actions::Dict{State, Float64}
    cache_values::Dict{State, Float64}
end

function feasible_actions(model::NewsvendorDP)
    Q = Int(model.capacity / model.stepSize)
    return [i * model.stepSize for i in 0:Q-1]
end

function transition(model::NewsvendorDP, s::State, a, d)
    nextI = s.inventory + a - d
    nextI = clamp(nextI, model.min_I, model.max_I)
    return State(s.period + 1, nextI)
end

function immediate_cost(model::NewsvendorDP, s::State, a, d)
    fix = a > 0 ? model.fix_cost : 0.0
    vari = a * model.var_cost

    nextI = clamp(s.inventory + a - d, model.min_I, model.max_I)

    hold = max(model.hold_cost * nextI, 0.0)
    penalty = max(-model.penalty_cost * nextI, 0.0)

    return fix + vari + hold + penalty
end

function recursion(model::NewsvendorDP, s::State)
    if haskey(model.cache_values, s)
        return model.cache_values[s]
    end

    best_val = Inf
    best_q = 0.0

    for a in feasible_actions(model)
        val = 0.0

        for (d, p) in model.pmf[s.period]
            val += p * immediate_cost(model, s, a, d)

            if s.period < model.T
                ns = transition(model, s, a, d)
                val += p * recursion(model, ns)
            end
        end

        if val < best_val
            best_val = val
            best_q = a
        end
    end

    model.cache_actions[s] = best_q
    model.cache_values[s] = best_val

    return best_val
end

# --------------------------
# main
# --------------------------

function main()
    T = 20
    mean_demand = 40.0
    demands = fill(mean_demand, T)

    capacity = 150.0
    stepSize = 1.0
    fix_cost = 0.0
    var_cost = 1.0
    hold_cost = 2.0
    penalty_cost = 10.0

    trunc_q = 0.9999
    maxI = 100.0
    minI = -100.0

    pmf = get_pmf_poisson(demands, trunc_q)

    model = NewsvendorDP(
        T, capacity, stepSize,
        fix_cost, var_cost,
        hold_cost, penalty_cost,
        maxI, minI, pmf,
        Dict(), Dict()
    )

    s0 = State(1, 0.0)

    t0 = time()
    val = recursion(model, s0)
    t1 = time()

    println("planning horizon = $T")
    println("runtime = $(t1 - t0) sec")
    println("optimal value = $val")
end


main()
end
