using SDDP
using JuMP
using HiGHS
using Statistics
using Random
using Distributions

"""
多阶段随机库存模型（有限期）

状态变量:
    inventory_t : 期末净库存（可为负，负值表示backlog）

决策变量:
    order_t     : 本期订货量

随机变量:
    demand_t    : 本期需求（离散场景）

阶段成本:
    order_cost * order
  + hold_cost * positive_inventory
  + shortage_cost * backlog

终端处理:
    最后一阶段增加终端价值：
    - salvage_value * positive_inventory
    + terminal_backlog_cost * backlog
"""
function build_inventory_sddp(;
    T::Int = 10,                             # 期数
    x0::Float64 = 0.0,                    # 初始库存
    order_cost::Float64 = 1.0,             # 单位订货成本
    hold_cost::Float64 = 2.0,              # 单位持有成本
    shortage_cost::Float64 = 10.0,          # 单位缺货/积压成本
    salvage_value::Float64 = 0.0,          # 终端剩余库存残值
    terminal_backlog_cost::Float64 = 0.0,  # 终端backlog罚成本
    max_inventory::Float64 = 200.0,        # 库存上界
    max_backlog::Float64 = 200.0,          # backlog上界（净库存下界=-max_backlog）
    max_order::Float64 = 150.0,            # 单期最大订货量
    demand_support::Vector{Float64} = [10.0, 20.0, 30.0, 40.0, 50.0],
    demand_prob::Vector{Float64} = [0.10, 0.20, 0.40, 0.20, 0.10],
)
    @assert length(demand_support) == length(demand_prob)
    @assert abs(sum(demand_prob) - 1.0) <= 1e-8

    model = SDDP.LinearPolicyGraph(
        stages = T,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t

        # 状态变量：净库存，可正可负
        @variable(
            sp,
            -max_backlog <= inventory <= max_inventory,
            SDDP.State,
            initial_value = x0
        )

        # 决策变量：订货量
        @variable(sp, 0 <= order <= max_order)

        # 随机需求（通过 parameterize 固定）
        @variable(sp, demand)

        # 用两个非负变量分解净库存:
        # inventory.out = pos_inv - backlog
        @variable(sp, pos_inv >= 0)
        @variable(sp, backlog >= 0)

        # 库存平衡
        @constraint(sp, inventory.out == inventory.in + order - demand)

        # 正库存 / 积压分解
        @constraint(sp, inventory.out == pos_inv - backlog)

        # 阶段目标
        if t < T
            @stageobjective(
                sp,
                order_cost * order +
                hold_cost * pos_inv +
                shortage_cost * backlog
            )
        else
            # 终端阶段加残值/清算项
            @stageobjective(
                sp,
                order_cost * order +
                hold_cost * pos_inv +
                shortage_cost * backlog -
                salvage_value * pos_inv +
                terminal_backlog_cost * backlog
            )
        end

        # 离散需求场景
        SDDP.parameterize(sp, demand_support, demand_prob) do ω
            fix(demand, ω)
            return
        end
    end
    println(demand_support)
    return model
end

function solve_inventory_model()
    Random.seed!(2026)

    model = build_inventory_sddp(
        T = 1,
        x0 = 0.0,
        order_cost = 1.0,
        hold_cost = 2.0,
        shortage_cost = 10.0,
        salvage_value = 0.0,
        terminal_backlog_cost = 0.0,
        max_inventory = 200.0,
        max_backlog = 150.0,
        max_order = 120.0,
        demand_support = float(rand(Poisson(20), 10)),
        demand_prob = fill(0.1, 10),
    )

    # 训练 SDDP 策略
    SDDP.train(
        model,
        iteration_limit = 150,
        # log_frequency = 25,
    )

    # println("\n====================")
    # println("SDDP lower bound = ", SDDP.calculate_bound(model))
    # println("====================\n")

    # 仿真评估策略
    simulations = SDDP.simulate(model, 300, [:inventory, :order])

    total_costs = [
        sum(stage[:stage_objective] for stage in sim)
        for sim in simulations
    ]

    println("平均仿真总成本 = ", mean(total_costs))
    println("最小仿真总成本 = ", minimum(total_costs))
    println("最大仿真总成本 = ", maximum(total_costs))

    println("\n---- 第一条仿真路径（部分输出） ----")
    for (t, stage) in enumerate(simulations[1])
        inv_out = stage[:inventory].out
        ord = stage[:order]
        obj = stage[:stage_objective]
        println("阶段 $t: 订货=$(round(ord, digits=2)), 期末库存=$(round(inv_out, digits=2)), 阶段成本=$(round(obj, digits=2))")
    end

    return model
end

solve_inventory_model()