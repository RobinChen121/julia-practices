# 这里面的 x_buy 不知道为啥是一个状态变量，
# 里面也没给出它的状态转移方程

using SDDP
import Distributions
import HiGHS
import Plots
import Statistics

const x_0 = 10        # initial inventory
const c = 35          # unit ordering cost
const h = 1           # unit inventory holding cost
const p = 15          # unit shortage cost of backordering

const T = 10 # number of stages

Ω = range(0, 800; length = 20) # uniform distribution

model = SDDP.LinearPolicyGraph(;
    stages = T + 1,
    sense = :Min,
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
) do sp, t
    @variable(sp, x_inventory >= 0, SDDP.State, initial_value = x_0) # positive inventory
    @variable(sp, x_demand >= 0, SDDP.State, initial_value = 0) # x_demand is the backordering quantity
    # u_buy is a Decision-Hazard control variable. We decide u.out for use in
    # the next stage
    @variable(sp, u_buy >= 0, SDDP.State, initial_value = 0)# 进货量
    @variable(sp, u_sell >= 0) # decision variable
    @variable(sp, w_demand == 0) # random variable
    @constraint(sp, x_inventory.out == x_inventory.in + u_buy.in - u_sell)
    @constraint(sp, x_demand.out == x_demand.in + w_demand - u_sell)
    if t == 1
        fix(u_sell, 0; force = true)
        @stageobjective(sp, c * u_buy.out)
    elseif t == T + 1
        fix(u_buy.out, 0; force = true)
        @stageobjective(sp, -c * x_inventory.out + c * x_demand.out)
        SDDP.parameterize(ω -> fix(w_demand, ω), sp, Ω)
    else
        @stageobjective(sp, c * u_buy.out + h * x_inventory.out + p * x_demand.out)
        SDDP.parameterize(ω -> fix(w_demand, ω), sp, Ω)
    end
    return
end

SDDP.train(model)
simulations = SDDP.simulate(model, 200, [:x_inventory, :u_buy])
objective_values = [sum(t[:stage_objective] for t in s) for s in simulations]
μ, ci = round.(SDDP.confidence_interval(objective_values, 1.96); digits = 2)
lower_bound = round(SDDP.calculate_bound(model); digits = 2)
println("Confidence interval: ", μ, " ± ", ci)
println("Lower bound: ", lower_bound)

plt = SDDP.publication_plot(
    simulations;
    title = "x_inventory.out + u_buy.out",
    xlabel = "Stage",
    ylabel = "Quantity",
    ylims = (0, 1_000),
) do data
    return data[:x_inventory].out + data[:u_buy].out
end