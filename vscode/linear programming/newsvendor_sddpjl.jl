using HiGHS
using SDDP
using JuMP
using Statistics
using Random # 设置随机数种子的
using Distributions

"""
多阶段随机库存模型（有限期）
"""

const T = 1
const mean_demands = fill(20, T)
const x0 = 0.0   # initial inventory                 
const unit_order_cost = 1.0
const unit_hold_cost = 2.0
const unit_backorder_cost = 10.0

const num_support = 100
const iteration_limit = 200
const demand_supports = [rand(Poisson(mean_demands[t]), num_support) for t in 1:T]
const probs = [fill(1.0/num_support, num_support) for t in 1:T]


# 使用 do 就避免单独定义一个 subproblem 函数
model = SDDP.LinearPolicyGraph(
	stages = T,
	# :Min 前面的冒号表示这是一个 Symbol（符号） 类型
	# Symbol 是一种不可变的、高效的文本标识符。
	# 它可以粗略地理解为一种“轻量级的字符串”（String），但它在底层和普通的字符串有很大的区别
	sense = :Min,
	lower_bound = 0.0,
	optimizer = HiGHS.Optimizer,
) do sub_problem::Model, stage::Int

	# stagte variables：初始库存
	@variable(sub_problem, x, SDDP.State, initial_value = x0)

	# control variables：订货量
	@variables(sub_problem,
		begin
			q >= 0
			x_plus >= 0
			x_backorder >= 0
		end)

	# random variables: 需求
	@variable(sub_problem, d)
	SDDP.parameterize(sub_problem, demand_supports[stage], probs[stage]) do ω
		fix(d, ω)
		return
	end

	# transition function and constraints
	@constraints(
		sub_problem,
		begin
			x.out == x.in + q - d
			x.out == x_plus - x_backorder
		end
	)

	# stage objective
	@stageobjective(sub_problem, unit_order_cost * q +
								 unit_hold_cost * x_plus +
								 unit_backorder_cost * x_backorder
	)
end

SDDP.train(model, iteration_limit = iteration_limit)