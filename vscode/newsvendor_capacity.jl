# using 2D vector in DP;
# Julia's matrix computation is already very faster;
# flattern matrix can't not be faster.
# 20 periods, 0.026s on the windows PC;
# 40 periods, 0.048s on Dell windows;

using Distributions

##########################
# global parameters
# 全局变量会影响计算速度，使用 const 才能消除对计算速度的影响
const T = 10
const mean_demand = 20.0
const demands = fill(mean_demand, T)

const fix_order_cost = 0.0
const unit_order_cost = 1.0
const hold_cost = 2.0
const penalty_cost = 10.0

const truncQuantile = 0.9999
const minI = -100
const maxI = 100
const num_inv = maxI - minI + 1
################################

struct DemandProb
	demand::Int
	prob::Float64
end

function getPMF(
	demands::Vector{Float64},
	truncated_quantile::Float64,
)
	T = length(demands)

	support_lb = Vector{Int}(undef, T)
	support_ub = Vector{Int}(undef, T)

	pmf = Vector{Vector{DemandProb}}(undef, T)

	for t in 1:T
		dist = Poisson(demands[t])
		support_lb[t] = quantile(dist, 1 - truncated_quantile)
		support_ub[t] = quantile(dist, truncated_quantile)
		len = support_ub[t] - support_lb[t] + 1
		pmf[t] = Vector{DemandProb}(undef, len)

		for j in 1:len
			demand = support_lb[t] + j - 1
			pmf[t][j] = DemandProb(
				demand,
				pdf(dist, demand) / (2 * truncated_quantile - 1),
			)
		end
	end

	return pmf
end


function newsvendor(capacity::Int)
	start_time = time()
	pmf = getPMF(
		demands,
		truncQuantile,
	)

	holdBackorderCost = zeros(Float64, num_inv)
	for inventory in minI:maxI
		idx = inventory - minI + 1
		if inventory > 0
			holdBackorderCost[idx] =
				hold_cost * inventory
		elseif inventory < 0
			holdBackorderCost[idx] =
				-penalty_cost * inventory
		end
	end

	value = zeros(Float64, T + 1, num_inv)
	policy = zeros(Int, T, num_inv)
	valueG = zeros(Float64, T + 1, num_inv)

	#################################################
	# backward DP
	#################################################

	for t in T:-1:1
		for inventory in minI:maxI
			bestValue = Inf
			bestValueG = Inf
			bestAction = 0

			for action in 0:capacity
				fixCost = action > 0 ? fix_order_cost : 0.0
				variCost = action * unit_order_cost
				if t == 1
					fixCostG = 0.0
					variCostG = inventory * unit_order_cost
				end
				expectedCost = 0.0
				expectedCostG = 0.0

				for dp in pmf[t]
					demand = dp.demand
					prob = dp.prob
					nextInventory = clamp(
						inventory + action - demand,
						minI,
						maxI,
					)
					if t == 1
						nextInventoryG = clamp(
							inventory - demand,
							minI,
							maxI,
						)
					end

					immediateCost =
						fixCost +
						variCost +
						holdBackorderCost[nextInventory-minI+1]
					if t == 1
						immediateCostG =
							fixCostG +
							variCostG +
							holdBackorderCost[nextInventoryG-minI+1]
					end

					idx_next = nextInventory - minI + 1
					futureCost =
						value[t+1, idx_next]
					expectedCost +=
						prob * (immediateCost + futureCost)
					if t == 1
						idx_nextG = nextInventoryG - minI + 1
						futureCostG =
							value[t+1, idx_nextG]
						expectedCostG +=
							prob * (immediateCostG + futureCostG)
					end
				end

				if expectedCost < bestValue
					bestValue = expectedCost
					bestAction = action
				end
				if t == 1
					if expectedCostG < bestValueG
						bestValueG = expectedCostG
					end
				end
			end

			idx = inventory - minI + 1
			value[t, idx] = bestValue
			policy[t, idx] = bestAction
			if t == 1
				valueG[t, idx] = bestValueG
			else
				valueG[t, idx] = bestValue
			end
		end
	end

	elapsed = time() - start_time
	initialInventory = 0
	idx0 = initialInventory - minI + 1
	optimalValue = value[1, idx0]

	println("planning horizon = $T")
	println("running time = $elapsed seconds")
	println("optimal value = $optimalValue")
	println(
		"optimal Q at t=1, inventory=0 is: ",
		policy[1, idx0],
	)

	Cs = value[1, :]
	Qs = policy[1, :]
	Gs = valueG[1, :]
	return Cs, Qs, Gs
end

# println(newsvendor(150))
########################
# draw picture
using Plots
# using PyPlot

# 交互式绘图
plotly() # run in REPL 中交互式效果好

Is = minI:maxI
Cs, Qs, Gs = newsvendor(150)
p1 = plot(Is, Cs, label = "C", ylabel = "C")
p2 = plot(Is, Qs, label = "Q", ylabel = "Q")
p3 = plot(Is, Gs, label = "G", ylabel = "G")

p = plot(p1, p2, p3, layout = (3, 1), size = (1000, 900))
display(p)