# using 2D vector in DP;
# Julia's matrix computation is already very faster;
# flattern matrix can't not be faster.
# 20 periods, 0.026s on the windows PC;
# 40 periods, 0.048s on Dell windows;

using Distributions

const mean_demand = 10
# demands = fill(mean_demand, T)
demands = [10, 20, 10, 20, 10, 20, 10, 20]
const T = length(demands)
const unit_order_cost = 1.0
const hold_cost = 2.0
const penalty_cost = 10.0
const minI = -100
const maxI = 100
const capacity = 150

struct DemandProb
	demand::Int
	prob::Float64
end

function getPMFPoisson(
	demands::Vector{Int},
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

function getPMFSelf(
	values::Vector{Int},
	probs::Vector{Float64},
)
	len = length(values)
    pmf = Vector{Vector{DemandProb}}(undef, T)
    for t in 1:T
        pmf[t] = Vector{DemandProb}(undef, len)
		for j in 1:len
			pmf[t][j] = DemandProb(
				values[j],
				probs[j]
			)
		end
	end
	return pmf
end


function main(capacity::Int = capacity, fix_order_cost::Int = 0)
	# capacity = 150
    truncQuantile = 0.9999

    ## poisson demand
    pmf = getPMFPoisson(
		demands,
		truncQuantile,
	)

    # ### self defined distribution
    # values = collect(10:10:100) # collect 能展开等差数列为数组
    # N = length(values)
	# probs = fill(1/N, N)
    # pmf = getPMFSelf(values, probs)


	num_inv = maxI - minI + 1
	start_time = time()
	

	# value[t+1, i]
	# t = 0,...,T
	# inventory = i + minI - 1

	########################################################
	# State cost
	########################################################

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

	#################################################
	# backward DP
	#################################################

	for t in T:-1:1
		for inventory in minI:maxI
			bestValue = Inf
			bestAction = 0

			for action in 0:capacity
				fixCost = action > 0 ? fix_order_cost : 0.0
				variCost = action * unit_order_cost
				expectedCost = 0.0

				for dp in pmf[t]
					demand = dp.demand
					prob = dp.prob
					nextInventory = clamp(
						inventory + action - demand,
						minI,
						maxI,
					)

					immediateCost =
						fixCost +
						variCost +
						holdBackorderCost[nextInventory-minI+1]

					idx_next = nextInventory - minI + 1
					futureCost =
						value[t+1, idx_next]
					expectedCost +=
						prob * (immediateCost + futureCost)
				end

				if expectedCost < bestValue
					bestValue = expectedCost
					bestAction = action
				end
			end

			idx = inventory - minI + 1
			value[t, idx] = bestValue
			policy[t, idx] = bestAction
		end
	end

	elapsed = time() - start_time

	initialInventory = 0
	idx0 = initialInventory - minI + 1
	optimalValue = value[1, idx0]

	println("planning horizon = $T")
	@printf("running time = .4f% seconds", elapsed)
	println("optimal value = $optimalValue")
	println(
		"optimal order at t=1, inventory=0 is: ",
		policy[1, idx0],
	)
end

main()
