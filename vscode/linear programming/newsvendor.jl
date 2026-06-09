using JuMP
using HiGHS
using Random
using Statistics

# --------- sampling functions ----------
function generateSamplesPoisson(N, mean)
	return rand(Poisson(mean), N)
end

# random scenario path
function generateScenarioPaths(N, sample_nums)
	T = length(sample_nums)
	paths = [[rand(1:sample_nums[t]) for t in 1:T] for _ ∈ 1:N]
	return paths
end

# --------- main solve ----------
function solve_newsvendor()
	# parameters
	ini_I = 0.0
	T = 2
	unit_vari_costs = fill(1.0, T)
	unit_holding_costs = fill(2.0, T)
	unit_backorder_costs = fill(10.0, T)

	mean_demand = 10.0
	mean_demands = fill(mean_demand, T)

	sample_num = 2 #
	forward_num = 4 #
	iter_num = 50
	theta_lb = 0.0

	# sampling
	sample_nums = fill(sample_num, T)
	# sample_details = [generateSamplesPoisson(sample_nums[t], mean_demands[t]) for t=1:T]
	sample_details = [[5, 15], [5, 15]]

	# --------- build models ----------
	models = [Model(HiGHS.Optimizer) for _ ∈ 1:(T+1)]
	set_silent.(models)

	# decision variables
	q = Vector{VariableRef}(undef, T)
	I = Vector{VariableRef}(undef, T)
	B = Vector{VariableRef}(undef, T)
	theta = Vector{VariableRef}(undef, T)

	# 必须保存 ConstraintRef，否则无法修改
	balance_constr = Vector{ConstraintRef}(undef, T)  # I-B=rhs

	for t in 1:(T+1)
		m = models[t]

		if t <= T
			q[t] = @variable(m, lower_bound=0, base_name="q_"*string(t))
			theta[t] = @variable(m, lower_bound=theta_lb, base_name="theta_"*string(t))
		end

		if t > 1
			I[t-1] = @variable(m, lower_bound=0, base_name="I_"*string(t-1))
			B[t-1] = @variable(m, lower_bound=0, base_name="B_"*string(t-1))

			balance_constr[t-1] = @constraint(m, I[t-1] - B[t-1] == 0)
		end
	end

	# --------- storage ----------
	# 这样写有利于利用 Julia 的矩阵操作优势
	intercepts = [zeros(T, forward_num) for _ ∈ 1:iter_num]
	slopes     = [zeros(T, forward_num) for _ ∈ 1:iter_num]

	q_values = [zeros(T, forward_num) for _ ∈ 1:iter_num]
	I_values = [zeros(T, forward_num) for _ ∈ 1:iter_num]
	B_values = [zeros(T, forward_num) for _ ∈ 1:iter_num]

	# --------- iteration ----------
	for iter in 1:iter_num

		scenario_paths = generateScenarioPaths(forward_num, sample_nums)
		scenario_paths = [[1, 1], [1, 2], [2, 1], [2, 2]]

		# ----- stage 0 -----
		m0 = models[1]

		@objective(m0, Min, unit_vari_costs[1]*q[1] + theta[1])

		if iter > 1
			@constraint(m0, theta[1] >= slopes[iter-1][1, 1]*q[1] +
										intercepts[iter-1][1, 1])
		end

		optimize!(m0)
		write_to_file(m0, "iter" * string(iter) * ".lp")

		for n in 1:forward_num
			q_values[iter][1, n] = value(q[1])
		end

		# ----- forward -----
		for t in 2:(T+1)
			m = models[t]

			if t < T+1 && iter > 1
				for n in 1:forward_num
					@constraint(m, theta[t] >= slopes[iter-1][t, n]*q[t] +
											   intercepts[iter-1][t, n])
				end
			end

			for n in 1:forward_num

				demand = sample_details[t-1][scenario_paths[n][t-1]]

				if t < T+1
					@objective(m, Min,
						unit_vari_costs[t]*q[t] +
						unit_backorder_costs[t-1]*B[t-1] +
						unit_holding_costs[t-1]*I[t-1] +
						theta[t]
					)
				else
					@objective(m, Min,
						unit_backorder_costs[t-1]*B[t-1] +
						unit_holding_costs[t-1]*I[t-1]
					)
				end

				rhs = if t == 2
					ini_I - demand + q_values[iter][t-1, n]
				else
					I_values[iter][t-2, n] -
					B_values[iter][t-2, n] +
					q_values[iter][t-1, n] -
					demand
				end

				set_normalized_rhs(balance_constr[t-1], rhs)

				optimize!(m)
				write_to_file(m, "iter" * string(iter) * "_sub_" * string(t) * '^' * string(n) * ".lp")

				I_values[iter][t-1, n] = value(I[t-1])
				B_values[iter][t-1, n] = value(B[t-1])

				if t <= T
					q_values[iter][t, n] = value(q[t])
				end
			end
		end

		# ----- backward -----
		for t in T:-1:1
			for n in 1:forward_num

				S = length(sample_details[t])

				intercept_list = zeros(S)
				slope_list     = zeros(S)

				for s in 1:S
					demand = sample_details[t][s]

					rhs = if t == 1
						ini_I - demand + q_values[iter][t, n]
					else
						I_values[iter][t-1, n] -
						B_values[iter][t-1, n] +
						q_values[iter][t, n] -
						demand
					end

					set_normalized_rhs(balance_constr[t], rhs)
					optimize!(models[t+1])

					π = dual(balance_constr[t])

					slope_list[s] = π
					intercept_list[s] = -π * demand
				end

				slopes[iter][t, n]     = mean(slope_list)
				intercepts[iter][t, n] = mean(intercept_list)
			end
		end
	end

	final_value = objective_value(models[1])
	Q1 = q_values[end][1, 1]

	return final_value, Q1
end

solve_newsvendor()
