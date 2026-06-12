#=
ini_I = 0
vari_cost = 1
unit_back_cost = 10
unit_hold_cost = 2
mean_demands = [10, 20, 10, 20, 10, 20, 10, 20]
----
218.41 for sdp optimal cost, java 0.5s, c++ 0.008s, julia  0.0000s;

sample_num = 30 #
forward_num = 1 # 4
iter_num = 50

SDDP julia Highs 222.43 without enhancement, 1.76s;
SDDP C++ gurobi 219.46 without enhancement, 0.91s;
=#

using JuMP
using HiGHS
using Random
using Statistics
using Distributions

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
	mean_demand = 10.0
	# mean_demands = fill(mean_demand, T)
	mean_demands = [10, 20, 10, 20, 10, 20, 10, 20]
	T = length(mean_demands)

	ini_I = 0.0
	unit_vari_costs = fill(1.0, T)
	unit_holding_costs = fill(2.0, T)
	unit_backorder_costs = fill(10.0, T)

	sample_num = 30 #
	forward_num = 1 # 4
	iter_num = 50
	theta_lb = 0.0

	# sampling
	sample_nums = fill(sample_num, T)
	sample_details = [generateSamplesPoisson(sample_nums[t], mean_demands[t]) for t=1:T]
	# sample_details = [[5, 15], [5, 15]]

	# --------- build models ----------
	models = [Model(HiGHS.Optimizer) for _ ∈ 1:(T+1)]
	set_silent.(models)

	# decision variables
	q = Vector{VariableRef}(undef, T)
	I = Vector{VariableRef}(undef, T)
	B = Vector{VariableRef}(undef, T)
	theta = Vector{VariableRef}(undef, T)

	# 必须保存 ConstraintRef，否则无法修改
	# undef 是 undefined（未初始化）的缩写
	# 它是一个占位标记，专门用来告诉 Julia：“请先帮我把内存空间申请好，但暂时不要往里面填任何数据
	# 不能用push往里面赋值，可以直接赋值
	balance_constr = Vector{ConstraintRef}(undef, T)  # I-B=rhs
	benders_constr = [Vector{ConstraintRef}() for _ in 1:T]

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
		# scenario_paths = [[1, 1], [1, 2], [2, 1], [2, 2]]

		# ----- stage 0 -----
		m0 = models[1]

		@objective(m0, Min, unit_vari_costs[1]*q[1] + theta[1])

		if iter > 1
			push!(benders_constr[1], @constraint(m0, theta[1] >= slopes[iter-1][1, 1]*q[1] + intercepts[iter-1][1, 1]))
		end

		optimize!(m0)

		# write_to_file(m0, "iter" * string(iter) * ".lp")
		# open("iter" * string(iter) * ".txt", "w") do io
		# 	println(io, "Objective Value: ", objective_value(m0))
		# 	println(io, "Variable value of " * name(q[1]) * ": ", value(q[1]))
		# end

		for n in 1:forward_num
			q_values[iter][1, n] = value(q[1])
		end

		# ----- forward -----
		for t in 2:(T+1)
			m = models[t]

			if t < T+1 && iter > 1
				for n in 1:forward_num
					push!(benders_constr[t], @constraint(m, theta[t] >= slopes[iter-1][t, n]*(I[t-1] - B[t-1] + q[t]) +
																		intercepts[iter-1][t, n]))
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

				set_normalized_rhs(balance_constr[t-1], rhs) # 更改右端值

				optimize!(m)

				# write_to_file(m, "iter" * string(iter) * "_sub_" * string(t-1) * '^' * string(n) * ".lp")
				# open("iter" * string(iter) * "_sub_" * string(t-1) * '^' * string(n) * ".txt", "w") do io
				# 	println(io, "Objective Value: ", objective_value(m))
				# 	if t < T + 1
				# 		println(io, "Variable" * name(q[t]) * ": ", value(q[t]))
				# 		println(io, "Variable" * name(I[t-1]) * ": ", value(I[t-1]))
				# 		println(io, "Variable" * name(B[t-1]) * ": ", value(B[t-1]))
				# 	else
				# 		println(io, "Variable" * name(I[t-1]) * ": ", value(I[t-1]))
				# 		println(io, "Variable" * name(B[t-1]) * ": ", value(B[t-1]))
				# 	end
				# end

				I_values[iter][t-1, n] = value(I[t-1])
				B_values[iter][t-1, n] = value(B[t-1])

				if t <= T
					q_values[iter][t, n] = value(q[t])
				end
			end
		end

		# ----- backward -----
		for t in T + 1:-1:2
			for n in 1:forward_num

				S = length(sample_details[t-1])

				intercept_list = zeros(S)
				slope_list     = zeros(S)

				for s in 1:S
					demand = sample_details[t-1][s]

					rhs = if t == 2
						ini_I - demand + q_values[iter][t-1, n]
					else
						I_values[iter][t-2, n] -
						B_values[iter][t-2, n] +
						q_values[iter][t-1, n] -
						demand
					end

					set_normalized_rhs(balance_constr[t-1], rhs)
					optimize!(models[t])

					# cons = all_constraints(models[t+1], VariableRef, MOI.GreaterThan{Float64})
					# πs = [dual(con) for con in cons]
					# rhs = [constraint_object(con).set.lower for con in cons]
					# write_to_file(models[t], "iter" * string(iter) * "_sub_" * string(t-1) * '^' * string(n) * "-back.lp")


					pi1 = dual(balance_constr[t-1])
					pi2 = if iter > 1 && t < T + 1
						[dual(con) for con in benders_constr[t]]
					end
					rsh2 = if iter > 1 && t < T + 1
						[constraint_object(con).set.lower for con in benders_constr[t]]
					end

					# if t == 2 && iter == 2
					# 	println()
					# end


					slope_list[s] = pi1
					intercept_list[s] = -pi1 * demand
					if iter > 1 && t < T + 1
						intercept_list[s] += sum(pi2 .* rsh2)
					end
				end

				slopes[iter][t-1, n]     = mean(slope_list)
				intercepts[iter][t-1, n] = mean(intercept_list)
			end
		end
	println("iter " * string(iter) * " value: " * string(objective_value(models[1])))
	end

	final_value = objective_value(models[1])
	Q1 = q_values[end][1, 1]

	return final_value, Q1
end

start_time = time()
solve_newsvendor()
elapsed_time = time() - start_time
println("running time is $elapsed_time seconds")
