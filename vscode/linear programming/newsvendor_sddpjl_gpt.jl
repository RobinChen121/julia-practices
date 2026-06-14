using SDDP
using HiGHS
using Statistics

# ==========================
# Parameters
# ==========================

stage_num = 12

# cost parameters
c = 2.0      # ordering cost
h = 1.0      # holding cost
p = 5.0      # backlog penalty

# ordering capacity
capacity = 20.0

# demand distribution
demand_support = [5.0, 10.0, 15.0]
demand_prob = [0.3, 0.4, 0.3]

# ==========================
# Model
# ==========================

model = SDDP.LinearPolicyGraph(
    stages = stage_num,
    sense = :Min,
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
) do sp, t

    # inventory state
    @variable(
        sp,
        inventory,
        SDDP.State,
        initial_value = 0.0
    )

    # order quantity
    @variable(
        sp,
        0 <= q <= capacity
    )

    # positive inventory
    @variable(
        sp,
        pos_inventory >= 0
    )

    # backlog
    @variable(
        sp,
        backlog >= 0
    )

    SDDP.parameterize(
        sp,
        demand_support,
        demand_prob,
    ) do d

        # inventory balance
        @constraint(
            sp,
            inventory.out ==
            inventory.in + q - d
        )

        # pos_inventory = max(inventory.out,0)
        @constraint(
            sp,
            pos_inventory >= inventory.out
        )

        # backlog = max(-inventory.out,0)
        @constraint(
            sp,
            backlog >= -inventory.out
        )

        @stageobjective(
            sp,
            c * q +
            h * pos_inventory +
            p * backlog
        )
    end
end

# ==========================
# Training
# ==========================

SDDP.train(
    model;
    iteration_limit = 500,
)

