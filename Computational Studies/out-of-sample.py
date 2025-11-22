from make_case import make_case
import gurobipy as gp

seed = 50
n_vessels = None
n_wind_farms = None
n_scenarios = None

fixed_gamma_decisions = []
fixed_alpha_decisions = []

scenarios = []

for scenario in scenarios:
    
    model = make_case(
        f"out_of_sample",
        n_vessels,
        n_wind_farms,
        seed,
        True
        )

    #set model params
    model.Params.MIPGap = 0.002 #set gap to 0.2%
    model.Params.TimeLimit = 300 #set max solving time to 5 minutes
    model.Params.OutputFlag = 0 #turn off output

    #fix charter decisions
    for g_decision in fixed_gamma_decisions:
        var_name, value = g_decision.split(": ")
        var = model.getVarByName(var_name)
        var.setAttr("ub", float(value))
        var.setAttr("lb", float(value))

    for a_decision in fixed_alpha_decisions:
        var_name, value = a_decision.split(": ")
        var = model.getVarByName(var_name)
        var.setAttr("lb", float(value))
        var.setAttr("ub", float(value))

    model.optimize()

    #print results
    runtime = model.Runtime
    total_cost = model.ObjVal
    print(f"Out-of-sample optimization completed in {runtime} seconds.")
    print(f"Total cost: {total_cost}")