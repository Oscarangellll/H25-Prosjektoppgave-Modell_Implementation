from make_case import make_case
import gurobipy as gp

#3x3x8-10

starting_seed = 50
vessels = range(3, 4) #runs for n_vessels = 3
wind_farms = range(3, 4) #runs for n_wind_farms = 3
scenarios = range(8, 11) #runs for n_scenarios = 8, 9, ..., 10
running_instances = 5 #number of instances per VxWxS combination

runtime_results = {(v, w, s): [] for v in vessels for w in wind_farms for s in scenarios}
objval_results = {(v, w, s): [] for v in vessels for w in wind_farms for s in scenarios}
charter_decisions = {(v, w, s): [] for v in vessels for w in wind_farms for s in scenarios}

for n_vessels in vessels:
    for n_wind_farms in wind_farms:
        for n_scenarios in scenarios:
            seed = starting_seed
            for running_instance in range(1, running_instances + 1):
                model = make_case(
                    f"run_instance {running_instance} for VxWxS = {n_vessels}x{n_wind_farms}x{n_scenarios}",
                    n_vessels,
                    n_wind_farms,
                    n_scenarios,
                    seed,
                    False
                    )
                
                print(f"Optimizing model for VxWxS = {n_vessels}x{n_wind_farms}x{n_scenarios}, instance {running_instance} with seed {seed}")
                
                #set model params
                model.Params.MIPGap = 0.002 #set gap to 0.2%
                model.Params.TimeLimit = 7200 #set max solving time to 5 minutes
                model.Params.OutputFlag = 0 #turn off output
                
                model.optimize()
                
                #store report data
                runtime = model.Runtime
                runtime_results[(n_vessels, n_wind_farms, n_scenarios)].append(runtime)
                total_cost = model.ObjVal
                objval_results[(n_vessels, n_wind_farms, n_scenarios)].append(total_cost)
                chartered_vessels = []
                for v in model.getVars():
                    if v.X > 0:
                        if "gamma" in v.VarName:
                            chartered_vessels.append(f"{v.VarName}: {v.X}")
                charter_decisions[(n_vessels, n_wind_farms, n_scenarios)].append(chartered_vessels)                
                            
                seed += 1
            n_scenarios += 1
        n_wind_farms += 1
        
# write results to csv file
import csv
with open("runtime_analysis_results 3x3x8_10.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["n_vessels", "n_wind_farms", "n_scenarios", "instance", "runtime", "objval", "charter_decision"])
    for key in runtime_results.keys():
        for instance_index in range(running_instances):
            n_vessels, n_wind_farms, n_scenarios = key
            runtime = runtime_results[key][instance_index]
            objval = objval_results[key][instance_index]
            charter = charter_decisions[key][instance_index]
            writer.writerow([n_vessels, n_wind_farms, n_scenarios, instance_index + 1, runtime, objval, charter])