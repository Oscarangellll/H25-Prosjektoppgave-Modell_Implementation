import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from make_case import make_case

solutions = [
    ['gamma_LT[CTV]: 1.0', 'gamma_LT[SOV]: 1.0'],
    ['gamma_ST[CTV,February]: 1.0', 'gamma_LT[CTV]: 1.0', 'gamma_LT[SOV]: 1.0']
]

avg = []

for solution in solutions:
    results = []
    for seed in range(101, 1001):
        model = make_case(
            name = "name",
            n_vessels = 3,
            n_wind_farms = 2,
            n_scenarios = 1,
            seed = [seed],
            scenario = True
            )
        model.update()
        #set all ub to 0
        for v in model.getVars():
            if v.VarName.startswith("gamma"):
                v.setAttr("ub", 0)
        #fix active decisions
        for g_decision in solution:
            var_name, value = g_decision.split(": ")
            var = model.getVarByName(var_name)
            var.setAttr("ub", float(value))
            var.setAttr("lb", float(value))
        model.Params.MIPGap = 0.002 #set gap to 0.2%
        model.Params.TimeLimit = 7200 #set max solving time to 2 hours
        model.Params.OutputFlag = 0 #turn off output
        model.update()    
        model.optimize()
        results.append(model.ObjVal)
        
    avg.append(sum(results) / len(results))

#save average to csv
import csv
with open("OOS_4_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Solution", "Average Objective Value"])
    for sol, avg_val in zip(solutions, avg):
        writer.writerow([sol, avg_val])