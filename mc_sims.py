import os
import matplotlib.pyplot as plt
import pandas as pd
from main import *
import shutil
from tqdm import tqdm
import warnings
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_environment(parameter_sets):
    base_dir = "simulation_data"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    for i, params in enumerate(parameter_sets, start=1):
        param_dir = os.path.join(base_dir, f"param_set_{i}")
        os.makedirs(param_dir)
        with open(os.path.join(param_dir, "parameters.txt"), "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
                
        for subfolder in ["pnl", "net_position", "true_values", "expected_vals", "bids", "asks"]:
            os.makedirs(os.path.join(param_dir, subfolder))

# informed = 0, noisy_informed = 0, noisy = 0, stoch_noisy = 0, mr = 0, mom = 0
parameter_sets = [{"informed": 1, "noisy_informed": 2, "noisy": 2, "stoch_noisy": 0, "mr": 1, "mom": 0}, #3
                  {"informed": 1, "noisy_informed": 2, "noisy": 2, "stoch_noisy": 0, "mr": 0, "mom": 1}, #3
                  {"informed": 0, "noisy_informed": 6, "noisy": 0, "stoch_noisy": 0, "mr": 0, "mom": 0}, #3
                  {"informed": 0, "noisy_informed": 0, "noisy": 0, "stoch_noisy": 6, "mr": 0, "mom": 0}] #3

# parameter_sets = [{"informed": 0, "noisy_informed": 0, "noisy": 6, "stoch_noisy": 0, "mr": 0, "mom": 0}, #1
#                   {"informed": 1, "noisy_informed": 0, "noisy": 5, "stoch_noisy": 0, "mr": 0, "mom": 0}, #1
#                   {"informed": 1, "noisy_informed": 5, "noisy": 0, "stoch_noisy": 0, "mr": 0, "mom": 0}, #1 
#                   {"informed": 1, "noisy_informed": 2, "noisy": 3, "stoch_noisy": 0, "mr": 0, "mom": 0}, #1
#                   {"informed": 6, "noisy_informed": 0, "noisy": 0, "stoch_noisy": 0, "mr": 0, "mom": 0}, #2
#                   {"informed": 1, "noisy_informed": 1, "noisy": 1, "stoch_noisy": 1, "mr": 1, "mom": 1}, #2
#                   {"informed": 0, "noisy_informed": 0, "noisy": 0, "stoch_noisy": 2, "mr": 2, "mom": 2}, #2
#                   {"informed": 1, "noisy_informed": 2, "noisy": 2, "stoch_noisy": 1, "mr": 0, "mom": 0}, #2
#                   {"informed": 1, "noisy_informed": 2, "noisy": 2, "stoch_noisy": 0, "mr": 1, "mom": 0}, #3
#                   {"informed": 1, "noisy_informed": 2, "noisy": 2, "stoch_noisy": 0, "mr": 0, "mom": 1}, #3
#                   {"informed": 0, "noisy_informed": 6, "noisy": 0, "stoch_noisy": 0, "mr": 0, "mom": 0}, #3
#                   {"informed": 0, "noisy_informed": 0, "noisy": 0, "stoch_noisy": 6, "mr": 0, "mom": 0}, #3
#                   {"informed": 0, "noisy_informed": 0, "noisy": 0, "stoch_noisy": 0, "mr": 6, "mom": 0}, #4
#                   {"informed": 0, "noisy_informed": 0, "noisy": 0, "stoch_noisy": 0, "mr": 0, "mom": 6}, #4
#                   {"informed": 2, "noisy_informed": 2, "noisy": 2, "stoch_noisy": 0, "mr": 0, "mom": 0}] #4

prepare_environment(parameter_sets)

def aggregate_results(model, sim_num):
    data = {
        'Simulation': [sim_num] * list(range(100)),
        'TrueValue': model.god.return_data()[0],
        'ExpectedValue': model.god.return_data()[1],
        'Bids' : model.god.return_data()[2], 
        'Asks' : model.god.return_data()[3]
    }

    for agent_id, agent in model.directory.items():
        if agent_id == 1:
            col_pnl = f"agent1_MM_pnl"
            col_pos = f"agent1_MM_pos"
        else:
            strategy_type = type(agent.strategy).__name__
            col_pnl = f"agent{agent_id}_{strategy_type}_pnl" 
            col_pos = f"agent{agent_id}_{strategy_type}_pos"
        
        data[col_pnl] = agent.pnl_over_time
        data[col_pos] = agent.pos_over_time

    return pd.DataFrame([data])

def run_simulation(parameter_set, set_index, num_simulations=100):
    base_dir = f"simulation_data/param_set_{set_index}"
    results = pd.DataFrame()
    
    print(f'running set {parameter_set}')
    for sim_num in tqdm(range(0, num_simulations), desc = "Outer number of simulations", position = 0):  
        model = MarketModel(6, **parameter_set)
        for i in tqdm(range(100), desc = "Inner sims inside a sim", position = 1, miniters = 1):
            model.step()
        
        sim_results = aggregate_results(model, sim_num)
        results = pd.concat([results, sim_results], ignore_index=True)

    results.to_csv(os.path.join(base_dir, "aggregated_results.csv"), index=False)
        
for i, params in enumerate(parameter_sets, start=1):
    print(f'On the {i}th parameter set')
    run_simulation(params, i)

print("\n Simulation runs complete. \n")
