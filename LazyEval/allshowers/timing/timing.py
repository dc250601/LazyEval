import numpy as np
import torch
from tqdm.auto import tqdm

from .helper import (
    DEFAULT_SAMPLING_ENERGY,
    DEFAULT_TIMING_STEPS,
    DEFAULT_WARMUP_STEPS,
    PDG_LIST,
    get_current_time,
    prepare_conditions,
    to_labels,
)

@torch.inference_mode()
def time_generation(
    cnf_transformer,
    point_count_fm,
    pdg_code,
    batch_size,
    device = "cpu",
    pdg_list = PDG_LIST,
    num_warmup = DEFAULT_WARMUP_STEPS,
    num_timing_steps = DEFAULT_TIMING_STEPS,
    energy_steps = DEFAULT_SAMPLING_ENERGY,
):

    
    if isinstance(device, str):
        device = torch.device(device)

    pdg_list = list(pdg_list)
    energies = torch.full((batch_size, 1), 10, device=device)
    pdg_ids = torch.full((batch_size,), pdg_code)
    labels = to_labels(pdg_ids, pdg_list).to(device)
    directions = torch.tensor([[0.0, 0.0, 1.0]] * batch_size, device=device)

    
    for _ in tqdm(range(num_warmup)):
        point_count_fm_input = prepare_conditions(energies,
                                                      directions,
                                                      labels,
                                                      len(pdg_list))
        num_points = point_count_fm(point_count_fm_input)
        num_points = (num_points.clamp(min=0) + 0.5).to(torch.int64)
        samples = cnf_transformer(energies, num_points, directions, labels)

    ## Time_PFM,Time_CNF,Time_Total    
    times = np.zeros((len(energy_steps),num_timing_steps,3))
    

    
    for e,energy in tqdm(enumerate(energy_steps)):
        
        energies = torch.full((batch_size, 1), energy, device=device)
        
        
        for i in range(num_timing_steps):


            
            ###### This will compute the complete time without sync
            time_begin_all = get_current_time(device)
            point_count_fm_input = prepare_conditions(energies,
                                                      directions,
                                                      labels,
                                                      len(pdg_list))
            
            
            num_points = point_count_fm(point_count_fm_input)
            
            num_points = (num_points.clamp(min=0) + 0.5).to(torch.int64)
            
            samples = cnf_transformer(energies, num_points, directions, labels)
            time_end_all = get_current_time(device)
            ################################################
            times[e,i,-1] = time_end_all - time_begin_all           
            ################################################
            time_b = get_current_time(device)
            _ = point_count_fm(point_count_fm_input)
            time_e = get_current_time(device)
            ################################################
            times[e,i,0] = time_e - time_b
            ################################################
            time_b = get_current_time(device)
            _ = cnf_transformer(energies, num_points, directions, labels)
            time_e = get_current_time(device)
            ################################################
            times[e,i,1] = time_e - time_b

    times /= batch_size

    return times, np.mean(times, 1), np.mean(times,(0,1))