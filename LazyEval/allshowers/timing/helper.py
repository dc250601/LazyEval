import torch
import torch.nn.functional as F
import time

#############################################################################
#### Default Configs
PDG_LIST = [11,
            -11,
            22,
            130,
            211,
            -211,
            321,
            -321,
            2112,
            -2112,
            2212,
            -2212]



DEFAULT_WARMUP_STEPS = 20
DEFAULT_TIMING_STEPS = 100
DEFAULT_SAMPLING_ENERGY = range(10,100,10)
#############################################################################

def to_labels(pdg_codes,
              pdgs
             ):
    
    labels = torch.full(pdg_codes.shape, -1, dtype=torch.int64)
    
    for label, pdg in enumerate(pdgs):
        labels[pdg_codes == pdg] = label
    
    return labels

def prepare_conditions(
    energies,
    directions,
    labels,
    num_particles,
    correct_directions = False
):
    
    cond_list = [energies.reshape(-1, 1)]
    one_hot = F.one_hot(labels, num_classes=num_particles).to(torch.float32)
    cond_list.append(one_hot)
    cond_list.append(directions)
    result = torch.concatenate(cond_list, dim=1)
    
    if correct_directions:
        result[:, -3:-1] *= -1.0
    
    return result


def get_current_time(device = torch.device("cpu")):

    cuda_synchronize(device=device)
    
    return time.time()



def cuda_synchronize(device = torch.device("cpu")):

    if device.type == "cuda":
        torch.cuda.synchronize()