import numpy as np
import showerdata
from LazyEval.core import LazyShowerData

def filter_event_by_energy(event,energy_range):
    
    idx = _get_energies(event.energies,energy_range)

    filtered = showerdata.Showers(points=event.points[idx],
                                  energies=event.energies[idx],
                                  directions=event.directions[idx],
                                  pdg=event.pdg[idx])
    return filtered


    

def lazy_filter_event_by_energy(lazydata,energy_range):

    assert isinstance(lazydata,LazyShowerData), "For lazy processing Lazy Shower Data format is necessary"

    energy_values = lazydata.energy[:]
    idx = _get_energies(energy_values,energy_range)

    return np.where(idx)[0]


def _get_energies(energy_list,energy_range):
    cut_low = 0

    if isinstance(energy_range,list):
        assert len(energy_range) == 2, "Invalid length of energy_range list found"
        cut_low = energy_range[0]
        cut_high = energy_range[1]
    else:
        cut_high = energy_range

    idx_low = energy_list > cut_low
    idx_high = energy_list <= cut_high

    idx = idx_low*idx_high
    idx = idx.squeeze()

    return idx
    