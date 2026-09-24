import numpy as np
import showerdata
from LazyEval.core import LazyShowerData

def filter_event_by_particles(shower,
                                pdg):

    idx = shower.pdg == pdg
    
    filtered = showerdata.Showers(points=shower.points[idx],
                                  energies=shower.energies[idx],
                                  directions=shower.directions[idx],
                                  pdg=shower.pdg[idx])

    return filtered


def lazy_filter_event_by_particle(lazydata,pdg):

    assert isinstance(lazydata,LazyShowerData), "For lazy processing Lazy Shower Data format is necessary"

    pdg_values = lazydata.pdg[:]
    idx = pdg_values == pdg

    return np.where(idx)[0]