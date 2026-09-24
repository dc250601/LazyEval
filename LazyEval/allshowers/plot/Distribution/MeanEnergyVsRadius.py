import numpy as np
import os
import matplotlib.pyplot as plt
import showerdata
import h5py as h5

from .config import PLOT_CONFIG
from LazyEval.core import util
def get_plot_data_per_shower(shower,
                             random_shift = True,
                             detector_config = showerdata.detector.get_ILD_geometry(),
                          # plot_config = PLOT_CONFIG
                            ):

    shower = showerdata.cluster_module.cluster(shower,
                                               random_shift,
                                               detector_config)
    
    energy_per_layer = showerdata.observables.calc_energy_per_radial_bin(shower,
                                                                         detector_config=detector_config)

    energy_per_layer = energy_per_layer * 1e3 # Scaling it to MeV

    return energy_per_layer


def get_plot_data(data,
                  random_shift = True,
                  detector_config = showerdata.detector.get_ILD_geometry(),
                  BATCH_SIZE = 256,
                  NThreads = 1,
                  ValidIndices = None
                 ):
    energy_per_layer = util.process_in_batch_and_return(data = data,
                                                        func = lambda x: get_plot_data_per_shower(shower = x,
                                                                                                  random_shift = random_shift,
                                                                                                  detector_config = detector_config
                                                                                                 ),
                                                        n_threads = NThreads,
                                                        batch_size = BATCH_SIZE,
                                                        valid_indices = ValidIndices
                                                       )
    
    # energy_per_layer = np.mean(energy_per_layer,axis=0)
    return energy_per_layer
    
                                                                
        
        

    

    





