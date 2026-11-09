import os
import h5py as h5
import torch
import showerdata
import fcntl
from pathlib import Path

class Writer:
    def __init__(self,
                 output_file,
                 length,
                 shower_shape,
                 conditioning_file,
                 GlobalStartIdx
                ):
                    
        self.conditioning_data_file = h5.File(conditioning_file,"r")
        self.output_file = output_file

        if os.path.exists(output_file) == False:
            
            showerdata.create_empty_file(self.output_file,
                                         shape=(length,
                                                *shower_shape)
                                        )

        path = Path(output_file).resolve()
        self.LOCK_PATH = path.with_name(path.name + ".lock")
        self.GlobalStartIdx = GlobalStartIdx
        
    def write_func(self,
                   data,
                   idx):
        if torch.is_tensor(data):
            data = data.to("cpu").numpy()

        showers = showerdata.Showers(points=data,
                                     energies=showerdata.core._get_float_data(self.conditioning_data_file,"energies",slice(idx,(idx+len(data)))),
                                     directions=showerdata.core._get_float_data(self.conditioning_data_file,"directions",slice(idx,(idx+len(data)))),
                                     pdg=showerdata.core._get_int_data(self.conditioning_data_file,"pdg",slice(idx,(idx+len(data))))
                                    )

        with self.LOCK_PATH.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            showerdata.save_batch(showers,
                                  self.output_file,
                                  start=idx-self.GlobalStartIdx)        

    def close(self):
        self.conditioning_data_file.close()
        