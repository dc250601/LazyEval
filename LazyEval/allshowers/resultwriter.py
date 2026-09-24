import os
import h5py as h5
import torch
import showerdata

class Writer:
    def __init__(self,
                 output_file,
                 length,
                 shower_shape,
                 conditioning_file
                ):
        
        if os.path.exists(output_file):
            os.remove(output_file)
            
        # self.write_file = h5.File(output_file,"w")
        # self.allshowers_group = self.write_file.create_group("allshowers")
        # self.AllShowerGeneratedDataset = self.allshowers_group.create_dataset(shape=(length,
        #                                                                              *shower_shape),
        #                                                                       name = "showers")
        self.conditioning_data_file = h5.File(conditioning_file,"r")
        self.output_file = output_file
        showerdata.create_empty_file(self.output_file,
                                     shape=(length,
                                            *shower_shape)
                                    )

    def write_func(self,data,idx):
        if torch.is_tensor(data):
            data = data.to("cpu").numpy()
        # This is inefficient and will definitely crash for large evaluation. Instead use this to make plots
        # print(data.shape)
        showers = showerdata.Showers(points=data,
                                     energies=showerdata.core._get_float_data(self.conditioning_data_file,"energies",slice(idx,(idx+len(data)))),
                                     directions=showerdata.core._get_float_data(self.conditioning_data_file,"directions",slice(idx,(idx+len(data)))),
                                     pdg=showerdata.core._get_int_data(self.conditioning_data_file,"pdg",slice(idx,(idx+len(data))))
                                    )


        # print("Going to write in ",self.output_file)
        # print("Print at idx",idx)
        # print("Shower",showers)

        
        showerdata.save_batch(showers,
                              self.output_file,
                              start=idx)        

    def close(self):
        self.conditioning_data_file.close()
        # self.write_file.close()
        