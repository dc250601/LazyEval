import torch
import allshowers
from allshowers.data_sets import to_label_tensor
import showerdata

class AllShowerDataset:
    
    def __init__(self,
                 data_path,
                 StartIdx,
                 EndIdx,
                 BatchSize
                ):

        self.data_path = data_path
        self.StartIdx = StartIdx
        self.EndIdx = EndIdx
        self.BatchSize = BatchSize
        
        self.index = self.StartIdx
        
        self.GetGeant4Showers = True
        self.GetObservables = True

        self.NumBatches = (self.EndIdx - self.StartIdx)//self.BatchSize
        
    def get_energies(self,
                     start_idx, 
                     end_idx):
        
        dat = showerdata.observables.read_observables_from_file(
            path=self.data_path,
            observables = ["incident_energies"],
            start= start_idx,
            stop = end_idx)
        
        energies = torch.from_numpy(dat["incident_energies"]).to(torch.float32)

        return energies
        
    def get_num_points(self,
                       start_idx, 
                       end_idx):
        
        dat = showerdata.observables.read_observables_from_file(
            path=self.data_path,
            observables = ["num_points_per_layer"],
            start= start_idx,
            stop = end_idx)
        
        num_pts = torch.from_numpy(dat["num_points_per_layer"])

        return num_pts
    
    def get_angle(self,
                  start_idx, 
                  end_idx):
        
        dat = showerdata.observables.read_observables_from_file(
            path=self.data_path,
            observables = ["incident_directions"],
            start= start_idx,
            stop = end_idx)
        
        directions = torch.from_numpy(dat["incident_directions"])

        return directions
    
    def get_pdg(self,
                start_idx, 
                end_idx):
        
        dat = showerdata.observables.read_observables_from_file(
            path=self.data_path,
            observables = ["incident_pdg"],
            start= start_idx,
            stop = end_idx)
        
        pdg = torch.from_numpy(dat["incident_pdg"])
        labels = to_label_tensor(pdg=pdg,
                                 label_list=[11,-11,22,130,
                                             211,-211,321,-321,
                                             2112, -2112, 2212, -2212])
        

        return pdg, labels

    def get_geant4_showers(self,
                           start_idx,
                           end_idx,
                          ):
        geant4Samples = showerdata.load(path = self.data_path,
                                        start = start_idx,
                                        stop = end_idx
                                       )
        return geant4Samples



    def __iter__(self):
        return self

    def __next__(self):
        if self.index + self.BatchSize> self.EndIdx:
            raise StopIteration

        batch = {}
        if self.GetObservables:
            batch["energies"] = self.get_energies(self.index,
                                                  self.index + self.BatchSize)
            batch["num_points"] = self.get_num_points(self.index,
                                                      self.index+self.BatchSize)
            batch["angle"] = self.get_angle(self.index,
                                            self.index+self.BatchSize)
            pdg,labels = self.get_pdg(self.index,
                                      self.index+self.BatchSize)
            batch["pdg"] = pdg
            batch["label"] = labels

        if self.GetGeant4Showers:
            batch["Geant4Showers"] = self.get_geant4_showers(self.index,
                                                             self.index+self.BatchSize)
        
        self.index += self.BatchSize
        return batch

    def __len__(self):
        return self.NumBatches
    

    

class ReaderObject:
    def __init__(self,batch_size):
        self.batch_size = batch_size
        
    def read_function(self,datapath,start,end):
        return AllShowerDataset(data_path = datapath,
                                StartIdx = start,
                                EndIdx = end,
                                BatchSize = self.batch_size
                               )
        