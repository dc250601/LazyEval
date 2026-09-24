### Abstraction wrapper to be placed on top of Showerdata

import showerdata
import h5py as h5


class LazyAttributes:
    def __init__(self,
                 parent,
                 attribute,
                 loader
                ):
        self.parent = parent
        self.attribute = attribute
        self.attribute_loader = loader
    def __getitem__(self,
                    idx):
        if isinstance(idx, int):
            start = self.parent.start + idx
            stop = start + 1
        elif isinstance(idx,slice):
            
            start = self.parent.start if idx.start is None else self.parent.start+idx.start
            stop = self.parent.stop if idx.stop is None else min(start+idx.stop,self.parent.stop)
            
        else :
            raise TypeError(f"Invalid index type: {type(idx)}")

        with h5.File(self.parent.path, "r") as file:
            s = self.attribute_loader(file,slice(start,stop))
        return s 


class LazyShowerDataLoader:
    def __init__(self,
                 path,
                 start,
                 stop,
                 max_points = -1
                ):
        
        self.path = path
        self.start = start
        self.stop = stop

        self.points = LazyAttributes(self,
                                    "points",
                                     lambda x,y:showerdata.core._get_shower_data(x,
                                                                                 "showers",
                                                                                 y, max_points)
                                    )
        
        self.energy = LazyAttributes(self,
                                     "energy",
                                     lambda x,y:showerdata.core._get_float_data(x,
                                                                                "energies",
                                                                                y)
                                    )


        self.directions = LazyAttributes(self,
                                     "energy",
                                     lambda x,y:showerdata.core._get_float_data(x,
                                                                                "directions",
                                                                                y)
                                    )
        self.pdg = LazyAttributes(self,
                                  "pdg",
                                  lambda x,y:showerdata.core._get_int_data(x,
                                                                           "pdg",
                                                                           y)
                                 )
        
        
        self.shower_ids = LazyAttributes(self,
                                  "shower_ids",
                                  lambda x,y:showerdata.core._get_int_data(x,
                                                                           "shower_ids",
                                                                           y)
                                 )
        
        self.num_points = LazyAttributes(self,
                                  "num_points",
                                  lambda x,y:showerdata.core._get_int_data(x,
                                                                           "num_points",
                                                                           y)
                                 )

    def __getitem__(self,idx):
        
        if isinstance(idx,int):
            s = showerdata.load(path=self.path,
                                start=self.start+idx,
                                stop=self.start+idx+1)
        elif isinstance(idx,slice):
            
            start = 0 if idx.start is None else idx.start
            stop = (self.stop - self.start) if idx.stop is None else idx.stop
            
            s = showerdata.load(path=self.path,
                                start=self.start + start,
                                stop = min(self.stop,self.start + stop)
                               )
        elif isinstance(idx,list):
            s = []
            for elem in idx:
                s.append(self.__getitem__(elem))
            s = showerdata.concatenate(s)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
            
        return s
        