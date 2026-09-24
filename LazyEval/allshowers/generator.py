import torch
import warnings
import allshowers
from allshowers.preprocessing import compose

class AllShowerGenerator:
    def __init__(self,
                 flow,
                 device,
                 run_params,
                 num_timesteps=16,
                 resize_factor=1
                ):
        self.device = device
        self.resize_factor = resize_factor
        self.num_timesteps = num_timesteps
        self.run_params = run_params
        self.flow = flow
        self.cond_trafo = compose(run_params["data"].get("cond_trafo"))
        
        self.samples_coordinate_trafo = compose(run_params["data"].get("samples_coordinate_trafo"))
        self.samples_energy_trafo = compose(run_params["data"].get("samples_energy_trafo"))

        self.to(device)
    
    def get_weights(self,
                    weight_path,
                    trafo_path
                   ):
        
        trafo_state = torch.load(trafo_path,
                                 map_location="cpu",
                                 weights_only=True)
        
        self.samples_energy_trafo.load_state_dict(trafo_state["samples_energy_trafo"])
        self.samples_coordinate_trafo.load_state_dict(trafo_state["samples_coordinate_trafo"])
        self.cond_trafo.load_state_dict(trafo_state["cond_trafo"])

        model_state = torch.load(weight_path,
                        map_location="cpu",
                        weights_only=True)
        
        self.flow.load_state_dict(model_state)
        
        
    def to(self,device):
        self.cond_trafo = self.cond_trafo.to(device)
        self.samples_coordinate_trafo = self.samples_coordinate_trafo.to(device)
        self.samples_energy_trafo = self.samples_energy_trafo.to(device)
        self.flow = self.flow.to(device)
        self.device = device

        return self
        
    def __call__(self,batch):
        energies_,num_points_,angle_,labels_ = [e.to(self.device) for e in batch]
        
        if self.run_params["model"]["dim_inputs"][-1] > 1:
            condition = torch.concatenate([self.cond_trafo(energies_ * self.resize_factor), angle_], dim=-1)
        else:
            condition = cond_trafo(energies_)

        max_points = self.run_params["data"].get("max_num_points", 6016)
    
        layer = torch.zeros((condition.shape[0], max_points, 1), dtype=torch.int32)
        mask = torch.zeros((condition.shape[0], max_points, 1), dtype=torch.bool)


        for i in range(condition.shape[0]):
                total_points = torch.sum(num_points_[i])
                layer_i = torch.repeat_interleave(num_points_[i])
                if total_points > max_points:
                    warnings.warn(f"num points {total_points} exceeds max points {max_points}, truncating")
                    total_points = max_points
                    layer_i = layer_i[: self.max_points]
                layer[i, :total_points, 0] = layer_i
                mask[i, :total_points, 0] = True

        layer = layer.to(self.device)
        mask = mask.to(self.device)

        raw_samples = self.flow.sample(
            shape=(condition.shape[0], max_points, 3),
            num_timesteps=self.num_timesteps,
            cond=condition,
            num_points=num_points_,
            layer=layer,
            mask=mask,
            label=labels_,
        )        

        samples_ = torch.zeros((condition.shape[0], max_points, 4), device=self.device)
        samples_[:, :, :2] = self.samples_coordinate_trafo.inverse(raw_samples[:, :, :2])
        samples_[:, :, 2] = layer.squeeze(2)
        samples_[:, :, 3] = self.samples_energy_trafo.inverse(raw_samples[:, :, 2])
        samples_[~mask.repeat(1, 1, 4)] = 0
        
        return samples_



class InferObject:
    def inference_function(self,model, data):
        with torch.inference_mode():
            batch = [data["energies"],
                     data["num_points"],
                     data["angle"],
                     data["label"]
                    ]
            sample = model(batch)
        sample = sample.detach()
        
        return sample

            
        
        