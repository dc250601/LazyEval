import showerdata
from .core import flow_matching as fm
from .core import transformer
from .core.data_sets import to_label_tensor
from .core.preprocessing import compose
from .core import data_sets

import yaml


from .resultwriter import Writer
from .generator import InferObject, AllShowerGenerator
from .dataset import ReaderObject

import torch

def generate_from_file(
    run_params_file,
    model_state_dict_file,
    trafo_state_dict_file,
    conditioning_data_file,
    outfile_path,
    StartIdx = 0,
    EndIdx = None,
    rank = None,
    BatchSize = 4 # Rank-wise,
    
    
    ):
    
    if EndIdx == None:
        EndIdx = showerdata.get_file_length(conditioning_data_file)

    SHOWER_SHAPE = (6016,4)

    with open(run_params_file) as f:
        run_params = yaml.load(f, Loader=yaml.FullLoader)


    model_params = run_params["model"].copy()
    flow_config = model_params.pop("flow_config")
    network = transformer.Transformer(**model_params)
    network = torch.compile(network)
    flow = fm.CNF(network, **flow_config)


    model = AllShowerGenerator(flow = flow,
                               device = "cpu",
                               run_params = run_params,)

    model.get_weights(weight_path=model_state_dict_file,
                      trafo_path=trafo_state_dict_file)

    if rank == 0:
        WriterObject = Writer(output_file = outfile_path,
                              length = (EndIdx - StartIdx),
                              shower_shape = SHOWER_SHAPE,
                              conditioning_file = conditioning_data_file
                             )
    else:
        WriterObject = None
    
    infer_object = InferObject()
    read_object = ReaderObject(BatchSize)
    ### writer.write_func



    return read_object.read_function, model, infer_object.inference_function,StartIdx, EndIdx, WriterObject, SHOWER_SHAPE
    

    

    