import showerdata
import allshowers
from allshowers import flow_matching as fm
from allshowers import transformer
from allshowers.data_sets import to_label_tensor
from allshowers.preprocessing import compose
from allshowers import data_sets

import yaml


from .resultwriter import Writer
from .generator import InferObject, AllShowerGenerator
from .dataset import ReaderObject
from ..util import setup_distributed
import argparse

import torch

import fcntl
from pathlib import Path

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
        dist.barrier()
    else:
        dist.barrier()
        WriterObject = Writer(output_file = outfile_path,
                              length = None,
                              shower_shape = SHOWER_SHAPE,
                              conditioning_file = conditioning_data_file
                             )
    
    infer_object = InferObject()
    read_object = ReaderObject(BatchSize)
    ### writer.write_func



    return read_object.read_function, model, infer_object.inference_function,StartIdx, EndIdx, WriterObject, SHOWER_SHAPE
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lazy Evaluator for Crazy Simulators !!!"
    )

    parser.add_argument(
        "--run-params",
        type=str,
        required=True,
        help="Path to the AllShowers run-parameters/config file."
    )

    parser.add_argument(
        "--model-weights",
        type=str,
        required=True,
        help="Path to the model state_dict file."
    )

    parser.add_argument(
        "--trafo-weights",
        type=str,
        required=True,
        help="Path to the transformation/preprocessor state_dict file."
    )

    parser.add_argument(
        "--conditioning-data",
        type=str,
        required=True,
        help="Path to the conditioning data file."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.h5",
        help="Path to the output file."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
        ,
        help="Batch size per rank."
    )

    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Global starting index."
    )

    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Global stopping index. Defaults to the full dataset."
    )

    args = parser.parse_args()

    rank, local_rank, world_size, device = util.setup_distributed()

    run_utils= generateshower.generate_from_file(run_params_file = args.run_params,
                                                 model_state_dict_file = args.model_weights,
                                                 trafo_state_dict_file = args.trafo_weights,
                                                 conditioning_data_file = args.conditioning_data,
                                                 outfile_path = args.output,
                                                 StartIdx = args.start_idx,
                                                 EndIdx = args.end_idx,
                                                 BatchSize = args.batch_size,
                                                 rank = rank)

    read_func,model,infer_func,GlobalStart,GlobalStop,write_obj,shape = run_utils

    

    ##########################################################################################################
    NumSamplesPerRank = (args.end_idx - args.start_idx)// world_size
    WorkerStart = args.start_idx + NumSamplesPerRank * rank 
    WorkerStop = args.start_idx + NumSamplesPerRank* (rank+1)

    if rank == world_size - 1:
        WorkerStop = args.end_idx

    BatchIter = read_func(datapath = args.conditioning_data_file,
                          start = WorkerStart,
                          end = WorkerStop
                         )
    
    ##########################################################################################################
    batch_idx = -1
    current_idx = WorkerStart
    while True:
        try:
            batch = next(BatchIter)
            infer_samples = infer_func(model, data)
            batch_idx += 1
            current_idx += len(infer)
            write_obj.write_func(infer_samples, current_idx)
        except StopIteration:
            print(f"[rank {rank}] Completed ", flush=True)

    dist.barrier()
    dist.destroy_process_group()