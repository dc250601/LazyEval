import argparse

from LazyEval.allshowers import generateshower

from LazyEval import core
from LazyEval import util

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

    #### Starting the AllShowerSimulation ...

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

    write_func = None
    if rank == 0:
        write_func = write_obj.write_func

    core.evaluator.Infer(DataPath = args.conditioning_data,
                         read_func = read_func,
                         model = model,
                         infer_func = infer_func,
                         GlobalStart = GlobalStart,
                         GlobalStop = GlobalStop,
                         write_func = write_func,
                         output_shape = (args.batch_size,*shape),
                         rank = rank,
                         local_rank = local_rank,
                         world_size = world_size,
                         device = device
                        )
    
    if rank == 0:
        write_obj.close()
        print("All Complete")