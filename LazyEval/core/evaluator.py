import os
import socket

import numpy as np
import torch
import torch.distributed as dist


###########################
# Communication mode.
# 0 Completed
# 1 Active
###########################
class EvaluationModel:
    def __init__(self,
                 model,
                 device,
                 infer_func,
                 rank,
                 world_size,
                 output_shape,
                 write_func,
                ):
        
        self.model = model # The pytorch model only !!!
        self.infer_func = infer_func
        self.rank = rank
        self.world_size = world_size
        self.output_shape = output_shape
        self.write_func = write_func
        self.device = device

        self.model = self.model.to(self.device)
        
        if self.rank == 0:
            self.running_ranks = list(range(1,self.world_size))
            
    def __call__(self,data):

        batch = self.infer_func(self.model, data)
        return batch
        

    def postprocess(self,data,start_index):

        if data.shape != self.output_shape:
            raise RuntimeError(
        f"[rank {self.rank}] Model output shape mismatch: "
        f"got {data.shape}, expected {self.output_shape}")
        
        
        if self.rank == 0:
            ### The output of the master_GPU
            self.write_func(data,start_index)

            ### The output of the other GPUs
            self.recieve_and_process()
        
        else:
            self.send_and_forget(start_index,data)
            
    def recieve_and_process(self):
        
        for src_rank in self.running_ranks.copy():
            # print(f"||At Recieve and process working with {src_rank}||",flush=True)
            recv_buffer_status = torch.empty(1,
                                             dtype = torch.long,
                                             device = self.device
                                            )
            dist.recv(recv_buffer_status,src=src_rank)

            if recv_buffer_status.item() == 0:
                # print(f"||At Recieve and process working with {src_rank}: Recieved status signal 0 Deleting rank||",flush=True)
                self.running_ranks.remove(src_rank)
            else:
                recv_buffer_data = torch.empty(self.output_shape,
                                               dtype = torch.float32,
                                               device = self.device)
                recv_buffer_idx = torch.empty(1,
                                              dtype=torch.long,
                                              device = self.device
                                             )
                
                dist.recv(recv_buffer_idx,src=src_rank)
                dist.recv(recv_buffer_data,src=src_rank)

                src_start_idx = recv_buffer_idx.item()
                src_data = recv_buffer_data.detach().cpu().numpy()
                
                self.write_func(src_data,src_start_idx)

        # print("At the end of for loop",flush=True)

    def send_and_forget(self,start_index,data):
        dist.send(torch.tensor([1],
                               dtype=torch.long,
                               device = self.device),
                  dst=0)
        data = data.contiguous()
        dist.send(torch.tensor([start_index],
                               dtype=torch.long,
                               device = self.device
                              ),
                  dst=0
                 )
            
        dist.send(tensor=data,
                  dst=0
                 )
        
    
class Dataset:
    def __init__(self,
                 datapath,
                 read_func,
                 start,
                 stop,
                 rank,
                 world_size
                ):
        
        self.datapath = datapath
        self.read_func = read_func
        self.start = start
        self.stop=stop

        self.NumSamples = self.stop - self.start

        self.WorkerStart = self.start + self.NumSamples * rank // world_size
        self.WorkerStop = self.start + self.NumSamples * (rank+1) // world_size

        self.batch_iter = self.read_func(self.datapath,self.WorkerStart,self.WorkerStop)
        
    def get_batch(self):
        return next(self.batch_iter)

    def __len__(self):
        return len(self.batch_iter)



def Infer(DataPath,
          read_func,
          model,
          infer_func,
          GlobalStart,
          GlobalStop,
          write_func,
          output_shape,
          rank,
          local_rank,
          world_size,
          device
         ):

    dataset = Dataset(datapath = DataPath,
                      read_func = read_func,
                      start = GlobalStart,
                      stop = GlobalStop,
                      rank = rank,
                      world_size = world_size
                     )
    print(f"[rank {rank}] {len(dataset)} to be processed ", flush=True,)
    
    evaluator = EvaluationModel(model=model,
                                device = device,
                                infer_func=infer_func,
                                rank=rank,
                                world_size=world_size,
                                output_shape=output_shape,
                                write_func=write_func
                               )

    dist.barrier()

    ###########################################################

    batch_idx = -1
    
    while True:
        try:
            batch = dataset.get_batch()

            # print(f"[rank {rank}] Batch{batch_idx + 1} recieved ", flush=True,)
            infer = evaluator(batch)
            # print(f"[rank {rank}] Batch{batch_idx + 1} Infered ", flush=True,)
            batch_idx += 1
            # print(f"[rank {rank} Infer Object Batch Size {len(infer)}",flush=True)
            current_idx = dataset.WorkerStart + len(infer)*batch_idx
            evaluator.postprocess(infer,current_idx)
            # print(f"[rank {rank}] Batch{batch_idx + 1} Postprocessed ", flush=True,)
        except StopIteration:
            # print(f"[rank {rank}] StopIterationTriggered "
            # f"host={socket.gethostname()} "
            # f"gpu={local_rank}",
            # flush=True,)
            
            if rank !=0:
                # print(f"[rank {rank}] Sending Batch {batch_idx} ", flush=True,)
                dist.send(torch.tensor([0],
                                       dtype=torch.long,
                                       device = device),
                          dst=0)
                # print(f"[rank {rank}] Sent Batch {batch_idx} !!!!!", flush=True,)
                dist.barrier()
                dist.destroy_process_group()
                break
            else:
                # print("Rank 0 completed, remaining ranks",evaluator.running_ranks, flush=True)
                while True:
                    if len(evaluator.running_ranks) == 0:
                        # print("Going to call dist.barrier",flush=True)
                        dist.barrier()
                        # print("Going for destruction",flush=True)
                        dist.destroy_process_group()
                        # print("After destruction",flush=True)
                        return 0
                    # print("RANK-0: WAITING for ",evaluator.running_ranks, flush=True)
                    evaluator.recieve_and_process()
                    # print("Back at Infer",flush=True)
                break
    ###############################################################
               
        

    
    