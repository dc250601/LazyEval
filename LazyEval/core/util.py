from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np

def process_in_batch_and_return(data,
                                func,
                                n_threads,
                                batch_size,
                                valid_indices = None,
                               ):

    ## The data should be slicable
    if valid_indices is not None:
        data_length = len(valid_indices)
    else:
        data_length = len(data)
    
    assert data_length > n_threads, "More Threads than data_points"
    assert batch_size > 0, "Invalid batch size"
    if n_threads == 1:
        results = process_single_thread(data,
                                        0,
                                        data_length,
                                        valid_indices,
                                        func,
                                        batch_size)
    else:
        assert batch_size > n_threads, "More Threads than elements in batch"
        local_batch_size = batch_size//n_threads
        task_list = []
        
        executor = ThreadPoolExecutor(max_workers=n_threads)
        for t in range(n_threads):
            start = t*(data_length//n_threads)
            if t == n_threads - 1:
                end = data_length
            else:
                end = (t+1)*(data_length//n_threads)

            task_list.append(
                executor.submit(
                    process_single_thread,
                    data = data,
                    start_idx = start,
                    end_idx = end,
                    valid_indices = valid_indices,
                    func=func,
                    batch_size=local_batch_size
                )
            )
        
        results = np.concatenate([task.result() for task in task_list])
        executor.shutdown()
    return results
        
def process_single_thread(data,
                          start_idx,
                          end_idx,
                          valid_indices,
                          func,
                          batch_size):

    results = []
    if valid_indices is None:
        valid_indices = lambda x: x
        
    for local_start in range(start_idx,end_idx,batch_size):
        local_end = min(local_start+batch_size,end_idx)
        results.append(func(data[valid_indices[local_start:local_end]]))

    return np.concatenate(results,0)