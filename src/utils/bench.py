import torch
import gc
import time

def bench(name, fn, t_warmup, t_sample):
    sample_count = 0
    try:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()
        torch.cuda.synchronize()
        print(f'[{name}] warmup... ', end = '', flush=True)
        t = time.time()
        while True:
            with torch.no_grad():
                fn()
            if time.time() - t > t_warmup:
                break
        torch.cuda.synchronize()
        print('benchmarking', end = '', flush=True)
        elapsed = 0
        last_report = time.time()
        while True:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.no_grad():
                fn()
            end.record()
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(end) / 1000
            
            sample_count += 1
            if time.time() - t > t_sample:
                break
            if time.time() - last_report > 0.5:
                last_report = time.time()
                print('.', end='', flush=True)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() - start_mem
    except torch.cuda.OutOfMemoryError as ex: # type: ignore
        mem = 0
        elapsed = 0
    interval = elapsed/(sample_count + 1e-8)
    print(f' done. sampled {sample_count}its. {interval*1000:.2f}ms/it {mem / 1024 / 1024:.2f} MB', flush=True)
    return interval, mem
