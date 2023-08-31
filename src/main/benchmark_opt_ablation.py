from .benchmark_bert import *
from ..models.perlin_attention import modules as perlin_modules

perlin_modules.CAUSAL_CONV_FORCE_NON_CAUSAL = True

def main():
    nbfs = [1, 2, 4, 8]
    ks = [32, 64, 128]
    ws = [128, 256, 384]
    n_hashs = [2, 4, 8, 16]
    
    data = {}
    
    latency, mem = exam_config(BenchConfig(
        method='none',
        bsize=1,
        seq_len=2048,
        k=64,
        w=128,
        nbf=1,
        trace=False,
        causal=True,
        n_hash=1,
    ))
    data[f'none'] = {
        'latency': latency * 1000, 
        'mem': mem / (1024 ** 2),
    }
    
    for nbf in nbfs:
        for k in ks:
            for w in ws:
                latency, mem = exam_config(BenchConfig(
                    method='perlin',
                    bsize=1,
                    seq_len=2048,
                    k=k,
                    w=w,
                    nbf=nbf,
                    trace=False,
                    causal=True,
                ))
                data[f'nbf:{nbf},k:{k},w:{w}'] = {
                    'latency': latency * 1000, 
                    'mem': mem / (1024 ** 2),
                }
    
    for nbf in nbfs:
        latency, mem = exam_config(BenchConfig(
            method='performer',
            bsize=1,
            seq_len=2048,
            k=64,
            w=128,
            nbf=nbf,
            trace=False,
            causal=True,
        ))
        data[f'performer=nbf:{nbf}'] = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
    
    for n_hash in n_hashs:
        latency, mem = exam_config(BenchConfig(
            method='reformer',
            bsize=1,
            seq_len=2048,
            k=64,
            w=128,
            nbf=nbf,
            trace=False,
            causal=True,
            n_hash=n_hash,
        ))
        data[f'reformer=n_hash:{n_hash}'] = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
    
    path = './plots/main/benchmark_opt_ablation'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'data.json'), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()