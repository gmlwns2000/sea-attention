from .benchmark_bert import *

def main():
    nbfs = [1, 2, 4, 8]
    ks = [7, 13, 25]
    ws = [32, 64, 128]
    
    data = {}
    
    for nbf in nbfs:
        for k in ks:
            for w in ws:
                latency, mem = exam_config(BenchConfig(
                    method='perlin',
                    bsize=32,
                    seq_len=256,
                    k=k,
                    w=w,
                    nbf=nbf,
                    trace=False,
                ))
                data[f'nbf:{nbf},k:{k},w:{w}'] = {
                    'latency': latency * 1000, 
                    'mem': mem / (1024 ** 2),
                }
                
    for baseline in BASELINES:
        latency, mem = exam_config(BenchConfig(
            method=baseline,
            bsize=32,
            seq_len=256,
            k=7,
            w=128,
            nbf=1,
            trace=False,
        ))
        data[f'{baseline}'] = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
    
    path = './plots/main/benchmark_bert_ablation'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'data.json'), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()