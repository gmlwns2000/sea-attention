method = ['performer',]#  'performer', 'reformer', 'sinkhorn','synthesizer','scatterbrain','longformer','bigbird', 'performer', 'reformer', 'sinkhorn','synthesizer','scatterbrain','longformer','bigbird'

ks = ['7'] # ,'13','25'

predictor_length = ['128',] # '64','128'

nbf = ['1','2','4','8']# '2','4','8'

subset = ['mnli', 'cola', 'mrpc']
for s in subset:
    for m in method:
        print("### "+m)
        for k in ks:
            for pl in predictor_length:
                if m!='perlin' and pl!='128':continue
                for nf in nbf:
                    print('cd /d1/jinakim/permutation-learning/')
                    print("tmux new -s "+s+"_"+m+"k"+k+"w"+pl+"_nbf"+nf)
                    print('conda activate torch4')
                    print('CUDA_VISIBLE_DEVICES=0 python -m src.trainer.perlin_trainer --method '+m+" --dataset glue --subset "+s+" --k "+k+" --k-flatten-dim batch --predictor-length "+pl+" --performer-nb-feature-factor "+nf+" --gradient-accumulation-steps 4 --epochs 20")
            print('\n')
        print('\n')
    print('\n')
    