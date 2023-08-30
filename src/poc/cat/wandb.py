method = ['perlin', ]#'performer', 'reformer', 'sinkhorn','synthesizer','scatterbrain','longformer','bigbird'

ks = ['7',] # '13','25'

predictor_length = ['128'] # '64','256'

nbf = ['1','2','4','8']
for m in method:
    print("### "+m)
    for k in ks:
        for pl in predictor_length:
            for nf in nbf:
                print('cd /d1/jinakim/permutation-learning/')
                print("tmux new -s mp_"+m+"k"+k+"w"+pl)
                print('conda activate torch4')
                print('CUDA_VISIBLE_DEVICES=6 python -m src.trainer.perlin_trainer --method '+m+" --dataset glue --subset mnli --k "+k+" --k-flatten-dim batch --predictor-length "+pl+" --performer-nb-feature-factor "+nf+" --gradient-accumulation-steps 8")
        print('\n')
    print('\n')