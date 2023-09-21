method = ['perlin',]#  'performer', 'reformer', 'sinkhorn','synthesizer','scatterbrain','longformer','bigbird', 'performer', 'reformer', 'sinkhorn','synthesizer','scatterbrain','longformer','bigbird'

flatten = ['causal_batch',] #'head'

ks = ['7', '13','25'] # ,'13','25'

predictor_length = ['32', '64', '128'] # '32', '64', '128'

subset = ["mnli",'cola','mrpc']

nbf = ['1',]# '2','4','8'
for s in subset:
    for m in method:
        print("### "+s+"_"+m)
        for fttn in flatten:
            for k in ks:
                for pl in predictor_length:
                    if m!='perlin' and pl!='128':continue
                    for nf in nbf:
                        print('cd /d1/jinakim/permutation-learning/')
                        print("tmux new -s "+s+"_"+m+"k"+k+"w"+pl+"_nbf"+nf+"_fttn"+fttn)
                        print('conda activate torch4')
                        if s == 'mnli':
                            print('WANDB__SERVICE_WAIT=5000 CUDA_VISIBLE_DEVICES=0 python -m src.trainer.perlin_trainer --method '+m+" --dataset glue --subset "+s+" --k-flatten-dim "+fttn+" --k "+k+" --predictor-length "+pl+" --performer-nb-feature-factor "+nf+" --gradient-accumulation-steps 4 --epochs 20")
                        else:
                            print('WANDB__SERVICE_WAIT=5000 CUDA_VISIBLE_DEVICES=0 python -m src.trainer.perlin_trainer --method '+m+" --dataset glue --subset "+s+" --k-flatten-dim "+fttn+" --k "+k+" --predictor-length "+pl+" --performer-nb-feature-factor "+nf+" --gradient-accumulation-steps 4 --epochs 50")
                print('\n')
            print('\n')
        print('\n')
    print('\n')
        # for k in ks:
        #     for pl in predictor_length:
        #         if m!='perlin' and pl!='128':continue
        #         for nf in nbf:
        #             print('cd /d1/jinakim/permutation-learning/')
        #             print("tmux new -s "+m+"k"+k+"w"+pl+"_nbf"+nf+"_colwise")
        #             print('conda activate torch4')
        #             print('CUDA_VISIBLE_DEVICES=0 python -m src.trainer.perlin_trainer --method '+m+" --dataset glue --subset "+subset+" ----k-colwise True --k "+k+"--predictor-length "+pl+" --performer-nb-feature-factor "+nf+" --gradient-accumulation-steps 4 --epochs 20")
        #     print('\n')
        # print('\n')


# method = ['perlin',]#  'performer', 'reformer', 'sinkhorn','synthesizer','scatterbrain','longformer','bigbird', 'performer', 'reformer', 'sinkhorn','synthesizer','scatterbrain','longformer','bigbird'

# flatten = ['head',]

# ks = ['7', '13','25'] # ,'13','25'

# predictor_length = ['128',] # '32', '64', '128'

# subset = "mnli"

# nbf = ['1',]# '2','4','8'
# for m in method:
#     print("### "+m)
#     for fttn in flatten:
#         for k in ks:
#             for pl in predictor_length:
#                 if m!='perlin' and pl!='128':continue
#                 for nf in nbf:
#                     print('cd /d1/jinakim/permutation-learning/')
#                     print("tmux new -s "+m+"k"+k+"w"+pl+"_nbf"+nf+"_fttn"+fttn)
#                     print('conda activate torch4')
#                     print('CUDA_VISIBLE_DEVICES=0 python -m src.trainer.perlin_trainer --method '+m+" --dataset glue --subset "+subset+" --k-flatten-dim "+fttn+" --k "+k+" --predictor-length "+pl+" --performer-nb-feature-factor "+nf+" --gradient-accumulation-steps 4 --epochs 20")
#             print('\n')
#         print('\n')
#     for k in ks:
#         for pl in predictor_length:
#             if m!='perlin' and pl!='128':continue
#             for nf in nbf:
#                 print('cd /d1/jinakim/permutation-learning/')
#                 print("tmux new -s "+m+"k"+k+"w"+pl+"_nbf"+nf+"_colwise")
#                 print('conda activate torch4')
#                 print('CUDA_VISIBLE_DEVICES=0 python -m src.trainer.perlin_trainer --method '+m+" --dataset glue --subset "+subset+" ----k-colwise True --k "+k+"--predictor-length "+pl+" --performer-nb-feature-factor "+nf+" --gradient-accumulation-steps 4 --epochs 20")
#         print('\n')
#     print('\n')