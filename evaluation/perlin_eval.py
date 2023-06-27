class Visualize():
    def __init__(
        self, 
    ):
        ###
    
    def 
    

    



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='glue', type=str) # glue
    parser.add_argument('--subset', default='mnli', type=str) # mnli # TODO
    parser.add_argument('--eval-task', action='store_true', default="evaluate,attn_vis") # add what ever you want
    
    parser.add_argument('--mode', default='perlin', type=str) 
    parser.add_argument('--layerwise', action='store_true', default=False)
    parser.add_argument('--k-colwise', action='store_true', default=False)
    
    parser.add_argument('--checkpoint-path', action='store_true', default="") # TODO
    parser.add_argument('--save-name', action='store_true', default="") # TODO
    args = parser.parse_args()
    
    
    
    
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()