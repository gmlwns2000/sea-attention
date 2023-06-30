# this file contains every visualization functions

# 2. 

class Visualization():
    def __init__(self) -> None:
        pass
    
    def plot_attention()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='perlin', type=str) # in ["base", "perlin", ""]
    parser.add_argument('--layerwise', action='store_true', default=False) # 
    parser.add_argument('--k-colwise', action='store_true', default=False) # columnwise relationwise
    args = parser.parse_args()
    
    PERLIN_MODE = args.mode
    PERLIN_K_FLATTEN = not args.k_colwise
    PERLIN_LAYERWISE = args.layerwise
    
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()