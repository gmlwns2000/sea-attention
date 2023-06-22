from .bert_glue_trainer import Trainer as BaseTrainer
from ..models import perlin_bert as perlin
from .bert_glue_trainer import task_to_batch_size

PERLIN_LAYERWISE = False
PERLIN_MODE = 'perlin'#'performer'

task_to_batch_size['mnli'] = 4 if not PERLIN_LAYERWISE else 32 #16 32

class Trainer(BaseTrainer):
    def __init__(
        self, subset = 'mnli'
    ):
        super().__init__(
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            amp_enabled=False,
            trainer_name='perlin_trainer',
            using_kd=(not PERLIN_LAYERWISE) and (PERLIN_MODE != 'performer_'),
            using_loss=not PERLIN_LAYERWISE,
            eval_steps=2000,
            lr = 1e-4,
            epochs = 20
        )
        
        for module in self.model.modules():
            if isinstance(module, perlin.BertSelfAttention):
                module.perlin_mode = PERLIN_MODE
        
        if PERLIN_LAYERWISE:
            for module in self.model.modules():
                if isinstance(module, perlin.BertSelfAttention):
                    module.perlin_layerwise = True
            
            for name, param in self.model.named_parameters():
                if 'perlin' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

if __name__ == '__main__':
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()