from .bert_glue_trainer import Trainer as BaseTrainer
from ..models import perlin_bert as perlin
from .bert_glue_trainer import task_to_batch_size

task_to_batch_size['mnli'] = 8

class Trainer(BaseTrainer):
    def __init__(
        self, subset = 'mnli'
    ):
        super().__init__(
            subset=subset,
            model_cls=perlin.BertForSequenceClassification,
            amp_enabled=False,
            trainer_name='perlin_trainer',
            using_kd=True,
            eval_steps=3000,
        )
        
        # for name, param in self.model.named_parameters():
        #     if 'perlin' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

if __name__ == '__main__':
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()