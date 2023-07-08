
from ..models.dict.model_task import basem_task_dict, model_task_dict
from ..models.dict.pretrained_model import pretrained_model_dict
from transformers import AutoConfig

def get_model_init(base_model_type, task_type, dataset, subset):
    model_task = model_task_dict(base_model_type, task_type)
    base_model_task = basem_task_dict(base_model_type, task_type)
    pretrained_model = pretrained_model_dict(base_model_type, dataset, subset)

    config = AutoConfig.from_pretrained(pretrained_model)
    model = model_task(config)
    base_model = base_model_task(config)

    return model, base_model