from ...models import perlin_bert
from ...models import hf_bert as berts

def model_task_dict(base_model_type, task_type):
    if base_model_type == 'bert':
        model = {
            "masked_lm" : perlin_bert.BertForMaskedLM,
            "next_sent_predict" : perlin_bert.BertForNextSentencePrediction,
            "seq_classification" : perlin_bert.BertForSequenceClassification,
            "multi_choice" : perlin_bert.BertForMultipleChoice,
            "token_classification" : perlin_bert.BertForTokenClassification,
            "question_answer" : perlin_bert.BertForQuestionAnswering
        }[task_type]
    # elif base_model_type == 'gpt':
    #     ...

    return model

def basem_task_dict(base_model_type, task_type):
    if base_model_type == 'bert':
        base_model = {
            "masked_lm" : berts.BertForMaskedLM,
            "next_sent_predict" : berts.BertForNextSentencePrediction,
            "seq_classification" : berts.BertForSequenceClassification,
            "multi_choice" : berts.BertForMultipleChoice,
            "token_classification" : berts.BertForTokenClassification,
            "question_answer" : berts.BertForQuestionAnswering
        }[task_type]
    # elif base_model_type == 'gpt':
    #     ...

    return base_model