import torch
from .BartTrimmer import BartTrimmer

class BartTrimmer(BartTrimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def initialize_new_model(self):
        arch = self.config.architectures[0]
        if arch=='MBartModel':
            from transformers import MBartModel
            model = MBartModel(self.config)
        elif arch=='MBartForConditionalGeneration':
            from transformers import MBartForConditionalGeneration
            model = MBartForConditionalGeneration(self.config)
        elif arch=='MBartForSequenceClassification':
            from transformers import MBartForSequenceClassification
            model = MBartForSequenceClassification(self.config)
        elif arch=='MBartForQuestionAnswering':
            from transformers import MBartForQuestionAnswering
            model = MBartForQuestionAnswering(self.config)
        elif arch=='MBartForCausalLM':
            from transformers import MBartForCausalLM
            model = MBartForCausalLM(self.config)
        else:
            raise NotImplementedError("ERROR: MT5Trimmer does not support this architecture!")

        self.trimmed_model = model
