import torch
from .T5Trimmer import T5Trimmer

class MT5Trimmer(T5Trimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def initialize_new_model(self):
        # arch = self.config.architectures[0]
        arch = self.model.__class__.__name__
        if arch=='MT5Model':
            from transformers import MT5Model
            model = MT5Model(self.config)
        elif arch=='MT5ForConditionalGeneration':
            from transformers import MT5ForConditionalGeneration
            model = MT5ForConditionalGeneration(self.config)
        elif arch=='MT5EncoderModel':
            from transformers import MT5EncoderModel
            model = MT5EncoderModel(self.config)
        else:
            raise NotImplementedError("ERROR: MT5Trimmer does not support this architecture!")

        self.trimmed_model = model
