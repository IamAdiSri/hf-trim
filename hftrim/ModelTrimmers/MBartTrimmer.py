from .BartTrimmer import BartTrimmer

class MBartTrimmer(BartTrimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def initialize_new_model(self):
        # arch = self.config.architectures[0]
        arch = self.model.__class__.__name__
        if arch=='MBartModel':
            from transformers import MBartModel
            model = MBartModel(self.config)
            changed_params = [
                'shared.weight', 
                'encoder.embed_tokens.weight', 
                'decoder.embed_tokens.weight'
            ]
        elif arch=='MBartForConditionalGeneration':
            from transformers import MBartForConditionalGeneration
            model = MBartForConditionalGeneration(self.config)
            changed_params = [
                'final_logits_bias',
                'model.shared.weight', 
                'model.encoder.embed_tokens.weight', 
                'model.decoder.embed_tokens.weight', 
                'lm_head.weight'
            ]
        elif arch=='MBartForSequenceClassification':
            from transformers import MBartForSequenceClassification
            model = MBartForSequenceClassification(self.config)
            changed_params = [
                'model.shared.weight', 
                'model.encoder.embed_tokens.weight', 
                'model.decoder.embed_tokens.weight'
            ]
        elif arch=='MBartForQuestionAnswering':
            from transformers import MBartForQuestionAnswering
            model = MBartForQuestionAnswering(self.config)
            changed_params = [
                'model.shared.weight', 
                'model.encoder.embed_tokens.weight', 
                'model.decoder.embed_tokens.weight'
            ]
        elif arch=='MBartForCausalLM':
            from transformers import MBartForCausalLM
            model = MBartForCausalLM(self.config)
            changed_params = [
                'model.decoder.embed_tokens.weight', 
                'lm_head.weight'
            ]
        else:
            raise NotImplementedError("ERROR: MBartTrimmer does not support this architecture!")

        self.trimmed_model = model
        self.changed_params = changed_params
