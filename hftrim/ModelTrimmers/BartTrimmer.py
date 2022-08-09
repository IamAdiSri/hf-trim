import torch
from .BaseTrimmer import BaseTrimmer

class BartTrimmer(BaseTrimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def trim_weights(self):
        # final logits bias
        if 'final_logits_bias' in self.model.state_dict():
            flb = self.model.final_logits_bias
            self.trimmed_weights['final_logits_bias'] = flb[:, self.trimmed_vocab_ids]

        # embedding matrix
        if 'model.shared.weight' in self.model.state_dict():
            em = self.model.model.shared.weight.detach().numpy()
            self.trimmed_weights['embeds'] = em[self.trimmed_vocab_ids, :]
        else:
            em = self.model.model.decoder.embed_tokens.weight.detach().numpy()
            self.trimmed_weights['embeds'] = em[self.trimmed_vocab_ids, :]

        # LM head matrix
        if 'lm_head.weight' in self.model.state_dict():
            lmh = self.model.lm_head.weight.detach().numpy()
            self.trimmed_weights['lm_head'] = lmh[self.trimmed_vocab_ids, :]


    def initialize_new_model(self):
        # arch = self.config.architectures[0]
        arch = self.model.__class__.__name__
        if arch=='BartModel':
            from transformers import BartModel
            model = BartModel(self.config)
            changed_params = [
                'shared.weight', 
                'encoder.embed_tokens.weight', 
                'decoder.embed_tokens.weight'
            ]
        elif arch=='BartForConditionalGeneration':
            from transformers import BartForConditionalGeneration
            model = BartForConditionalGeneration(self.config)
            changed_params = [
                'final_logits_bias',
                'model.shared.weight', 
                'model.encoder.embed_tokens.weight', 
                'model.decoder.embed_tokens.weight', 
                'lm_head.weight'
            ]
        elif arch=='BartForSequenceClassification':
            from transformers import BartForSequenceClassification
            model = BartForSequenceClassification(self.config)
            changed_params = [
                'model.shared.weight', 
                'model.encoder.embed_tokens.weight', 
                'model.decoder.embed_tokens.weight'
            ]
        elif arch=='BartForQuestionAnswering':
            from transformers import BartForQuestionAnswering
            model = BartForQuestionAnswering(self.config)
            changed_params = [
                'model.shared.weight', 
                'model.encoder.embed_tokens.weight', 
                'model.decoder.embed_tokens.weight'
            ]
        elif arch=='BartForCausalLM':
            from transformers import BartForCausalLM
            model = BartForCausalLM(self.config)
            changed_params = [
                'model.decoder.embed_tokens.weight', 
                'lm_head.weight'
            ]
        else:
            raise NotImplementedError("ERROR: BartTrimmer does not support this architecture!")

        self.trimmed_model = model
        self.changed_params = changed_params

    def trim_model(self):
        # copy unchanged params over from the old model
        for param in self.model.state_dict().keys():
            if param in self.changed_params:
                continue
            self.trimmed_model.state_dict()[param].copy_(self.model.state_dict()[param])
        
        # set trimmed params
        if 'final_logits_bias' in self.model.state_dict():
            self.trimmed_model.final_logits_bias = self.trimmed_weights['final_logits_bias']

        prunedEmbeddingMatrix = torch.nn.Embedding.from_pretrained(torch.Tensor(self.trimmed_weights['embeds']), 
                                                                   freeze=False, 
                                                                   padding_idx=self.tokenizer.pad_token_id)
        self.trimmed_model.set_input_embeddings(prunedEmbeddingMatrix)

        if 'lm_head' in self.trimmed_weights:
            prunedLMHeadMatrix = torch.Tensor(self.trimmed_weights['lm_head'])
            _ = self.trimmed_model.lm_head.weight.data.copy_(prunedLMHeadMatrix)

        # tie weights as described in model config
        self.trimmed_model.tie_weights()