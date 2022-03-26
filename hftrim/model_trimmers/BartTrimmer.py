import torch
from .BaseTrimmer import BaseTrimmer

class BartTrimmer(BaseTrimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def trim_weights(self):
        # final logits bias
        if 'final_logits_bias' in self.model.state_dict():
            flb = model.final_logits_bias
            self.trimmed_weights['final_logits_bias'] = flb[:, new_vocab]

        # embedding matrix
        if 'shared' in self.model.state_dict():
            em = self.model.shared.weight.detach().numpy()
            self.trimmed_weights['embeds'] = em[self.trimmed_vocab, :]
        else:
            em = self.model.decoder.embed_tokens.weight.detach().numpy()
            self.trimmed_weights['embeds'] = em[self.trimmed_vocab, :]

        # LM head matrix
        if 'lm_head' in self.model.state_dict():
            lmh = self.model.lm_head.weight.detach().numpy()
            self.trimmed_weights['lm_head'] = lmh[self.trimmed_vocab, :]


    def initialize_new_model(self):
        arch = self.config.architectures[0]
        if arch=='BartModel':
            from transformers import BartModel
            model = BartModel(self.config)
        elif arch=='BartForConditionalGeneration':
            from transformers import BartForConditionalGeneration
            model = BartForConditionalGeneration(self.config)
        elif arch=='BartForSequenceClassification':
            from transformers import BartForSequenceClassification
            model = BartForSequenceClassification(self.config)
        elif arch=='BartForQuestionAnswering':
            from transformers import BartForQuestionAnswering
            model = BartForQuestionAnswering(self.config)
        elif arch=='BartForCausalLM':
            from transformers import BartForCausalLM
            model = BartForCausalLM(self.config)
        else:
            raise NotImplementedError("ERROR: T5Trimmer does not support this architecture!")

        self.trimmed_model = model

    def trim_model(self):
        # copy unchanged params over from the old model
        for param in self.model.state_dict().keys():
            if param in [
                'final_logits_bias',
                'shared.weight', 
                'encoder.embed_tokens.weight', 
                'decoder.embed_tokens.weight', 
                'lm_head.weight'
            ]:
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
            prunedLMHeadMatrix = torch.Tensor(new_lmh)
            _ = self.trimmed_model.lm_head.weight.data.copy_(prunedLMHeadMatrix)

        # tie weights as described in model config
        self.trimmed_model.tie_weights()