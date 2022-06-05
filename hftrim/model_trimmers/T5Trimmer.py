import torch
from .BaseTrimmer import BaseTrimmer

class T5Trimmer(BaseTrimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def trim_weights(self):
        # embedding matrix
        em = self.model.shared.weight.detach().numpy()
        self.trimmed_weights['shared'] = em[self.trimmed_vocab_ids, :]

        # LM head matrix
        if 'lm_head.weight' in self.model.state_dict():
            lmh = self.model.lm_head.weight.detach().numpy()
            self.trimmed_weights['lm_head'] = lmh[self.trimmed_vocab_ids, :]

    def initialize_new_model(self):
        # arch = self.config.architectures[0]
        arch = self.model.__class__.__name__
        if arch=='T5Model':
            from transformers import T5Model
            model = T5Model(self.config)
        elif arch=='T5ForConditionalGeneration':
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration(self.config)
        elif arch=='T5EncoderModel':
            from transformers import T5EncoderModel
            model = T5EncoderModel(self.config)
        else:
            raise NotImplementedError("ERROR: T5Trimmer does not support this architecture!")

        self.trimmed_model = model

    def trim_model(self):
        # copy unchanged params over from the old model
        for param in self.model.state_dict().keys():
            if param in [
                'shared.weight', 
                'encoder.embed_tokens.weight', 
                'decoder.embed_tokens.weight', 
                'lm_head.weight'
            ]:
                continue
            self.trimmed_model.state_dict()[param].copy_(self.model.state_dict()[param])
        
        # set trimmed params
        prunedEmbeddingMatrix = torch.nn.Embedding.from_pretrained(torch.Tensor(self.trimmed_weights['shared']), 
                                                                   freeze=False, 
                                                                   padding_idx=self.tokenizer.pad_token_id)
        self.trimmed_model.set_input_embeddings(prunedEmbeddingMatrix)

        if 'lm_head' in self.trimmed_weights:
            prunedLMHeadMatrix = torch.Tensor(new_lmh)
            _ = self.trimmed_model.lm_head.weight.data.copy_(prunedLMHeadMatrix)

        # tie weights as set up in config
        self.trimmed_model.tie_weights()