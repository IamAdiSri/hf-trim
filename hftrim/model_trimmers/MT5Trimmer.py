import torch
from .BaseTrimmer import BaseTrimmer

class MT5Trimmer(BaseTrimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def trim_weights(self):
        # embedding matrix
        em = self.model.shared.weight.detach().numpy()
        self.trimmed_weights['shared'] = em[self.trimmed_vocab, :]

        # LM head matrix
        if 'lm_head' in self.model.state_dict():
            lmh = self.model.lm_head.weight.detach().numpy()
            self.trimmed_weights['lm_head'] = lmh[self.trimmed_vocab, :]

    def initialize_new_model(self):
        arch = self.config.architectures[0]
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