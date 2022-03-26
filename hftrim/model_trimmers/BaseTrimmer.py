import uuid

class BaseTrimmer:
    def __init__(self, model, config, tokenizer):
        self.uid = uuid.uuid4().hex
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        self.trimmed_vocab = None
        self.trimmed_model = None
        self.trimmed_weights = {}

    def make_weights(self, vocab):
        self.set_vocab(vocab)
        self.set_config()
        self.trim_weights()
    
    def make_model(self):
        self.initialize_new_model()
        self.trim_model()
        return self.trimmed_model

    def set_vocab(self, vocab):
        self.trimmed_vocab = sorted(vocab)

    def set_config(self):
        assert self.trimmed_vocab != None, "ERROR: Trimmed vocabulary has not yet been set!"
        self.config.update(dict(vocab_size=len(self.trimmed_vocab)))

    def trim_weights(self):
        pass
    
    def initialize_new_model(self):
        pass

    def trim_model(self):
        pass