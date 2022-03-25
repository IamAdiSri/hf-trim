# hf-trim
Reduce the size of pretrained Hugging Face models via vocabulary trimming.

example https://github.com/zack-ashen/keep-cli/
        https://github.com/glotzerlab/plato

## Installation

Run the following command to install from PyPI;
```
$ pip install hf-trim
```

You can also install from source;

```
$ git clone https://github.com/IamAdiSri/hf-trim
$ cd hf-trim
$ pip install .
```

## Usage
### Simple Example
```
from transformers import MT5Config, MT5Tokenizer, MT5ForConditionalGeneration
from hf-trim.TokenizerTrimmer import TokenizerTrimmer
from hf-trim.ModelTrimmer import ModelTrimmer

data = [
        " UN Chief Says There Is No Military Solution in Syria", 
        "Şeful ONU declară că nu există o soluţie militară în Siria"
]

config = MT5Config.from_pretrained("google/mt5-small")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(data)
tt.make_tokenizer()

mt = ModelTrimmer(model, config, tokenizer, tt.trimmed_vocab)
mt.make_weights()
mt.make_model()

# save the model and tokenizer or use them directly if you want
tt.tokenizer.save_pretrained('triMT5/)
mt.model.save_pretrained('triMT5/')
```