# hf-trim
Reduce the size of pretrained Hugging Face models via vocabulary trimming.

The library currently supports the following models;

1. [BART](https://huggingface.co/docs/transformers/main/en/model_doc/bart) and its derivatives (such as [mBART](https://huggingface.co/docs/transformers/main/en/model_doc/mbart) and models on the [Hugging Face Models hub](https://huggingface.co/models)).
2. [T5](https://huggingface.co/docs/transformers/model_doc/t5) and its derivatives (such as [MT5](https://huggingface.co/docs/transformers/model_doc/mt5) and models on the [Hugging Face Models hub](https://huggingface.co/models)).

## Installation

Run the following command to install from PyPI;
```bash
$ pip install hf-trim
```

You can also install from source;

```bash
$ git clone https://github.com/IamAdiSri/hf-trim
$ cd hf-trim
$ pip install .
```


## Usage
### Simple Example
```python
from transformers import MT5Config, MT5Tokenizer, MT5ForConditionalGeneration
from hf-trim.TokenizerTrimmer import TokenizerTrimmer
from hf-trim.ModelTrimmer import ModelTrimmer

data = [
        " UN Chief Says There Is No Military Solution in Syria", 
        "Şeful ONU declară că nu există o soluţie militară în Siria"
]

# load pretrained config, tokenizer and model
config = MT5Config.from_pretrained("google/mt5-small")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# trim tokenizer
tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(data)
tt.make_tokenizer()

# trim model
mt = ModelTrimmer(model, config, tt.trimmed_tokenizer)
mt.make_weights(tt.trimmed_vocab_ids)
mt.make_model()
```

You can directly use the trimmed model with `mt.trimmed_model` and the trimmed tokenizer with `tt.trimmed_tokenizer`.

### Saving and Loading
```python
# save with
tt.trimmed_tokenizer.save_pretrained('trimT5')
mt.trimmed_model.save_pretrained('trimT5')

# load with
config = MT5Config.from_pretrained("trimT5")
tokenizer = MT5Tokenizer.from_pretrained("trimT5")
model = MT5ForConditionalGeneration.from_pretrained("trimT5")
```
## Limitations
- Fast tokenizers are currently unsupported.
- Tensorflow and Flax models are currently unsupported.

## Upcoming Features
- Support for MarianMT models.
- Support for FSMT models.

## Issues

Feel free to open an issue if you run into bugs, have any queries or want to request support for an architecture.


## Contributing

Contributions are welcome, especially those adding functionality for currently unsupported models.