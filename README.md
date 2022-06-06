# hf-trim
A tool to reduce the size of Hugging Face models via vocabulary trimming.

The library currently supports the following models (and their pretrained versions available on the [Hugging Face Models hub](https://huggingface.co/models));

1. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation](https://huggingface.co/docs/transformers/main/en/model_doc/bart)
2. [mBART: Multilingual Denoising Pre-training for Neural Machine Translation](https://huggingface.co/docs/transformers/main/en/model_doc/mbart)
3. [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://huggingface.co/docs/transformers/model_doc/t5)
4. [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://huggingface.co/docs/transformers/model_doc/mt5)


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
from hftrim.TokenizerTrimmer import TokenizerTrimmer
from hftrim.ModelTrimmer import ModelTrimmer

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


## Roadmap
- Add support for MarianMT models.
- Add support for FSMT models.


## Issues
Feel free to open an issue if you run into bugs, have any queries or want to request support for an architecture.


## Contributing
Contributions are welcome, especially those adding functionality for currently unsupported models.