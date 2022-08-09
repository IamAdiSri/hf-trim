# hf-trim

[![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)](#) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-red)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](#)

[<img alt="PyPI" src="https://img.shields.io/pypi/v/hf-trim">](https://pypi.org/project/hf-trim) [<img alt="GitHub tag (latest by date)" src="https://img.shields.io/github/v/tag/IamAdiSri/hf-trim">](https://github.com/IamAdiSri/hf-trim/releases) [<img alt="PyPI - License" src="https://img.shields.io/pypi/l/hf-trim">](#)


**A package to reduce the size of 🤗 Hugging Face models via vocabulary trimming.**

The library currently supports the following models (and their pretrained versions available on the [Hugging Face Models hub](https://huggingface.co/models));

1. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation](https://huggingface.co/docs/transformers/main/en/model_doc/bart)
2. [mBART: Multilingual Denoising Pre-training for Neural Machine Translation](https://huggingface.co/docs/transformers/main/en/model_doc/mbart)
3. [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://huggingface.co/docs/transformers/model_doc/t5)
4. [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://huggingface.co/docs/transformers/model_doc/mt5)

### _"Why would I need to trim the vocabulary on a model?"_ 🤔

To put it simply, vocabulary trimming is a way to reduce a language model's memory footprint while retaining most of its performance.

Read more [here](WHY.md).


## Citation

If you use this software, please cite it as given below;
```
@software{Srivastava_hf-trim,
author = {Srivastava, Aditya},
license = {MPL-2.0},
title = {{hf-trim}}
url = {https://github.com/IamAdiSri/hf-trim}
}
```

## Installation

You can also run the following command to install from PyPI;
```bash
$ pip install hf-trim
```

You can install from source;
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
from hftrim.ModelTrimmers import MT5Trimmer

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
mt = MT5Trimmer(model, config, tt.trimmed_tokenizer)
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
Contributions are welcome, especially those adding functionality for new or currently unsupported models.
