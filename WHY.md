# hf-trim

## Why Trim Model Vocabulary?

Trimming the embeddings for large language models can be very helpful where a large part of their vocabulary may be unused and consuming unnecessary memory during training or inference. By eliminating these unused tokens, we can free valuable spacy which can then be utilized for larger batch sizes or to load models which would otherwise be too large to fit in the memory under given constraints. One of the best examples of this are multilingual models, which are often used for only a subset of the languages they have been pretrained on.

Pruning generally comes at a slight cost of performance. I have not extensively tested this myself, however a [drop in BLEU score of under 1 point has been reported](https://github.com/pytorch/fairseq/issues/2120#issuecomment-661509060). I plan to do some additional testing and will add those results here.