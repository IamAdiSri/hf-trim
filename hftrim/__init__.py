"""
hf-trim

Reduce the size of pretrained Hugging Face models via vocabulary trimming.
"""

__version__ = "2.3.1"
__author__ = "Aditya Srivastava"
__email__ = "adi.srivastava@hotmail.com"
__url__ = "https://github.com/IamAdiSri/hf-trim"

__supported_architectures__ = [
    'BartModel', 'BartForConditionalGeneration', 'BartForSequenceClassification', 'BartForQuestionAnswering', 'BartForCausalLM',
    'MBartModel', 'MBartForConditionalGeneration', 'MBartForSequenceClassification', 'MBartForQuestionAnswering', 'MBartForCausalLM',
    'T5Model', 'T5ForConditionalGeneration', 'T5EncoderModel',
    'MT5Model', 'MT5ForConditionalGeneration', 'MT5EncoderModel'
]