import os
import shutil
import glob
import uuid
import warnings

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from sentencepiece import sentencepiece_model_pb2 as spm
from tqdm import tqdm

class TokenizerTrimmer:
    def __init__(self, tokenizer):
        if 'Fast' in tokenizer.__class__.__name__:
            warnings.warn(f"WARNING: Fast tokenizers are not supported. You can ignore this warning if you are not using an instance of PreTrainedTokenizerFast class of tokenizers!", RuntimeWarning)
        
        self.uid = uuid.uuid4().hex
        self.tokenizer = tokenizer
        self.m = None
        self.trimmed_vocab = set()
        self.trimmed_vocab_ids = None
        self.trimmed_tokenizer = None

    def make_vocab(self, data, tokenized=False):
        if tokenized:
            for sample in data:
                self.update_vocab_by_indices(sample)
        else:
            self.update_vocab_with_texts(data)

        self.add_special_tokens_to_vocab()
        self.extract_trimmed_vocab_ids()
    
    def make_tokenizer(self, cleanup=True):
        _ = self.save_tokenizer()
        _ = self.load_spm()
        self.trim_spm()
        _ = self.save_spm()
        _ = self.trim_tokenizer()
        if cleanup:
            self.cleanup()
        return self.trimmed_tokenizer

    def update_vocab_with_texts(self, texts):
        if len(texts)>0:
            tokenized = self.tokenizer(texts, add_special_tokens=False)
            for sample in tokenized['input_ids']:
                self.update_vocab_by_indices(sample)

    def update_vocab_by_indices(self, indices):
        for idx in indices:
            self.trimmed_vocab.add(self.tokenizer.convert_ids_to_tokens(idx))

    def add_special_tokens_to_vocab(self):
        self.trimmed_vocab.update(self.tokenizer.all_special_tokens)
        self.trimmed_vocab.update(self.tokenizer.additional_special_tokens)
        
    def extract_trimmed_vocab_ids(self):
        self.trimmed_vocab_ids = sorted(self.tokenizer.convert_tokens_to_ids(self.trimmed_vocab))

    def save_tokenizer(self, tokenizer=None, save_path=None):
        save_path = os.path.join('/tmp', self.uid) if save_path == None else save_path
        assert not os.path.exists(save_path), f"ERROR: {save_path} already exists!"
        os.mkdir(save_path)

        tokenizer = self.tokenizer if tokenizer == None else tokenizer
        self.tokenizer.save_pretrained(save_path)
        return save_path

    def load_spm(self, load_path=None):
        load_path = os.path.join('/tmp', self.uid) if load_path == None else load_path
        files = glob.glob(f'{load_path}/*.model')
        assert len(files)>0, "ERROR: No sentencepiece model found in {load_path}!"
        assert len(files)==1, "ERROR: Found more than one sentencepiece model in {load_path}!"

        spm_fname = files[0]
        self.m = spm.ModelProto()
        self.m.ParseFromString(open(spm_fname, 'rb').read())
        return load_path
    
    def save_spm(self, save_path=None):
        save_path = os.path.join('/tmp', self.uid) if save_path == None else save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        files = glob.glob(f'{save_path}/*.model')
        
        spm_fname = files[0] if len(files)==1 else os.path.join(save_path, 'spiece.model')
        with open(spm_fname, 'wb') as f:
            f.write(self.m.SerializeToString())
        return save_path

    def trim_spm(self):
        assert self.m != None, "ERROR: No sentencepiece model has been loaded!"
        assert len(self.trimmed_vocab) > 0, "ERROR: No tokens found in the trimmed vocabulary!"

        l = len(self.m.pieces)
        for i in tqdm(range(l)):
            p = self.m.pieces[i]
            if p.HasField('type') or p.piece in self.trimmed_vocab:
                self.m.pieces.append(p)
        del self.m.pieces[:l]

    def trim_tokenizer(self, load_path=None):
        load_path = os.path.join('/tmp', self.uid) if load_path == None else load_path
        files = glob.glob(f'{load_path}/*.model')
        assert len(files)>0, "ERROR: No sentencepiece model found in {load_path}!"
        assert len(files)==1, "ERROR: Found more than one sentencepiece model in {load_path}!"

        self.trimmed_tokenizer = self.tokenizer.from_pretrained(load_path)

    def cleanup(self):
        path = os.path.join('/tmp', self.uid)
        assert os.path.exists(path), f"ERROR: Cannot cleanup as files are not in the default location /tmp/{self.uid}!"
        shutil.rmtree(path)
        
    def _sanity_check(self):
        # checks if encode-decode from old and new tokenizers is the same
        pass