from typing import Dict
import json
import logging
import csv

from overrides import overrides


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.tokenizers.token import Token
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from transformers import BertTokenizer
import numpy as np
import torch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("hur_reader")
class SarcasmDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 seq_len: int = 512,
                 bert_model_name: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        if bert_model_name:
            self._tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            # self.lowercase_input = "uncased" in bert_model_name
        else:
            self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.seq_len = seq_len
        self.labels = list()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as input:
            logger.info("Reading instances from lines in file at: %s", file_path)
            # for line_num, line in enumerate(Tqdm.tqdm(data_file.readlines())):
            data = json.load(input)

            for line in data:
                n_author = 0
                sarc = 0
                response = line['text']

                resp = line['resps']
                for author in resp:
                    if 'sarcasm' in resp[author]:
                        sarcastic = resp[author]['sarcasm']
                        n_author += 1
                        if sarcastic is True:
                            sarc += 1
                if sarc / n_author >= 0.4:
                    label = "1"
                else:
                    label = "0"
                if label not in self.labels:
                    self.labels.append(label)
                yield self.text_to_instance(response, label)

    @overrides
    def text_to_instance(self, response: str, label: str = None) -> Instance:

        # tokens = tokenizer.tokenize(text)[:args.seq_len - 2]
        response = self._tokenizer.tokenize(response)
        tokenized_response = []
        for w in response:
            tokenized_response.append(Token(w))
        print("QR : {}\ntype : {}".format(tokenized_response, len(tokenized_response)))

        token_ids = self._tokenizer.encode(response, add_special_tokens=True)
        print("token ids 0: {}\n".format(token_ids))

        token_ids_len = len(token_ids)
        print("token ids len: {}\n".format(token_ids_len))

        rspace = self.seq_len - token_ids_len  # <= args.seq_len
        token_ids = np.pad(token_ids, (0, rspace), 'constant',
                           constant_values=self._tokenizer.pad_token_id).tolist()
        print("cosa strana: {}\n".format(self._tokenizer.pad_token_id))
        print("token ids 1: {}\n".format(token_ids))
        response_field = torch.tensor(token_ids).long() # .long()
        print("QR field: {}\n".format(response_field))
        rf = TextField(tokenized_response, self._token_indexers)
        # len = torch.tensor(token_ids_len).long()
        # quote_response_field = TextField(tokenized_quote_response, self._token_indexers)
        # response_field = TextField(tokenized_response, self._token_indexers)

        metadata = {'quote_response': response_field, 'pad_token_id': self._tokenizer.pad_token_id } #, 'alllabels': alllabels_field}
        fields = {'quote_response': rf, 'metadata':MetadataField(metadata)}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)


"""
    @classmethod
    def from_params(cls, params: Params) -> 'SemanticScholarDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers)
"""


