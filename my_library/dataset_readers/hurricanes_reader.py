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
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
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

        tokenized_quote_response = self._tokenizer.tokenize(response)  # (quote + " " + response)
        # tokenized_response = self._tokenizer.tokenize(response)
        quote_response_field = TextField(tokenized_quote_response, self._token_indexers)
        # response_field = TextField(tokenized_response, self._token_indexers)

        fields = {'quote_response': quote_response_field }#, 'alllabels': alllabels_field}
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


