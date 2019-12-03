from typing import Dict
import json
import logging
import csv

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import torch

from transformers import BertTokenizer
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("emo_reader")
class EmotionDatasetReader(DatasetReader):
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
        with open(cached_path(file_path), "r") as tsvfile:
            logger.info("Reading instances from lines in file at: %s", file_path)
            # for line_num, line in enumerate(Tqdm.tqdm(data_file.readlines())):
            reader = csv.reader(tsvfile, delimiter='\t')

            for line in reader:
                response = line[0]
                response = response.rsplit('#', 1)[0]

                label = line[1]
                # print(line[1])
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

        # print("QR : {}\n".format(tokenized_response))

        token_ids = self._tokenizer.encode(response) # add_special_tokens=True)
        # print("token ids 0: {}\n".format(token_ids))

        token_ids_len = len(token_ids)
        # print("token ids len: {}\n".format(token_ids_len))

        rspace = self.seq_len - token_ids_len  # <= args.seq_len
        token_ids = np.pad(token_ids, (0, rspace), 'constant',
                           constant_values=self._tokenizer.pad_token_id).tolist()
        # print("cosa strana: {}\n".format(self._tokenizer.pad_token_id))
        # print("token ids 1: {}\n".format(token_ids))
        response_field = torch.tensor(token_ids) # .long()
        # print("QR field: {}\n".format(response_field))

        # len = torch.tensor(token_ids_len).long()
        # quote_response_field = TextField(tokenized_quote_response, self._token_indexers)
        # response_field = TextField(tokenized_response, self._token_indexers)
        rf = TextField(tokenized_response, self._token_indexers)
        # print("QR field ok: {}\n".format(rf))
        # len = torch.tensor(token_ids_len).long()
        # quote_response_field = TextField(tokenized_quote_response, self._token_indexers)
        # response_field = TextField(tokenized_response, self._token_indexers)

        # , 'alllabels': alllabels_field}
        fields = {'quote_response': rf}
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


