from typing import Dict
import json
import logging
import csv

from overrides import overrides


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers.token import Token
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from transformers import BertTokenizer
import numpy as np
import torch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("hur_reader_emo_0")
class SarcasmDatasetReaderAux(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 seq_len: int = 256,
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
        self.emo_labels = list()


    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as input:
            logger.info("Reading instances from lines in file at: %s", file_path)
            # for line_num, line in enumerate(Tqdm.tqdm(data_file.readlines())):
            data = json.load(input)

            positive = ['acceptance', 'admiration', 'amazement',
                        'anticipation', 'distraction',
                        'ecstasy', 'interest', 'joy',
                        'serenity', 'surprise', 'sympathy',
                        'trust', 'vigilance']
            negative = ['anger', 'annoyance',
                        'apprehension', 'boredom', 'disgust',
                        'fear', 'grief', 'loathing', 'pensiveness',
                        'rage', 'sadness',
                        'terror']  # sarcasm?
            for line in data:
                pos = 0
                neg = 0
                response = line['text']
                resp = line['resps']
                for author in resp:
                    n_pos=0
                    n_neg=0
                    for k, v in resp[author].items():
                        if v is True:
                            if k in positive:
                                n_pos+=1
                            else:
                                n_neg+=1
                    if n_pos > n_neg:
                        pos +=1
                    else:
                        neg +=1

                if pos >= neg:
                    emo_label = "pos"
                else:
                    emo_label = "neg"
                if emo_label not in self.emo_labels:
                    self.emo_labels.append(emo_label)
                yield self.text_to_instance(response, emo_label)


    @overrides
    def text_to_instance(self, response: str, emo_label: str = None) -> Instance:

        response = self._tokenizer.tokenize(response)
        tokenized_response = []
        for w in response:
            tokenized_response.append(Token(w))
        # print("QR : {}\ntype : {}".format(tokenized_response, len(tokenized_response)))

        token_ids = self._tokenizer.encode(response, add_special_tokens=True)
        # print("token ids 0: {}\n".format(token_ids))

        token_ids_len = len(token_ids)
        # print("token ids len: {}\n".format(token_ids_len))

        rspace = self.seq_len - token_ids_len  # <= args.seq_len
        token_ids = np.pad(token_ids, (0, rspace), 'constant',
                           constant_values=self._tokenizer.pad_token_id).tolist()
        # print("cosa strana: {}\n".format(self._tokenizer.pad_token_id))
        # print("token ids 1: {}\n".format(token_ids))
        response_field = torch.tensor(token_ids).long() # .long()
        # print("QR field: {}\n".format(response_field))
        rf = TextField(tokenized_response, self._token_indexers)


        fields = {'quote_response': rf }#, 'alllabels': alllabels_field}
        if emo_label is not None:
            fields['label'] = LabelField(emo_label, label_namespace="labels")
        return Instance(fields)


"""
    @classmethod
    def from_params(cls, params: Params) -> 'SemanticScholarDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers)
"""


