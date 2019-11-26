from typing import Dict
import json
import logging
import csv

from overrides import overrides


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("emo_reader")
class EmotionDatasetReader(DatasetReader):
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
        with open(cached_path(file_path), "r") as tsvfile:
            logger.info("Reading instances from lines in file at: %s", file_path)
            # for line_num, line in enumerate(Tqdm.tqdm(data_file.readlines())):
            reader = csv.reader(tsvfile, delimiter='\t')

            for line in reader:
                response = line[0]
                response = response.rsplit('#', 1)[0]

                label = line[1]
                #print(line[1])
                if label not in self.labels:
                    self.labels.append(label)
                yield self.text_to_instance(response, label)
    @overrides
    def text_to_instance(self, response: str, label: str = None) -> Instance:
        tokenized_quote_response = self._tokenizer.tokenize(response)  # (quote + " " + response)
        # tokenized_response = self._tokenizer.tokenize(response)
        print(tokenized_quote_response)
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


