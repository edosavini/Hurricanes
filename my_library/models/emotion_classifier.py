import operator
from copy import deepcopy
from distutils.version import StrictVersion
from typing import Dict, Optional

import allennlp
import numpy as np
import torch
import torch.nn.functional as F
from allennlp.common import Params

from torch.nn import Parameter, Linear

import logging

from typing import Dict, Optional

from overrides import overrides
import torch.tensor
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.modules.attention.attention import Attention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import F1Measure
from transformers import BertModel


import torch.nn as nn


@Model.register("emotion_classifier")
class SarcasmClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 bert_model_name: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 # predict_mode: bool = False,
                 ) -> None:

        super(SarcasmClassifier, self).__init__(vocab, regularizer)


        self.text_field_embedder = BertModel.from_pretrained(bert_model_name)

        self.num_classes_emotions = self.vocab.get_vocab_size("labels")

        self.label_acc_metrics = {
            "accuracy": CategoricalAccuracy()
        }
        self.label_f1_metrics_emotions = {}
        # for i in range(self.num_classes):
        #     self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="label")] =\
        #         F1Measure(positive_label=i)


        for i in range(self.num_classes_emotions):
            self.label_f1_metrics_emotions[vocab.get_token_from_index(index=i, namespace="labels")] =\
                F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()
        if "large" in bert_model_name:
            self.linear = nn.Linear(1024, self.num_classes_emotions)
        else:
            self.linear = nn.Linear(768, self.num_classes_emotions)

        # self.attention_seq2seq = Attention(quote_response_encoder.get_output_dim())

        # self.predict_mode = predict_mode

        initializer(self)

    @overrides
    def forward(self,
                quote_response: Dict[str, torch.LongTensor],
                # metadata: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # print("num emotions. {}".format(self.num_classes_emotions))
        # print("num classes. {}".format(self.num_classes))
        # qr = metadata[0]['quote_response']
        # pad_tok_id = metadata[0]['pad_token_id']
        # print(quote_response['bert'])

        # print(qr)

        if label is not None:
            # pylint: disable=arguments-differ
            reps, _ = self.text_field_embedder(quote_response['bert']) #,
                # attention_mask=(qr != pad_tok_id).float()
            #)  # [bs, seq_len, 768]

            # print("after bert: {}\n{}".format(reps.size(),_.size()))
            cls_rep = reps[:, 0]  # [bs, 768]
            # print("cls: {}".format(cls_rep.size()))
            # shape: [batch, sent, output_dim]
            logits = self.linear(cls_rep)

            # print("logits: {}\n".format(logits.size()))
            # print("label: {}\n".format(label.size()))
            class_probs = F.softmax(logits, dim=1)
            output_dict = {"logits": logits}
            loss = self.loss(logits, label)
            output_dict["loss"] = loss
            for i in range(self.num_classes_emotions):
                metric = self.label_f1_metrics_emotions[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(logits, label)
            for metric_name, metric in self.label_acc_metrics.items():
                metric(logits, label)
            output_dict['label'] = label


#        if self.predict_mode:
#            logits = self.classifier_feedforward(encoded_quote_response)
#            class_probs = F.softmax(logits, dim=1)
        output_dict['quote_response'] = quote_response['bert']
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        label = [self.vocab.get_token_from_index(x, namespace="label")
                 for x in argmax_indices]
        output_dict['probabilities'] = class_probabilities
        output_dict['positive_label'] = label
        output_dict['prediction'] = label
        quote_response = []
        for batch_text in output_dict['quote_response']:
            quote_response.append([self.vocab.get_token_from_index(token_id.item()) for token_id in batch_text])
        output_dict['quote_response'] = quote_response
        output_dict['all_label'] = [self.vocab.get_index_to_token_vocabulary(namespace="labels")
                                     for _ in range(output_dict['logits'].shape[0])]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics_emotions.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + '_P'] = metric_val[0]
            metric_dict[name + '_R'] = metric_val[1]
            metric_dict[name + '_F1'] = metric_val[2]
            if name != 'none':  # do not consider `none` label in averaging F1
                sum_f1 += metric_val[2]
        for name, metric in self.label_acc_metrics.items():
            metric_dict[name] = metric.get_metric(reset)
        names = list(self.label_f1_metrics_emotions.keys())
        total_len = len(names) if 'none' not in names else len(names) - 1
        average_f1 = sum_f1 / total_len
        # metric_dict['combined_metric'] = (accuracy + average_f1) / 2
        metric_dict['average_F1'] = average_f1

        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SarcasmClassifier':
        bert_model_name = params.pop("bert_model_name")
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))



        # predict_mode = params.pop_bool("predict_mode", False)
        # print(f"pred mode: {predict_mode}")

        return cls(vocab=vocab,
                   bert_model_name=bert_model_name,
                   initializer=initializer,
                   regularizer=regularizer)


