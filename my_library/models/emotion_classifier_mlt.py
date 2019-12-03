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


@Model.register("emotion_classifier_mlt")
class SarcasmClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 quote_response_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 classifier_feedforward_2: FeedForward,
                 bert_model_name: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 report_auxiliary_metrics: bool = False
                 ) -> None:

        super(SarcasmClassifier, self).__init__(vocab, regularizer)

        self.quote_response_encoder = quote_response_encoder
        self.classifier_feedforward = classifier_feedforward
        self.text_field_embedder = BertModel.from_pretrained(bert_model_name)

        self.num_classes = self.vocab.get_vocab_size("labels")
        self.num_classes_emotions = self.vocab.get_vocab_size("emotion_labels")
        self.classifier_feedforward_2 = classifier_feedforward_2


        self.label_acc_metrics = {
            "accuracy": CategoricalAccuracy()
        }
        self.label_f1_metrics = {}
        self.label_f1_metrics_emotions = {}
        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="label")] =\
                F1Measure(positive_label=i)

        for i in range(self.num_classes_emotions):
            self.label_f1_metrics_emotions[vocab.get_token_from_index(index=i, namespace="labels")] =\
                F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()
        self.report_auxiliary_metrics = report_auxiliary_metrics
        # self.attention_seq2seq = Attention(quote_response_encoder.get_output_dim())
        # self.predict_mode = predict_mode
        initializer(self)

    @overrides
    def forward(self,
                quote_response: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                emotion_labels: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

        quote_response_mask = util.get_text_field_mask(quote_response)
        reps, _ = self.text_field_embedder(quote_response['bert'])  # ,
        # attention_mask=(qr != pad_tok_id).float()
        # )  # [bs, seq_len, 768]
        # print("after bert: {}\n{}".format(reps.size(),_.size()))
        cls_rep = reps[:, 0]  # [bs, 768]
        # print("cls: {}".format(cls_rep.size()))
        # shape: [batch, sent, output_dim]
        encoded_quote_response = self.quote_response_encoder(reps, quote_response_mask)

        if label is not None:
            # pylint: disable=arguments-differ
            logits = self.classifier_feedforward(encoded_quote_response)

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

        if emotion_labels is not None:  # this is the first scaffold task
            logits = self.classifier_feedforward_2(encoded_quote_response)
            # print("quote: {} - logits: {}\n".format(encoded_quote_response.size(), logits.size()))

            # class_probs = F.softmax(logits, dim=1)
            output_dict = {"logits": logits}
            loss = self.loss(logits, emotion_labels)
            output_dict["loss"] = loss
            for i in range(self.num_classes_emotions):
                metric = self.label_f1_metrics_emotions[self.vocab.get_token_from_index(index=i, namespace="emotion_labels")]
                metric(logits, emotion_labels)


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
            # metric_dict[name + '_P'] = metric_val[0]
            # metric_dict[name + '_R'] = metric_val[1]
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

        if self.report_auxiliary_metrics:
            sum_f1 = 0.0
            for name, metric in self.label_f1_metrics_emotions.items():
                metric_val = metric.get_metric(reset)
                metric_dict['aux-emo_' + name + '_P'] = metric_val[0]
                metric_dict['aux-emo_' + name + '_R'] = metric_val[1]
                metric_dict['aux-emo_' + name + '_F1'] = metric_val[2]
                if name != 'none':  # do not consider `none` label in averaging F1
                    sum_f1 += metric_val[2]
            names = list(self.label_f1_metrics_emotions.keys())
            total_len = len(names) if 'none' not in names else len(names) - 1
            average_f1 = sum_f1 / total_len
            # metric_dict['combined_metric'] = (accuracy + average_f1) / 2
            metric_dict['aux-emo_' + 'average_F1'] = average_f1

        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SarcasmClassifier':
        bert_model_name = params.pop("bert_model_name")
        quote_response_encoder = Seq2VecEncoder.from_params(params.pop("quote_response_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))
        classifier_feedforward_2 = FeedForward.from_params(params.pop("classifier_feedforward_2"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        report_auxiliary_metrics = params.pop_bool("report_auxiliary_metrics", False)

        # predict_mode = params.pop_bool("predict_mode", False)
        # print(f"pred mode: {predict_mode}")

        return cls(vocab=vocab,
                   bert_model_name=bert_model_name,
                   quote_response_encoder=quote_response_encoder,
                   classifier_feedforward=classifier_feedforward,
                   classifier_feedforward_2=classifier_feedforward_2,
                   initializer=initializer,
                   regularizer=regularizer,
                   report_auxiliary_metrics=report_auxiliary_metrics)


