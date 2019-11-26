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
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.modules.attention.attention import Attention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import F1Measure

import torch.nn as nn


@Model.register("emo_att_classifier")
class SarcasmClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 quote_response_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 # predict_mode: bool = False,
                 ) -> None:

        super(SarcasmClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes_emotions = self.vocab.get_vocab_size("emotion_labels")
        self.quote_response_encoder = quote_response_encoder
        self.classifier_feedforward = classifier_feedforward
        self.attention_seq2seq = Attention(quote_response_encoder.get_output_dim())


        self.label_acc_metrics = {
            "accuracy": CategoricalAccuracy()
        }
        self.label_f1_metrics_emotions = {}
        # for i in range(self.num_classes):
        #     self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="label")] =\
        #         F1Measure(positive_label=i)


        for i in range(self.num_classes_emotions):
            self.label_f1_metrics_emotions[vocab.get_token_from_index(index=i, namespace="emotion_labels")] =\
                F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()

        # self.attention_seq2seq = Attention(quote_response_encoder.get_output_dim())

        # self.predict_mode = predict_mode

        initializer(self)

    @overrides
    def forward(self,
                quote_response: Dict[str, torch.LongTensor],
                emotion_labels: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # print("num emotions. {}".format(self.num_classes_emotions))
        # print("num classes. {}".format(self.num_classes))

        quote_response_mask = util.get_text_field_mask(quote_response)

        # shape: [batch, output_dim]



        if emotion_labels is not None:
            # pylint: disable=arguments-differ
            quote_response_embedding = self.text_field_embedder(quote_response)

            # shape: [batch, sent, output_dim]
            encoded_quote_response = self.quote_response_encoder(quote_response_embedding, quote_response_mask)
            attn_dist, encoded_quote_response = self.attention_seq2seq(encoded_quote_response,
                                                                       return_attn_distribution=True)

            logits = self.classifier_feedforward(encoded_quote_response)
            # print("quote: {} - logits: {}\n".format(encoded_quote_response.size(), logits.size()))

            class_probs = F.softmax(logits, dim=1)
            output_dict = {"logits": logits}
            loss = self.loss(logits, emotion_labels.squeeze(-1))
            output_dict["loss"] = loss
            for i in range(self.num_classes_emotions):
                metric = self.label_f1_metrics_emotions[self.vocab.get_token_from_index(index=i, namespace="emotion_labels")]
                metric(logits.squeeze(-1), emotion_labels.squeeze(-1))
            for metric_name, metric in self.label_acc_metrics.items():
                metric(logits.squeeze(-1), emotion_labels.squeeze(-1))
            output_dict['label'] = emotion_labels


#        if self.predict_mode:
#            logits = self.classifier_feedforward(encoded_quote_response)
#            class_probs = F.softmax(logits, dim=1)
        output_dict['quote_response'] = quote_response['tokens']
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        label = [self.vocab.get_token_from_index(x, namespace="emotion_labels")
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
        embedder_params1 = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(embedder_params1, vocab=vocab)
        quote_response_encoder = Seq2SeqEncoder.from_params(params.pop("quote_response_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))



        # predict_mode = params.pop_bool("predict_mode", False)
        # print(f"pred mode: {predict_mode}")

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   quote_response_encoder=quote_response_encoder,
                   classifier_feedforward=classifier_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)


def new_parameter(*size):
    out = Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class Attention(nn.Module):
    """ Simple multiplicative attention"""
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in, reduction_dim=-2, return_attn_distribution=False):
        """
        return_attn_distribution: if True it will also return the original attention distribution

        this reduces the one before last dimension in x_in to a weighted sum of the last dimension
        e.g., x_in.shape == [64, 30, 100] -> output.shape == [64, 100]
        Usage: You have a sentence of shape [batch, sent_len, embedding_dim] and you want to
            represent sentence to a single vector using attention [batch, embedding_dim]

        Here we use it to aggregate the lexicon-aware representation of the sentence
        In two steps we convert [batch, sent_len, num_words_in_category, num_categories] into [batch, num_categories]
        """
        # calculate attn weights
        attn_score = torch.matmul(x_in, self.attention).squeeze()
        # add one dimension at the end and get a distribution out of scores
        attn_distrib = F.softmax(attn_score.squeeze(), dim=-1).unsqueeze(-1)
        scored_x = x_in * attn_distrib
        weighted_sum = torch.sum(scored_x, dim=reduction_dim)
        if return_attn_distribution:
            return attn_distrib.reshape(x_in.shape[0], -1), weighted_sum
        else:
            return weighted_sum
