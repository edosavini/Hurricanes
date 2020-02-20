
from typing import Dict
import json
import logging
import csv
import sys
import os

from overrides import overrides


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer



logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _read(dir_path, dest):

    nfiles = 0.0
    voc = {}
    voc["mean_validation_accuracy"] = 0.0
    voc["mean_validation_f1"] = 0.0
    voc["mean_validation_precision"] = 0.0
    voc["mean_validation_recall"] = 0.0
    voc["mean_validation_loss"] = 0.0
    voc["mean_test_accuracy"] = 0.0
    voc["mean_test_f1"] = 0.0
    voc["mean_test_precision"] = 0.0
    voc["mean_test_recall"] = 0.0
    voc["mean_test_loss"] = 0.0
    voc["mean_test_avg_f1"] = 0.0

    for file_path in os.listdir(dir_path):
        path = dir_path + "/" + file_path
        if os.path.isdir(path):
            with open(cached_path(path+"/metrics.json"), "r") as jsonfile:
                print(nfiles)
                nfiles +=1
                result = json.load(jsonfile)
                voc["mean_validation_accuracy"] += result["best_validation_accuracy"]
                voc["mean_validation_f1"] += result["best_validation_1_F1"]
                # voc["mean_validation_precision"] += OldMLT["best_validation_1_P"]
                # voc["mean_validation_recall"] += OldMLT["best_validation_1_R"]
                voc["mean_validation_loss"] += result["best_validation_loss"]
                voc["mean_test_accuracy"] += result["test_accuracy"]
                voc["mean_test_f1"] += result["test_1_F1"]
                print(path)
                print(result["test_1_F1"])
                voc["mean_test_avg_f1"] += result["test_average_F1"]
                # voc["mean_test_precision"] += OldMLT["test_1_R"]
                # voc["mean_test_recall"] += OldMLT["test_1_R"]
                voc["mean_test_loss"] += result["test_loss"]

    outfile = open(dest, 'wt')
    outfile.write("Mean metrics for {}:\n".format(os.path.basename(dir_path)))
    for key, value in voc.items():
        value /= nfiles
        outfile.write("{}: {}\n".format(key, value))






_read("ElmoBilstm", "mean_metrics.txt")

