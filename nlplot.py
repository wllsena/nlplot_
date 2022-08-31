# -*- coding: utf-8 -*-
__author__ = "William Sena <@wllsena>"
"""
Style Guide: PEP 8. Column limit: 100.
Author: William Sena <@wllsena>.
"""

from typing import List

import pandas as pd
import torch
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer)

dtypes = {'int64': 'int', 'object': 'str', 'float64': 'float'}


class Nlplot:
    def __init__(self, model_seq2seq: str, model_causal: str):
        self.tokenizer_seq2seq = AutoTokenizer.from_pretrained(model_seq2seq, local_files_only=True)
        self.model_seq2seq = AutoModelForSeq2SeqLM.from_pretrained(model_seq2seq,
                                                                   local_files_only=True)

        self.tokenizer_causal = AutoTokenizer.from_pretrained(model_causal, local_files_only=True)
        self.model_causal = AutoModelForCausalLM.from_pretrained(model_causal,
                                                                 local_files_only=True)

    def specify_dataset(self, df: pd.DataFrame) -> None:
        cols = [c.lower() + ' - ' + dtypes[str(t)] for c, t in zip(df.columns, df.dtypes)]
        self.prefix = str(df.shape[0]) + ' . ' + str(df.shape[1]) + ' . ' + ' , '.join(cols) + ' | '

    def seq2seq(self, question: str) -> str:
        source = self.prefix + question
        print(source)
        input_ids = self.tokenizer_seq2seq(source,
                                           return_tensors="pt",
                                           max_length=512,
                                           padding=True,
                                           truncation=True).input_ids
        outputs = self.model_seq2seq.generate(input_ids)

        decoded = ''.join(self.tokenizer_seq2seq.convert_ids_to_tokens(outputs[0])[1:-1]).replace(
            '▁', ' ').strip()

        return decoded

    def causal(self, question: str) -> List[str]:
        source = self.prefix + question
        print(source)
        input_ids = self.tokenizer_causal(source, return_tensors="pt").input_ids
        logits = self.model_causal(input_ids).logits[:, -1, :]

        pred_ids = torch.argsort(logits)[0, -5:]
        pred_words = [self.tokenizer_causal.decode(pred_id) for pred_id in pred_ids][::-1]

        return pred_words
