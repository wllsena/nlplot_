# -*- coding: utf-8 -*-
__author__ = "William Sena <@wllsena>"
"""
Style Guide: PEP 8. Column limit: 100.
Author: William Sena <@wllsena>.
"""

import pandas as pd
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoModelWithLMHead,
                          AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

if __name__ == "__main__":
    train_df = pd.read_csv('./dataset/final/train.csv', usecols=['source', 'label'])
    train_df['source'] = 'translate English to vega-zero: ' + train_df['source'] + ' </s>'
    train_ds = Dataset.from_pandas(train_df)

    test_df = pd.read_csv('./dataset/final/test.csv', usecols=['source', 'label'])
    test_df['source'] = 'translate English to vega-zero: ' + test_df['source'] + ' </s>'
    test_ds = Dataset.from_pandas(test_df)

    dev_df = pd.read_csv('./dataset/final/dev.csv', usecols=['source', 'label'])
    dev_df['source'] = 'translate English to vega-zero: ' + dev_df['source'] + ' </s>'
    dev_ds = Dataset.from_pandas(dev_df)

    all_df = pd.concat((train_df, test_df, dev_df))
    all_ds = Dataset.from_pandas(all_df)

    #tokenizer = AutoTokenizer.from_pretrained('AhmedSSoliman/MarianCG-NL-to-Code')
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-wikiSQL')

    def preprocess(examples):
        model_inputs = tokenizer(examples['source'], max_length=512, padding=True, truncation=True)
        labels = tokenizer(examples['label'], max_length=512, padding=True, truncation=True)

        model_inputs['labels'] = labels['input_ids']

        return model_inputs

    train = train_ds.map(preprocess, batched=True)
    test = test_ds.map(preprocess, batched=True)
    dev = dev_ds.map(preprocess, batched=True)

    columns_to_return = ['input_ids', 'labels', 'attention_mask']
    train.set_format(type='torch', columns=columns_to_return)
    test.set_format(type='torch', columns=columns_to_return)
    dev.set_format(type='torch', columns=columns_to_return)

    #model = AutoModelForSeq2SeqLM.from_pretrained('AhmedSSoliman/MarianCG-NL-to-Code')
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./results_seq2seq',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=25,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model()
