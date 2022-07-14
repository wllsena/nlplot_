if __name__ == "__main__":
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
    from transformers import DataCollatorForSeq2Seq, AutoTokenizer

    train_df = pd.read_csv('./dataset/final/train.csv', usecols=['source', 'label'])
    train_ds = Dataset.from_pandas(train_df)

    test_df = pd.read_csv('./dataset/final/test.csv', usecols=['source', 'label'])
    test_ds = Dataset.from_pandas(test_df)

    dev_df = pd.read_csv('./dataset/final/dev.csv', usecols=['source', 'label'])
    dev_ds = Dataset.from_pandas(dev_df)

    all_df = pd.concat((train_df, test_df, dev_df))
    all_ds = Dataset.from_pandas(all_df)
    
    #from tokenizers import normalizers, pre_tokenizers, ByteLevelBPETokenizer

    #tokenizer = ByteLevelBPETokenizer()
    #tokenizer.normalizer = normalizers.Lowercase()
    #tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    #tokenizer.train_from_iterator(all_ds['source'] + all_ds['label'])

    #tokenizer.save_model('./tokenizer')

    tokenizer = AutoTokenizer.from_pretrained('AhmedSSoliman/MarianCG-NL-to-Code') # './tokenizer')  

    def preprocess(examples):
        model_inputs = tokenizer(examples['source'], max_length=256, padding=True, truncation=True)
        labels = tokenizer(examples['label'], max_length=256, padding=True, truncation=True)

        model_inputs['labels'] = labels['input_ids']

        return model_inputs

    train = train_ds.map(preprocess, batched=True)
    test = test_ds.map(preprocess, batched=True)
    dev = dev_ds.map(preprocess, batched=True)

    columns_to_return = ['input_ids', 'labels', 'attention_mask']
    train.set_format(type='torch', columns=columns_to_return)
    test.set_format(type='torch', columns=columns_to_return)
    dev.set_format(type='torch', columns=columns_to_return)
    
    model = AutoModelForSeq2SeqLM.from_pretrained('AhmedSSoliman/MarianCG-NL-to-Code')
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
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
