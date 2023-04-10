import numpy as np
import os, sys
import glob
import argparse

import torch

from transformers import T5Config, T5ForConditionalGeneration, EncoderDecoderConfig, EncoderDecoderModel
from transformers import TrainingArguments
from torchmetrics.functional.classification import accuracy

from KD_training.dataset import BertKdDataset
from KD_training.trainer import BertKDTrainer


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for fine-tuning')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default='results/')

    args = parser.parse_args()

    device = args.device

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Loading news articles dataset
    data_db = '/athena/rameaulab/store/mdo4009/KD_dataset/dump/de-en/DEEN.db'
    bert_dump = '/athena/rameaulab/store/mdo4009/KD_dataset/dump/de-en/targets/BERT-deen'
    data = '/athena/rameaulab/store/mdo4009/KD_dataset/dump/de-en/DEEN'

    vocab = torch.load(data + '.vocab.pt')
    src_vocab = vocab['src'].fields[0][1].vocab.stoi
    tgt_vocab = vocab['tgt'].fields[0][1].vocab.stoi
    train_dataset = BertKdDataset(data_db, bert_dump,
                                  src_vocab, tgt_vocab,
                                  max_len=150, k=8)
    print('Loaded Training dataset!')

    # Initializing the model
    distill_config = T5Config(vocab_size=32128, d_model=512, d_kv=64, d_ff=2048, num_layers=6, num_decoder_layers=6,
                            num_heads=8,
                            dropout_rate=0.3,   # according to paper
                            layer_norm_epsilon=1e-06,
                            initializer_factor=1.0, feed_forward_proj='relu', is_encoder_decoder=True, use_cache=True,
                            pad_token_id=0, eos_token_id=1, gradient_checkpointing=False)
    # model = EncoderDecoderModel(config=config_model)
    model = T5ForConditionalGeneration(config=distill_config)
    model = model.to(device)
    print('Model initialized')

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="steps", eval_steps=1000,
        max_steps=50000,
        warmup_steps=4000,
        learning_rate=1.0,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.98,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=6144,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy(preds=predictions, target=labels)

    trainer = BertKDTrainer(
        vocab,
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        # eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    # Training
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    trainer.train()
    trainer.save_model(output_dir=args.results_dir)


if __name__ == '__main__':
    main()
