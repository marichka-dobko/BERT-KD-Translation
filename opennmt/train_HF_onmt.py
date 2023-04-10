import numpy as np
import os, sys
import glob
import argparse
import yaml

import torch
from torch.utils.data import DataLoader

from transformers import T5Config, T5ForConditionalGeneration, EncoderDecoderConfig, EncoderDecoderModel

from KD_training.dataset import BertKdDataset

from onmt.utils.optimizers import Optimizer
from onmt.train_single import build_model_saver, build_trainer, cycle_loader
from onmt.inputters.bert_kd_dataset import TokenBucketSampler
from onmt.inputters.inputter import build_dataset_iter


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

    config_path = 'config/config-transformer-base-mt-deen.yml'
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    args = argparse.Namespace(**config)
    args.train_from = None
    args.max_grad_norm = None
    args.save_model = 'output/kd-hf_onmt'
    args.kd_topk = 8
    args.train_steps = 100000
    args.kd_temperature = 10.0
    args.kd_alpha = 0.5
    args.warmup_steps = 8000
    args.learning_rate = 2.0
    args.bert_dump = bert_dump
    args.data_db = data_db
    args.bert_kd = True
    args.data = data
    args.copy_attn = False
    args.report_align = None

    exp_root = args.save_model
    os.makedirs(os.path.join(exp_root, 'log'))
    os.makedirs(os.path.join(exp_root, 'ckpt'))
    args.save_model = os.path.join(exp_root, 'ckpt', 'model')
    args.log_file = os.path.join(exp_root, 'log', 'log')
    args.tensorboard_log_dir = os.path.join(exp_root, 'log')

    # Build optimizer.
    optim = Optimizer.from_opt(model, args, checkpoint=None)

    # Build model saver
    model_saver = build_model_saver(args, args, model, vocab, optim)

    trainer = build_trainer(args, 0, model, vocab, optim, model_saver=model_saver)

    BUCKET_SIZE = 8192
    train_sampler = TokenBucketSampler(
        train_dataset.keys, BUCKET_SIZE, 6144,
        batch_multiple=1)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=4,
                              collate_fn=BertKdDataset.pad_collate)
    train_iter = cycle_loader(train_loader, device)

    valid_iter = build_dataset_iter("valid", vocab, args, is_train=False)

    train_steps = 50000
    exit()
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=1000,
        valid_iter=valid_iter,
        valid_steps=1000)

    if trainer.report_manager.tensorboard_writer:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == '__main__':
    main()
