import torch
import argparse
import collections
import os
import time
import random

from utils import AverageMeter, logging, set_seed
from corpus_to_tokenized import Vocab, load_sent, get_tokenized_batches
from vaes import LSTM_VAE, TRANSFORMER_VAE

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--train', metavar='FILE', required=True,
                    help='path to training file')
parser.add_argument('--valid', metavar='FILE', required=True,
                    help='path to validation file')
parser.add_argument('--save-dir', default='checkpoints', metavar='DIR',
                    help='directory to save checkpoints and outputs')
parser.add_argument('--load-model', default='', metavar='FILE',
                    help='path to load checkpoint if specified')
# VAE arguments
parser.add_argument('--vocab-size', type=int, default=10000, metavar='N',
                    help='keep N most frequent words in vocabulary')
parser.add_argument('--model_type', metavar='M',
                    choices=['lstm', 'transformer'],
                    help='which model to learn')
parser.add_argument('--dim_emb', type=int, default=512, metavar='D',
                    help='dimension of word embedding')
parser.add_argument('--dim_z', type=int, default=128, metavar='D',
                    help='dimension of latent variable z')
parser.add_argument('--dim_h', type=int, default=1024, metavar='D',
                    help='LSTM: dimension of hidden state per layer, \
                        Transformer: dimension of encoder/decoder inputs')
parser.add_argument('--nlayers', type=int, default=1, metavar='N',
                    help='LSTM: number of layers, \
                        Transformer: number of encoder/decoder layers in encoder/decoder stacks')
parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP',
                    help='dropout probability (0 = no dropout)')
parser.add_argument('--lambda_kl', type=float, default=0, metavar='R',
                    help='weight for kl term in VAE')
# Transformer arguments
parser.add_argument('--nhead', type=int, default=6, metavar='N',
                    help='number of heads in the multiheadattention models')
parser.add_argument('--dim_feedforward', type=int, default=2048, metavar='D',
                    help='dimension of the feedforward network model')
parser.add_argument('--max_len', type=int, default=512, metavar='L',
                    help='maximum length for sequences')
# Training arguments
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')
# Others
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging(str(args), log_file)

    # Prepare data
    train_sents = load_sent(args.train)
    logging('# train sents {}, tokens {}'.format(
        len(train_sents), sum(len(s) for s in train_sents)), log_file)
    valid_sents = load_sent(args.valid)
    logging('# valid sents {}, tokens {}'.format(
        len(valid_sents), sum(len(s) for s in valid_sents)), log_file)
    vocab_file = os.path.join(args.save_dir, 'vocab.txt')
    vocab = Vocab(vocab_file)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')


    """TO DEVICE"""


    model = {'lstm': LSTM_VAE, 'transformer': TRANSFORMER_VAE}[args.model_type](
        vocab, args).to(device)
    if args.load_model:
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model'])
        model.flatten()
    logging('# model parameters: {}'.format(
        sum(x.data.nelement() for x in model.parameters())), log_file)

    train_batches, _ = get_tokenized_batches(train_sents, args.max_len, vocab, args.batch_size, device)
    valid_batches, _ = get_tokenized_batches(valid_sents, args.max_len, vocab, args.batch_size, device)
    best_val_loss = None
    for epoch in range(args.epochs):
        start_time = time.time()
        logging('-' * 80, log_file)
        model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(train_batches)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            inputs, targets = train_batches[idx]
            losses = model.autoenc(inputs, targets, is_train=True)

"""" MODEL.AUTOENC,
LINE 194 IN VAES.PY, THEN LINE 195,196"""


if __name__ == '__main__':
    args = parser.parse_args()
    args.noise = [float(x) for x in args.noise.split(',')]
    main(args)
