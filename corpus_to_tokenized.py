# Turkish sentence corpus to inputs tokenized with BERTurk Tokenizer
# !pip install transformers
# truncation=True
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
# define go_x, x_eos
import torch

def batchify(train_path, val_path, N: int):
  train_corpus = open(train_path, encoding='utf-8').read.splitlines()
  n = 0
  batches = []
  while n < N:
    batch = []
    for s in train_corpus[n*N: n*N+N]:
      s = tokenizer.tokenize(s, max_length=max_seq_length, truncation=True)
      src = [go_x] + s
      tgt = s + [x_eos]
      # word2idx, pad
      batches.append((torch.LongTensor(src).t().contiguous().to(device), \
                      torch.LongTensor(tgt).t().contiguous().to(device)))
    n += 1
  return batches
