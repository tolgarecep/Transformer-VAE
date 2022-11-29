# Turkish sentence corpus to inputs tokenized with BERTurk Tokenizer
# !pip install transformers
# truncation=True
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
import torch

def batchify(path, vocab, N: int):
  train_corpus = open(path, encoding='utf-8').read.splitlines()
  n = 0
  batches = []
  while n < N:
    batch = []
    for s in train_corpus[n*N: n*N+N]:
      s = tokenizer.tokenize(s, max_length=max_seq_length, truncation=True)
      src = [vocab.go] + s
      tgt = s + [vocab.eos]
      # word2idx, pad
      batches.append((torch.LongTensor(src).t().contiguous().to(device), \
                      torch.LongTensor(tgt).t().contiguous().to(device)))
    n += 1
  return batches
