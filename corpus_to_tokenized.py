import torch
from collections import Counter
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

class Vocab(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []

        with open(path) as f:
            for line in f:
                w = line.split()[0]
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']
        self.nspecial = 5

    @staticmethod
    def build(sents, path, size):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        words = [w for s in sents for w in s]
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.most_common(size):
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk

        with open(path, 'w') as f:
            for w in v:
                f.write('{}\t{}\n'.format(w, cnt[w]))

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents

def get_batch(x, vocab, model_type, device):
    max_len = max([len(s) for s in x])
    if model_type == 'transformer':
        enc_input, dec_input, target = [], [], []
        for s in x:
            t = tokenizer.tokenize(" ".join(s))
            s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in t]
            padding = [vocab.pad] * (max_len - len(s))
            enc_input.append([vocab.go] + s_idx + padding + [vocab.eos])
            dec_input.append([vocab.go] + s_idx + padding)
            target.append(s_idx + padding + [vocab.eos])
        return torch.LongTensor(enc_input).t().contiguous().to(device), \
            torch.LongTensor(dec_input).t().contiguous().to(device), \
            torch.LongTensor(target).t().contiguous().to(device)
    else:
        go_x, x_eos = [], []
        for s in x:
            s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
            padding = [vocab.pad] * (max_len - len(s))
            go_x.append([vocab.go] + s_idx + padding)
            x_eos.append(s_idx + [vocab.eos] + padding)
        return torch.LongTensor(go_x).t().contiguous().to(device), \
            torch.LongTensor(x_eos).t().contiguous().to(device)
            
def get_tokenized_batches(data, vocab, model_type, batch_size, device):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch(data[i: j], vocab, model_type, device))
        i = j
    return batches, order
