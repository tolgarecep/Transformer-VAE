import torch
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

def get_batch(x, vocab, device):
    go_x, x_eos = [], []
    max_len = max([len(s) for s in x])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

def get_tokenized_batches(data, max_len, vocab, batch_size, device):
    # Tokenize sentences with BERTurk tokenizer
    data_tokenized = []
    for i, s in enumerate(data):
        sentence = ' '.join(s)
        tokenized = tokenizer.tokenize(sentence, max_length=max_len, truncation=True)
        data_tokenized.append(tokenized)

    order = range(len(data_tokenized))
    z = sorted(zip(order, data_tokenized), key=lambda i: len(i[1]))
    order, data_tokenized = zip(*z)

    batches = []
    i = 0
    while i < len(data_tokenized):
        j = i
        while j < min(len(data_tokenized), i+batch_size) and len(data_tokenized[j]) == len(data_tokenized[i]):
            j += 1
        batches.append(get_batch(data_tokenized[i: j], vocab, device))
        i = j
    return batches, order
