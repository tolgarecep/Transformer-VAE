import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

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

def get_tokenized_batches(data, vocab, batch_size, device):
    # Tokenize sentences with BERTurk tokenizer
    data_tokenized = []
    for i, s in enumerate(data):
        sentence = ' '.join(s)
        tokenized = tokenizer.tokenize(sentence)
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
