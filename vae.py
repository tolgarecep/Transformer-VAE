import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - np.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class VAE(nn.Module):
    def __init__(self, vocab, args, device, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.embed = nn.Embedding(vocab.size, args.dim_emb if args.model_type == 'lstm' else args.dim_h).to(device)
        self.h2mu = nn.Linear(args.dim_h*2 if args.model_type == 'lstm' else args.dim_h, args.dim_z).to(device)
        self.h2logvar = nn.Linear(args.dim_h*2 if args.model_type == 'lstm' else args.dim_h, args.dim_z).to(device)
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb if args.model_type == 'lstm' else args.dim_h).to(device)
        self.proj = nn.Linear(args.dim_h, vocab.size).to(device)
        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))
        
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TRANSFORMER_VAE(VAE):
    """Transformer based Variational Auto-encoder"""

    def __init__(self, vocab, args, device):
        super().__init__(vocab, args, device)
        cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.pe = PositionalEncoding(d_model=args.dim_h, dropout=args.dropout, max_len=args.max_len).to(self.device).to(self.device)
        # TransformerEncoderLayer is made up of self-attn and feedforward network.
        self.EncoderStack = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=args.dim_h, nhead=args.nhead, dim_feedforward=args.dim_feedforward, dropout=args.dropout) 
            for _ in range(args.nlayers)]).to(self.device)
        # TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
        self.DecoderStack = nn.ModuleList([nn.TransformerDecoderLayer(
            d_model=args.dim_h, nhead=args.nhead, dim_feedforward=args.dim_feedforward, dropout=args.dropout) 
            for _ in range(args.nlayers)]).to(self.device)

    def flatten(self):
        self.EncoderStack.flatten_parameters()
        self.DecoderStack.flatten_parameters()
        
    def encode(self, src):
        x = self.pe(self.embed(src))
        for layer in self.EncoderStack:
            x = layer(x)
        return self.h2mu(x), self.h2logvar(x)

    def decode(self, src, z, trg):
        trg_mask = generate_square_subsequent_mask(trg.shape[0])
        x = self.pe(self.embed(trg))
        memory = self.pe(self.embed(src)) + self.z2emb(z)
        for layer in self.DecoderStack:
            x = layer(tgt=x, memory=memory, tgt_mask=trg_mask)
        logits = self.proj(x)
        return logits

    def forward(self, src, trg):
        mu, logvar = self.encode(src)
        z = reparameterize(mu, logvar)
        logits = self.decode(src, z, trg)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets):
        mu, logvar, _, logits = self(inputs, targets)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        self.opt.step()
"""
    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        for l in range(max_len):
            sents.append(input)
            logits = self.decode(z, input)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)"""

class LSTM_VAE(VAE):
    """LSTM based Variational Auto-encoder"""

    def __init__(self, vocab, args, initrange=0.1):
        super().__init__(vocab, args)
        self.drop = nn.Dropout(args.dropout)
        self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0)

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (h, _) = self.E(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets):
        mu, logvar, _, logits = self(inputs)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        self.opt.step()
