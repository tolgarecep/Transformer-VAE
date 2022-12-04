import torch
from torch import nn

from decoders import GenerationDecoder, RecoginitonDecoder
from encoders import GenerationEncoder, RecognitionEncoder

def make_pad_mask(self, q, k):
  len_q, len_k = q.size(1), k.size(1)

  # batch_size x 1 x 1 x len_k
  k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
  # batch_size x 1 x len_q x len_k
  k = k.repeat(1, 1, len_q, 1)

  # batch_size x 1 x len_q x 1
  q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
  # batch_size x 1 x len_q x len_k
  q = q.repeat(1, 1, 1, len_k)

  mask = k & q
  return mask

def make_no_peak_mask(self, q, k):
  len_q, len_k = q.size(1), k.size(1)

  # len_q x len_k
  mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

  return mask

class RecognitionTransformer(nn.Module):
  def __init__(self, vocab, args):
    super().__init__()
    self.encoder = RecognitionEncoder(d_model=args.r_d_model,
                               n_head=args.r_n_head,
                               max_len=args.max_len,
                               ffn_hidden=args.ffn_hidden,
                               enc_voc_size=vocab.size,
                               drop_prob=args.dropout,
                               n_layers=args.r_n_layers,
                               device=args.device)
    self.decoder = RecognitionDecoder(d_model=args.r_d_model,
                               n_head=args.r_n_head,
                               max_len=args.max_len,
                               ffn_hidden=args.ffn_hidden,
                               enc_voc_size=vocab.size,
                               drop_prob=args.dropout,
                               n_layers=args.r_n_layers,
                               device=args.device)
    
  def forward(self, src, trg):
    src_mask = make_pad_mask(src, src)
    src_trg_mask = make_pad_mask(trg, src)
    trg_mask = make_pad_mask(trg, trg) * make_no_peak_mask(trg, trg)
    
    enc_src = self.encoder(src, src_mask)
    decoder_stack_output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
    return decoder_stack_output
  
class GenerationTrasnformer(nn.Module):
  def __init__(self, vocab, args):
    super().__init__()
    self.encoder = GenerationEncoder(d_model=args.g_d_model,
                               n_head=args.g_n_head,
                               max_len=args.max_len,
                               ffn_hidden=args.ffn_hidden,
                               enc_voc_size=vocab.size,
                               drop_prob=args.dropout,
                               n_layers=args.g_n_layers,
                               device=args.device)
    self.decoder = GenerationDecoder(d_model=args.g_d_model,
                               n_head=args.g_n_head,
                               max_len=args.max_len,
                               ffn_hidden=args.ffn_hidden,
                               enc_voc_size=vocab.size,
                               drop_prob=args.dropout,
                               n_layers=args.g_n_layers,
                               device=args.device)
    
  def forward(self, src, trg):
    src_mask = make_pad_mask(src, src)
    src_trg_mask = make_pad_mask(trg, src)
    trg_mask = make_pad_mask(trg, trg) * make_no_peak_mask(trg, trg)
    
    enc_src = self.encoder(src, src_mask)
    decoder_stack_output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
    return decoder_stack_output
