import torch
from torch import nn

from masking_embedding import make_pad_mask, make_no_peak_mask
from encoder_decoder import Encoder, Decoder

class RecognitionTransformer(nn.Module):
  def __init__(self, vocab, args):
    super().__init__()
    self.encoder = Encoder(d_model=args.r_d_model,
                               n_head=args.r_n_head,
                               max_len=args.max_len,
                               ffn_hidden=args.ffn_hidden,
                               enc_voc_size=vocab.size,
                               drop_prob=args.dropout,
                               n_layers=args.r_n_layers,
                               device=args.device)
    self.decoder = Decoder(d_model=args.r_d_model,
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
  
class GenerationTransformer(nn.Module):
  def __init__(self, vocab, args):
    super().__init__()
    self.encoder = Encoder(d_model=args.g_d_model,
                               n_head=args.g_n_head,
                               max_len=args.max_len,
                               ffn_hidden=args.ffn_hidden,
                               enc_voc_size=vocab.size,
                               drop_prob=args.dropout,
                               n_layers=args.g_n_layers,
                               device=args.device)
    self.decoder = Decoder(d_model=args.g_d_model,
                               n_head=args.g_n_head,
                               max_len=args.max_len,
                               ffn_hidden=args.ffn_hidden,
                               enc_voc_size=vocab.size,
                               drop_prob=args.dropout,
                               n_layers=args.g_n_layers,
                               device=args.device)
    self.linear = nn.Linear(args.g_d_model, vocab.size)
    
  def forward(self, src, trg):
    src_mask = make_pad_mask(src, src)
    src_trg_mask = make_pad_mask(trg, src)
    trg_mask = make_pad_mask(trg, trg) * make_no_peak_mask(trg, trg)
    
    enc_src = self.encoder(src, src_mask)
    trg = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
    output = self.linear(trg)
    return output
