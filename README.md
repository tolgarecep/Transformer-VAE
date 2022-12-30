## todo
- [ ] best transformer-vae trainable
- [ ] optimize transformer-vae code
- [ ] diagnose posterior collapse

# Transformer based & LSTM based Variational Autoencoders - with BERTurk tokenizer
## Training
`vocab.txt` in `checkpoints/model_type` is used as vocabulary.
### Transformer-VAE
```console
python train.py --train data/yelp/train.txt --valid data/yelp/valid.txt --save-dir checkpoints/transformer --epochs 1 \  
                --dim_z 128 --nlayers 1 --dim_h 512 --dropout 0.5 \  
                --model_type transformer --nhead 8 --dim_feedforward 1024 --pe_max_len 5000
```

### LSTM-VAE
```console
python train.py --train data/yelp/train.txt --valid data/yelp/valid.txt --save-dir checkpoints/lstm --epochs 1 \  
                --dim_z 128 --nlayers 1 --dim_h 1024 --dropout 0.5 \  
                --model_type lstm --dim_emb 512 
```

## Testing
```console
python test.py --reconstruct --data data/yelp/test.txt --output test --checkpoint checkpoints/transformer/
```
## Results
Training data from trwiki, date 20220601. Settings are in the console commands above.

### LSTM-VAE
#### 3M, 1 epoch
| dim_z | text categorization (7) | sentiment analysis (binary) | sentiment analysis (neu, neg, pos) | text categorization (6)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| `dim_z=512` | 0.7 | 0.691844114102049 | 0.8653333333333333 | 0.69 |
| `dim_z=1024` | 0.7619047619047619 | 0.71112896745681 | 0.882 | 0.779 |
| `dim_z=2048` | **0.7673469387755102** | **0.7002812374447569** | **0.8833333333333333** | **0.786** |

### Transformer-VAE
#### 1.5M, 2 epoch
| latent space | text categorization (7) | sentiment analysis (binary) | sentiment analysis (neu, neg, pos) | text categorization (6)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| `dim_z=64` | 0.5156462585034014 | 0.7276014463640016 | 0.806 | 0.499 |
| `dim_z=128` | 0.5319727891156463 | 0.7420650863800723 | 0.828 | 0.54 |
