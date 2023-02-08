# Transformer based Variational Autoencoders - with BERTurk tokenizer
`vocab.txt` in `checkpoints/model_type` is used as vocabulary.

## Results
Training data from trwiki, date 20220601.

### LSTM-VAE
#### 3M, 1 epoch (# of parameters: ~72M)
| dim_z | text categorization (7) | sentiment analysis (binary) | sentiment analysis (neu, neg, pos) | text categorization (6)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| `dim_z=512` | 0.7 | 0.691844114102049 | 0.8653333333333333 | 0.69 |
| `dim_z=1024` | 0.7619047619047619 | 0.71112896745681 | 0.882 | 0.779 |
| `dim_z=2048` | 0.7673469387755102 | 0.7002812374447569 | 0.8833333333333333 | 0.786 |

### Transformer-VAE
#### 1.5M, 2 epoch (# of parameters: ~38M)
| latent space | text categorization (7) | sentiment analysis (binary) | sentiment analysis (neu, neg, pos) | text categorization (6)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| `dim_z=64` | 0.5156462585034014 | 0.7276014463640016 | 0.806 | 0.499 |
| `dim_z=128` | 0.5319727891156463 | 0.7420650863800723 | 0.828 | 0.54 |

#### 1.5M, 28 epoch, (# of parameters: ~38M)
| dim_z | text categorization (7) | sentiment analysis (binary) | sentiment analysis (neu, neg, pos) | text categorization (6)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| `dim_z=256` | 0.7632653061224491 | 0.7742065086380072 | 0.856 | 0.753 |
| `dim_z=512` | 0.773469387755102 | 0.7750100441944556 | 0.866 | 0.743 |
