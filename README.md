## Transformer-based VAE
`vocab.txt` in `checkpoints/model_type` is used as vocabulary.

## To-do
- [ ] Further training & experiments
- [ ] Diagnose posterior collapse
- [ ] BERT-based VAE

## Results
Training data from trwiki, date 20220601.

### BERTurk (Baseline)
Represents sentences in a 768-dimensional vector space.
| text categorization (7) | sentiment analysis (binary) | sentiment analysis (neu, neg, pos) | text categorization (6)
| ------------- | ------------- | ------------- | ------------- |
| 0.8626 | 0.8783 | 0.9333 | 0.9130 |

### LSTM-VAE
3M, 1 epoch (# of parameters: ~72M)
| dim_z | text categorization (7) | sentiment analysis (binary) | sentiment analysis (neu, neg, pos) | text categorization (6)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| `dim_z=512` | 0.7 | 0.6918 | 0.8653 | 0.69 |
| `dim_z=1024` | 0.7619 | 0.7111 | 0.882 | 0.779 |
| `dim_z=2048` | 0.7673 | 0.7002 | 0.883 | 0.786 |

### Transformer-VAE
1.5M, 28 epoch, (# of parameters: ~38M)
| dim_z | text categorization (7) | sentiment analysis (binary) | sentiment analysis (neu, neg, pos) | text categorization (6)
| ------------- | ------------- | ------------- | ------------- | ------------- |
| `dim_z=256` | 0.7632 | 0.7742 | 0.856 | 0.753 |
| `dim_z=512` | 0.7734 | 0.7750 | 0.866 | 0.743 |

This research is supported by TUBITAK, The Scientific and Technological Research Council of Turkey (2247-C)
