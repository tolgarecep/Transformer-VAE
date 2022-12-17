# Transformer based & LSTM based Variational Autoencoders - with BERTurk tokenizer
## Training
`vocab.txt` in `checkpoints/model_type` is used as vocabulary.
### Transformer-VAE
```console
python train.py --train data/yelp/train.txt --valid data/yelp/valid.txt --save-dir checkpoints/transformer --epochs 1 \  
                --dim_z 128 --nlayers 1 --dim_h 1024 --dropout 0.5 \  
                --model_type transformer --nhead 8 --dim_feedforward 2048 --max_len 1000
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
