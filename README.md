# MyGPT
**Atharv Kshirsagar 210050025**

**Library Installation**

If you want to `import mingpt` into your project:
```

git clone 
cd MyGPT
cd myGPT
pip install -e .
```

**Unit tests**
Not amazing but could try

```
python -m unittest discover tests
```

**Important Files**

`MyGPT/myGPT/mygpt :` This file conatins the main model has has three main files. `MyGPT/myGPT/mygpt/models.py` contains the implementation of transformer model.'MyGPT/myGPT/mygpt/bpe.py' contains the implemamtation of subword tokenization algorithm. 'MyGPT/myGPT/mygpt/trainer.py' is used to train the model and is independent of the GPT model.
 `MyGPT/myGPT/projects/adder` Our model is trained to add ndigits number and predict the sum.It uses Autoregressive  Generation.

 There is a `tunedmyGPT` folder that is which is myGPT with some changes to perform char and image related tasks.Please USE GPU or reduce batch size in case of Out of Memory Bound .Recommended to run every model in this file in Google Collab.

Model: block_size = 128, n_layer=3, n_head=8, n_embd=512

Trainer: max_epochs=150, batch_size=256, learning_rate=6e-4, lr_decay=True, warmup_tokens=512 * 20, final_tokens=150 * len(train_dataset) * block_size

(this is using old play_char.ipynb with smaller epoch size)

tunedmyGPT:

```
number of parameters: 9.983488e+06
epoch 1 iter 33: train loss 2.61996. lr 5.999354e-04:  [00:08<00:00,  4.19it/s]
epoch 2 iter 33: train loss 2.22801. lr 5.997392e-04:  [00:07<00:00,  4.35it/s]
epoch 3 iter 33: train loss 2.06643. lr 5.994116e-04:  [00:07<00:00,  4.36it/s]
......
epoch 50 iter 33: train loss 1.00424. lr 4.500336e-04:  [00:07<00:00,  4.31it/s]
......
epoch 100 iter 33: train loss 0.71006. lr 1.500168e-04:  [00:07<00:00,  4.31it/s]
......
epoch 148 iter 33: train loss 0.60897. lr 6.000000e-05:  [00:07<00:00,  4.32it/s]
epoch 149 iter 33: train loss 0.61404. lr 6.000000e-05:  [00:07<00:00,  4.31it/s]
epoch 150 iter 33: train loss 0.60880. lr 6.000000e-05:  [00:07<00:00,  4.32it/s]
```
```
myGPT:

number of parameters: 9.590272e+06
epoch 1 iter 33: train loss 2.64701. lr 5.999354e-04:  [00:07<00:00,  4.39it/s]
epoch 2 iter 33: train loss 2.49893. lr 5.997392e-04:  [00:07<00:00,  4.62it/s]
epoch 3 iter 33: train loss 2.43154. lr 5.994116e-04:  [00:07<00:00,  4.60it/s]
......
epoch 50 iter 33: train loss 1.08477. lr 4.500336e-04:  [00:07<00:00,  4.55it/s]
......
epoch 100 iter 33: train loss 0.80659. lr 1.500168e-04:  [00:07<00:00,  4.57it/s]
......
epoch 148 iter 33: train loss 0.67977. lr 6.000000e-05:  [00:07<00:00,  4.57it/s]
epoch 149 iter 33: train loss 0.69573. lr 6.000000e-05:  [00:07<00:00,  4.57it/s]
epoch 150 iter 33: train loss 0.68462. lr 6.000000e-05:  [00:07<00:00,  4.51it/s]
```

**Troubleshooting**

    *Reduce batch_size if you encounter out-of-video-memory.
    *Set num_workers to 0 if you are using Windows.
    *Remember to change final_tokens when you change max_epochs.

**References**

**Time-weighting**
```
self.time_weighting = nn.Parameter(torch.ones(self.n_head, config.block_size, config.block_size))
......
att = F.softmax(att, dim=-1)
att = att * self.time_weighting[:,:T,:T] # this is "time-weighting"
att = self.attn_drop(att)
```
Time-weighting works because tokens from different distances shall have different impacts on the current token.
Moreover, the self-attention effects shall be changed for early tokens because they have shorter history-windows.

p.s. because time_weighting is mostly a circulant matrix, use the following code to save more parameters:
```
self.time_weight = nn.Parameter(torch.ones(self.n_head, config.block_size))
self.time_bias = nn.Parameter(torch.ones(self.n_head, 1, config.block_size)) # dealing with early tokens 
......
ww = F.pad(self.time_weight, (0, TT))
ww = torch.tile(ww, [TT])
ww = ww[:, :-TT].reshape(-1, TT, 2 * TT - 1)
ww = ww[:, :, TT-1:]
ww = ww[:, :T, :T] # make a circulant matrix
time_weighting = ww * self.time_bias[:, :, :T]
```
p.s. with time-weighting, you can even remove positional encoding in deep models.

p.s. there might be a closed-form solution for optimal time-weighting.

Code Reference:

    `openai/gpt-2` 
    `openai/image-gpt` 
    `huggingface/transformers` 
     Google's "Attention is All you Need"
     Andrej Karpathy's Youtube lectures.

**More**

     *We can also use our model for Quantization but due to unavaliability of GPU I am not able to do it.
     
     *We can improve our model for GPT-4
     
     *Hyperparameters may not be great.
    

    
