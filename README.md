# MyGPT - CS337 Machine Learning Project
**Atharv Kshirsagar 210050025**

**Library Installation**

If you want to `import mygpt` into your project:
```

git clone 
cd MyGPT
cd myGPT
pip install -e .
```

**Model Architecture**
![Alt text](data:image/png;base64,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)

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
    

    
