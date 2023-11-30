# MyGPT - CS337 Machine Learning Project
**Atharv Kshirsagar 210050025**

**Library Installation**

If you want to `import mygpt` into your project:
```

git clone https://github.com/AtharvKshirsagar/MyGPT.git 
cd MyGPT
cd myGPT
pip install -e .
```
**Other normal ML libraries(like numpy,pytorch) can be installed using pip**

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

Trainer: max_epochs=150, batch_size=32, learning_rate=6e-4, lr_decay=True, warmup_tokens=512 * 20, final_tokens=150 * len(train_dataset) * block_size

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

**More (Beyond Project)**
     *We can improve our model for GPT-4
     
     *Hyperparameters may not be great.
     
     *We can also use our model for Quantization but due to unavaliability of GPU I am not able to do it.

     *We can train our model to generate music

     *We can train our model to generate reponses based on queries related to particular PDF just as PDFGPT would do.

     `price.py :` Though this project is nearly completed but there may be some unexpected errors.This project predicts future cryptocurrency price.May be nearly                    35 percent accuracy is acheived.

    These lines are part of the training process for your model. They provide information about the training iterations, the time it takes to process each iteration (iter_dt), the iteration number (iter), and the corresponding training loss.

```
iter_dt 0.00ms; iter 0: train loss 1.14302
iter_dt 43.69ms; iter 100: train loss 0.52618
iter_dt 44.73ms; iter 200: train loss 0.10185
iter_dt 43.73ms; iter 300: train loss 0.07574
iter_dt 43.82ms; iter 400: train loss 0.06957
iter_dt 44.19ms; iter 500: train loss 0.07069
iter_dt 43.72ms; iter 600: train loss 0.06959
iter_dt 43.98ms; iter 700: train loss 0.06859
iter_dt 43.63ms; iter 800: train loss 0.07137
iter_dt 43.94ms; iter 900: train loss 0.07161
iter_dt 45.07ms; iter 1000: train loss 0.09922
iter_dt 44.08ms; iter 1100: train loss 0.06671
iter_dt 44.66ms; iter 1200: train loss 0.07007
iter_dt 44.33ms; iter 1300: train loss 0.07830
iter_dt 44.60ms; iter 1400: train loss 0.07003
iter_dt 45.69ms; iter 1500: train loss 0.07011
iter_dt 43.63ms; iter 1600: train loss 0.07038
iter_dt 45.33ms; iter 1700: train loss 0.07432
iter_dt 47.61ms; iter 1800: train loss 0.06970
iter_dt 43.76ms; iter 1900: train loss 0.06859
iter_dt 44.59ms; iter 2000: train loss 0.06708
iter_dt 44.50ms; iter 2100: train loss 0.07012
iter_dt 44.92ms; iter 2200: train loss 0.07733
iter_dt 43.97ms; iter 2300: train loss 0.06426
iter_dt 44.50ms; iter 2400: train loss 0.06657
iter_dt 44.61ms; iter 2500: train loss 0.07034
iter_dt 45.24ms; iter 2600: train loss 0.06785
iter_dt 43.92ms; iter 2700: train loss 0.07009
iter_dt 45.71ms; iter 2800: train loss 0.06966
iter_dt 43.55ms; iter 2900: train loss 0.06751
iter_dt 46.22ms; iter 3000: train loss 0.06903
iter_dt 43.65ms; iter 3100: train loss 0.06652
iter_dt 45.71ms; iter 3200: train loss 0.06858
iter_dt 43.96ms; iter 3300: train loss 0.07034
iter_dt 44.23ms; iter 3400: train loss 0.06906
iter_dt 44.45ms; iter 3500: train loss 0.06980
iter_dt 43.45ms; iter 3600: train loss 0.07650
iter_dt 43.51ms; iter 3700: train loss 0.07337
iter_dt 43.56ms; iter 3800: train loss 0.06968
iter_dt 43.59ms; iter 3900: train loss 0.06673
iter_dt 43.48ms; iter 4000: train loss 0.06983
iter_dt 51.54ms; iter 4100: train loss 0.06530
iter_dt 50.56ms; iter 4200: train loss 0.07022
iter_dt 7.47ms; iter 4300: train loss 0.07006
iter_dt 7.30ms; iter 4400: train loss 0.06798
iter_dt 7.30ms; iter 4500: train loss 0.07333
iter_dt 7.26ms; iter 4600: train loss 0.06978
iter_dt 7.35ms; iter 4700: train loss 0.06982
iter_dt 7.35ms; iter 4800: train loss 0.07159
iter_dt 7.27ms; iter 4900: train loss 0.07200
iter_dt 7.30ms; iter 5000: train loss 0.06849
iter_dt 7.28ms; iter 5100: train loss 0.08039
iter_dt 7.61ms; iter 5200: train loss 0.06816
iter_dt 10.22ms; iter 5300: train loss 0.06468
iter_dt 7.33ms; iter 5400: train loss 0.06698
iter_dt 7.34ms; iter 5500: train loss 0.07363
iter_dt 8.16ms; iter 5600: train loss 0.06835
iter_dt 7.62ms; iter 5700: train loss 0.07031
iter_dt 14.43ms; iter 5800: train loss 0.06916
iter_dt 9.88ms; iter 5900: train loss 0.06916

```
Lnes are part of the evaluation process for your model on both the training and test datasets. Let's break down the information provided:

Prediction and Ground Truth Comparison:
        Each line represents a specific example during evaluation.
        The line starts with "GPT claims that with [...]" where the numbers in square brackets represent the input sequence to your model.
        It then compares the model's prediction ("future price dynamics") with the ground truth ("gt" is short for ground truth) sequence.

Final Score:
        After processing a set number of batches during evaluation (in this case, 50 batches for both training and test sets), the final score is presented.
        The "final score" is given as a ratio of the number of correct predictions to the total number of examples evaluated.

In your specific output:

    For the training dataset, the final score is "2704/5000 = 54.08% correct." This indicates that, based on the evaluation criteria, the model made correct predictions for approximately 54.08% of the training examples.

    For the test dataset, the final score is "3/10 = 30.00% correct." This indicates that, based on the evaluation criteria, the model made correct predictions for 3 out of 10 test examples.

It's important to note that the evaluation results provide insights into how well the model generalizes to unseen data. A higher final score generally indicates better generalization performance. If needed, you can further analyze and adjust your model based on these results to improve its performance.   
    

    
```
GPT claims that with [0, 2, 0, 0, 0, 0, 2, 2, 0, 2] the future price dynamics is [2, 0, 0, 0, 0, 2, 2, 0, 2, 0] but gt is [2, 0, 0, 0, 0, 2, 2, 0, 2, 2]
GPT claims that with [2, 0, 0, 0, 0, 2, 2, 0, 2, 2] the future price dynamics is [0, 0, 0, 0, 2, 2, 0, 2, 2, 0] but gt is [0, 0, 0, 0, 2, 2, 0, 2, 2, 2]
GPT claims that with [0, 0, 0, 0, 2, 2, 0, 2, 2, 2] the future price dynamics is [0, 0, 0, 2, 2, 0, 2, 2, 2, 0] but gt is [0, 0, 0, 2, 2, 0, 2, 2, 2, 2]
train final score: 2704/5000 = 54.08% correct
GPT claims that with [0, 0, 2, 2, 0, 2, 0, 2, 0, 2] the future price dynamics is [0, 2, 2, 0, 2, 0, 2, 0, 2, 0] but gt is [0, 2, 2, 0, 2, 0, 2, 0, 2, 2]
GPT claims that with [2, 2, 0, 2, 0, 2, 0, 2, 2, 0] the future price dynamics is [2, 0, 2, 0, 2, 0, 2, 2, 0, 0] but gt is [2, 0, 2, 0, 2, 0, 2, 2, 0, 2]
GPT claims that with [2, 0, 2, 0, 2, 0, 2, 2, 0, 2] the future price dynamics is [0, 2, 0, 2, 0, 2, 2, 0, 2, 0] but gt is [0, 2, 0, 2, 0, 2, 2, 0, 2, 2]
test final score: 3/10 = 30.00% correct

```
