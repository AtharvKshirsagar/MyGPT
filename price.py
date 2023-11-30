#!pip install ccxt

import ccxt
import pandas as pd
import datetime
import time

def fetch_historical_data(symbol='BTC/USDT', timeframe='1h', limit=10000):
    all_candles = []
    binance = ccxt.binance({
        'rateLimit': 1200,
        'options': {
            'adjustForTimeDifference': True,
        }
    })

    start_date = int(datetime.datetime(2018, 1, 1, 10, 20).timestamp() * 1000)
    since = start_date

    while len(all_candles) < limit:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since, min(limit, 1000))
        
        if len(ohlcv) == 0:
            break

        since = ohlcv[-1][0]  # Update the 'since' timestamp for the next batch
        all_candles += ohlcv
        time.sleep(binance.rateLimit / 1000)  # Respect the rate limit of the exchange

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

df = fetch_historical_data()
df.head()

import matplotlib.pyplot as plt

# Convert 'timestamp' to datetime format if it's not already
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set 'timestamp' as the index of the DataFrame
df.set_index('timestamp', inplace=True)

# Plotting
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.title('Price over time')
plt.plot(df['close'], label='Close Price')
plt.xlabel('Timestamp')
plt.ylabel('Close Price (in USDT)')
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Volume over time')
plt.plot(df['volume'], label='Volume', color='orange')
plt.xlabel('Timestamp')
plt.ylabel('Volume')
plt.legend()

plt.tight_layout()
plt.show()

import pandas as pd

# Add a new column with the shifted close prices to compare with
df['close_shifted'] = df['close'].shift(-1)

# Calculate the encoded 'price_movement' based on your conditions
df['price_movement'] = 1  # Initialize as no change
df.loc[df['close'] < df['close_shifted'], 'price_movement'] = 2  # Price goes up
df.loc[df['close'] > df['close_shifted'], 'price_movement'] = 0  # Price goes down

# Drop the 'close_shifted' as it was only used for the calculation
df.drop('close_shifted', axis=1, inplace=True)

df.head(5)

import torch
from torch.utils.data import Dataset

class PriceMovementDataset(Dataset):
    
    def __init__(self, split, data, length=10):
        assert split in {'train', 'test'}
        self.split = split
        self.data = data  # This is your encoded 'price_movement' series as a PyTorch tensor
        self.length = length  # Sequence length for training/testing

    def get_vocab_size(self):
        return 3
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1
    
    def __len__(self):
        return len(self.data) - self.length  # Adjust length to account for sequence length
    
    def __getitem__(self, idx):
        
        # Create sequences
        inp = self.data[idx:idx+self.length]
        sol = self.data[idx+1:idx+self.length+1]

        inp = torch.from_numpy(inp)
        sol = torch.from_numpy(sol)

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length-1] = -1

        return x, y
    

price_movement = df['price_movement'].values  # Assuming 'price_movement' is the column you want
train_dataset = PriceMovementDataset('train', price_movement[:9980])
test_dataset = PriceMovementDataset('test', price_movement[9980:10000])



# create a GPT instance
from mygpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

# create a Trainer object
from mygpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
train_config.max_iters = 6000
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

# now let's perform some evaluation
model.eval();

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mygpt.utils import set_seed
set_seed(3407)

def eval_split(trainer, split, max_batches):
    dataset = {'train':train_dataset, 'test':test_dataset}[split]
    n = train_dataset.length # naugy direct access shrug
    results = []
    mistakes_printed_already = 0
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(inp, n, do_sample=False) # using greedy argmax, not sampling
        sol_candidate = cat[:, n:] # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        #print("sol: ", sol[10:15,-8:])
        #print("sol_candidate: ", sol_candidate[10:15,-8:])
        correct = (sol == sol_candidate).all(1).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
        #print("correct:",correct)
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if not correct[i] and mistakes_printed_already < 3: # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                print("GPT claims that with %s the future price dynamics is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))
        if max_batches is not None and b+1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()

# run a lot of examples from both train and test through the model and verify the output correctness
with torch.no_grad():
    train_score = eval_split(trainer, 'train', max_batches=50)
    test_score  = eval_split(trainer, 'test',  max_batches=50)

