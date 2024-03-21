import os.path
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

#Check if file exists, if not download it. If file cannot be downloaded, give response code and exit.
if os.path.isfile('input.txt') == False:
    response = requests.get(url)

    if response.status_code == 200:
        with open('input.txt', 'wb') as file:
            file.write(response.content)
    else:
        print('Failed to download file:', url, '\nResponse Code:', response.status_code)
        exit()

#Open file and store into 'text'
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Hyperparameters
batch_size = 64  #How many sequences will be processed in parallel
context_length = 256  #Context length for predictions
max_iters = 3000  #How many steps/iters the training goes through
#dataset size / (batch_size * context_length) = number of iters to get an epoch | Ex: 300000 / (64 * 256) = 18.1 -> 19
eval_interval = 300 #How often the model is evaluated
learning_rate = 3e-4  #Learning rate of the model as it trains
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 #How many iterations for eval are sampled and averaged for losses
n_embed = 384  #Embedding dimension
n_head = 6  #Number of attention heads
n_layer = 6  #Number of transformer blocks
dropout = 0.2  #Dropout rate

#Unique Characters inside the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Simple Character level tokenizer
string_to_int = { ch:i for i,ch in enumerate(chars) } #Loop over characters and assign a int according to where character is inside 'chars', encoding
int_to_string = { i:ch for i,ch in enumerate(chars) } #Reverse of encoding, which is for decoding
encode = lambda s: [string_to_int[c] for c in s] #Takes in a string, outputs a list of ints
decode = lambda l: ''.join([int_to_string[i] for i in l]) #Takes in a string, outputs a list of ints

#Encode entire dataset, then store into a torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)

#Split data into train and val
n = int(0.9 * len(data))  #90% is for train, rest (10%) is val
train_data = data[:n]
val_data = data[n:]

#Loading the data
def get_batch(split):
    data = train_data if split == 'train' else val_data  #Depending on the value of 'split' ('train' or 'val'), select the appropriate data (train_data or val_data)
    ix = torch.randint(len(data) - context_length, (batch_size,)) #Generate random indices (ix) within the valid range (len(data) - context_length) of size (batch_size,)
    x = torch.stack([data[i:i + context_length] for i in ix]) #Create a batch of input sequences (x) by stacking context_length tensors from data at the randomly selected indices
    y = torch.stack([data[i + 1:i + context_length + 1] for i in ix]) #Create a batch of target sequences (y) by stacking context_length+1 tensors from data at the randomly selected indices (shifted by 1)
    x, y = x.to(device), y.to(device) #Move the input and target tensors to the selected device
    return x, y

#Calculate the losses for all iters
@torch.no_grad()
def estimate_loss():
    out = {} #Initialize an empty dictionary to store losses
    model.eval() #Set the model to evaluation mode
    for split in ['train', 'val']: #Loop through 'train' and 'val' splits
        losses = torch.zeros(eval_iters) #Initialize a tensor of zeros with length eval_iters to store losses
        for k in range(eval_iters): #Loop through eval_iters
            X, Y = get_batch(split) #Get a batch of input and target sequences
            logits, loss = model(X, Y) #Pass the input and target sequences through the model to get logits and loss
            losses[k] = loss.item() #Store the loss for the current iteration
        out[split] = losses.mean() #Calculate the mean loss for the current split and store it in the 'out' dictionary
    model.train() #Set the model back to training mode
    return out #Return the dictionary containing the average losses for 'train' and 'val' splits

#Head class for self-attention mechanism
class Head(nn.Module):
    #One head of Self-Attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)  #Linear layer for key
        self.query = nn.Linear(n_embed, head_size, bias=False)  #Linear layer for query
        self.value = nn.Linear(n_embed, head_size, bias=False)  #Linear layer for value
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        #Lower triangular matrix for masking future positions

        self.dropout = nn.Dropout(dropout)  #Dropout layer

    def forward(self, x):
        #x has shape (B, T, C)
        #B: Batch size (number of sequences in the batch)
        #T: Sequence length (context_length)
        #C: Embedding dimension (n_embed)
        B, T, C = x.shape
        k = self.key(x)  #(B, T, head_size)
        q = self.query(x)  #(B, T, head_size)
        #Compute attention scores ('Affinities')
        wei = q @ k.transpose(-2, -1) * C ** -0.5  #(B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        #The attention scores represent the similarity between each query and key pair
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  #(B, T, T)
        #Mask future positions and set them to -inf, so they are not attended to
        wei = F.softmax(wei, dim=-1)  #(B, T, T)
        #Apply softmax to get attention weights
        wei = self.dropout(wei)  #Apply dropout

        #Perform the weighted aggregation of the values
        v = self.value(x)  #(B, T, head_size)
        out = wei @ v  #(B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        #The output represents the weighted sum of the values, based on the attention weights
        return out

#Multi-Head Attention class
class MultiHeadAttention(nn.Module):
    #Multiple heads of Self-Attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  #Create multiple heads
        self.proj = nn.Linear(n_embed, n_embed)  #Linear layer for projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x has shape (B, T, C)
        #For each head, the input is projected to (B, T, head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  #Concatenate outputs from all heads, shape: (B, T, num_heads * head_size)
        out = self.dropout(self.proj(out))  #Apply dropout and projection to get shape (B, T, C)
        return out

#Feed-Forward Network class
class FeedForward(nn.Module):
    #Simple linear layer followed by a non-linearity

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  #Linear layer, 4 times larger than embedding dimension
            nn.ReLU(),  #ReLU activation
            nn.Linear(4 * n_embed, n_embed),  #Linear layer to map back to embedding dimension
            nn.Dropout(dropout),  #Dropout layer
        )

    def forward(self, x):
        #x has shape (B, T, C)
        return self.net(x)  #Output has shape (B, T, C)

#Transformer Block class
class Block(nn.Module):
    #Transformer Block: Communication followed by Computation

    def __init__(self, n_embed, n_head):
        #n_embed: Embedding dimension
        #n_head: The number of heads we will like
        super().__init__()
        head_size = n_embed // n_head  #Head size calculated from embedding dimension and number of heads
        self.sa = MultiHeadAttention(n_head, head_size)  #Multi-Head Attention layer
        self.ffwd = FeedForward(n_embed)  #Feed-Forward Network
        self.ln1 = nn.LayerNorm(n_embed)  #Layer Normalization 1
        self.ln2 = nn.LayerNorm(n_embed)  #Layer Normalization 2

    def forward(self, x):
        #x has shape (B, T, C)
        x = x + self.sa(self.ln1(x))  #Residual connection with Multi-Head Attention
        x = x + self.ffwd(self.ln2(x))  #Residual connection with Feed-Forward Network
        return x  #Output has shape (B, T, C)

#Defining Bigram Language Model using torch.nn.Module
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #Token Embedding Table
        self.position_embedding_table = nn.Embedding(context_length, n_embed) #Position Embedding Table
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)]) #Stack of Transformer Blocks
        self.ln_f = nn.LayerNorm(n_embed)  #Final Layer Norm
        self.lm_head = nn.Linear(n_embed, vocab_size)  #Linear layer for language model head

    def forward(self, idx, targets=None):
        #idx has shape (B, T)
        #targets has shape (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  #(B, T, C) | Obtain token embeddings by passing the input token indices (idx) through the token embedding table
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  #(T, C) | Obtain positional embeddings by passing a range of position indices (0 to T-1) through the position embedding table
        x = tok_emb + pos_emb  #(B, T, C) | Combine the token embeddings and positional embeddings by summing them together
        x = self.blocks(x)  #(B, T, C) | Pass the combined embeddings through the stack of Transformer blocks
        x = self.ln_f(x)  #(B, T, C) | Apply the final layer normalization to the output of the Transformer blocks
        logits = self.lm_head(x)  #(B, T, vocab_size) | Obtain the logits (unnormalized log probabilities) for each token in the vocabulary by passing the normalized output through the language model head

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape #Unpack the batch size (B), sequence length (T), and embedding dimension (C) from the logits tensor shape
            logits = logits.view(B * T, C)  #(B*T, vocab_size) |  Reshape the logits tensor to combine the batch and sequence dimensions into a single dimension
            targets = targets.view(B * T)  #(B*T) | Reshape the target tensor to combine the batch and sequence dimensions into a single dimension
            loss = F.cross_entropy(logits, targets) #Calculate the cross-entropy loss between the reshaped logits and targets

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        #idx is current context of characters in a batch
        for _ in range(max_new_tokens):
            #Crop idx to the last context_length tokens
            idx_cond = idx[:, -context_length:]  #(B, context_length)
            #Get predictions
            logits, loss = self(idx_cond)
            #Focus only on the last time step
            logits = logits[:, -1, :]  #(B, vocab_size)
            #Apply softmax to get prob.
            probs = F.softmax(logits, dim=-1)  #(B, vocab_size)
            #Samples from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  #(B, 1)
            #Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  #(B, T+1)
        return idx

model = BigramLanguageModel()  #Create a Bigram Language Model
m = model.to(device)  #Move model to selected 'device'
print(f"{sum(p.numel() for p in m.parameters()):,d}", 'parameters')  #Print the number of parameters in the model

#Create AdamW PyTorch optimizer with selected 'learning_rate'
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    #Every now and then, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step: {iter}, Training Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

    #Sample a batch of data
    xb, yb = get_batch('train')

    #Evaluate the loss with batch sample
    logits, loss = model(xb, yb)  #Evaluate Loss
    optimizer.zero_grad(set_to_none=True)  #Zero gradients from previous iteration
    loss.backward()  #Get gradients for all the parameters
    optimizer.step()  #Update parameters based on gradients

#Generate from the freshly trained model
#Create a 1 by 1 tensor, data type is 64-bit int (Zero is element for new line character, makes sense in this context as it is new generation)
#and move to selected 'device'
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))