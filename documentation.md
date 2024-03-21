# Documentation for transformer.py

### Built off of Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY) video lecture. (Documentation written with the help of Claude 3 Sonnet to help explain things better!) 

## Overview

### Libraries and Modules

- `os.path`: Interact with the operating system's file system
- `requests`: Send HTTP requests and download files
- `torch`: Core PyTorch library for building and training neural networks
- `torch.nn`: Classes and functions for creating and manipulating neural network layers and models
- `torch.nn.functional`: Functional operations for neural networks (activation functions, loss functions, etc.)

### Downloading the Dataset

- Download the tiny Shakespeare dataset from Andrej Karpathy's GitHub repository
- Check if the `input.txt` file exists locally, if not, download it from the specified URL
- Read the contents of the dataset file into the `text` variable

### Defining Hyperparameters

- `batch_size`: Number of sequences processed in parallel during training
- `context_length`: Length of the context window for making predictions
- `max_iters`: Maximum number of steps for training | Dataset Size / (Batch Size * Context Length) = # of Iterations to get one Epoch
- `eval_interval`: Frequency of evaluating the model's performance during training
- `learning_rate`: Initial learning rate for the optimization algorithm
- `device`: Device on which the model and computations will run ('cuda' or 'cpu')
- `eval_iters`: Number of iterations for estimating the model's performance during evaluation
- `n_embed`: Dimensionality of the embedding vectors
- `n_head`: Number of attention heads in the Multi-Head Attention mechanism
- `n_layer`: Number of Transformer blocks stacked in the model's architecture
- `dropout`: Dropout rate for regularization

### Character-Level Tokenization

- Create a sorted list of unique characters (`chars`) from the dataset
- Compute the vocabulary size (`vocab_size`)
- Define mapping dictionaries: `string_to_int` and `int_to_string`
- Define `encode` and `decode` functions for converting between strings and integer sequences

### Encoding the Dataset

- Encode the entire dataset into a PyTorch tensor (`data`) using the `encode` function

### Splitting the Dataset

- Split the encoded dataset into training (`train_data`) and validation (`val_data`) sets (90% and 10%, respectively)

### Loading Data Batches

- Define the `get_batch` function to load batches of input and target sequences from the training or validation data

### Estimating Loss

- Define the `estimate_loss` function to calculate the average loss on the training and validation sets over a specified number of iterations (`eval_iters`)

## Model Architecture

### Head

- Implements a single attention head in the Multi-Head Attention mechanism
- Computes the key, query, and value vectors from the input embeddings
- Calculates the attention scores (affinities) by taking the dot product of the queries and transposed keys
- Masks future positions in the attention scores
- Applies softmax to the attention scores to obtain the attention weights
- Computes the output of the attention head by taking the weighted sum of the value vectors

### MultiHeadAttention

- Consists of multiple `Head` modules (`self.heads`) and a projection layer (`self.proj`)
- Computes the multi-head attention by combining the outputs of the individual attention heads
- Projects the combined output back to the original embedding dimension

### FeedForward

- Implements the position-wise feed-forward network in the Transformer block
- Consists of two linear layers with ReLU activation in between

### Block

- Combines the Multi-Head Attention and Feed-Forward components in a single block
- Applies layer normalization and residual connections around each component
- Utilizes dropout for regularization

### Transformer

- The main Transformer model, composed of stacked `Block` modules
- Accepts input and target sequences, and computes the logits (unnormalized log probabilities) and loss
- Implements the training loop and evaluation functionality

## Loss Function and Training

- Uses cross-entropy loss for training the language model
- Utilizes the AdamW optimizer for weight updates
- Tracks the training and validation losses during training
- Evaluates the model's performance on the validation set at regular intervals
- Early stopping based on the validation loss to prevent overfitting

## Text Generation

- Implements the `sample` function for generating text samples from the trained language model