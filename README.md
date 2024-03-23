# simpliTransformer - A dead simple, character-level, transformer-based language model

## Overview

- **Data Preprocessing**: Downloads the tiny Shakespeare dataset, handles character-level tokenization, and encodes the dataset into integer sequences.
- **Model Architecture**: Implements the Transformer architecture, including the Multi-Head Attention mechanism, Feed-Forward network, and stacked Transformer blocks.
- **Training Loop**: Defines the training loop, loss function, and optimization algorithm for training the language model on the character-level dataset.
- **Text Generation**: Provides functionality for generating text samples from the trained language model.

The `transformer.py` file is heavily commented to allow for anyone to go in and hack around to get a grasp of a simple transformer-based language model. I am mainly creating this repository so I can use it as a reference for myself for a basic decoder-only transformer language model, but I figured that others could find this useful! <3

## Requirements

- Python 3.x
- PyTorch
- requests
- numpy

## Usage

1. Clone the repository.
2. Run the `transformer.py` script to train the model and generate text samples.

## Acknowledgements

This implementation is based on Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video lecture, with additional documentation and explanation provided by Claude 3 Sonnet because I am bad at explaining things.
