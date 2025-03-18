#!/usr/bin/env python3
"""
Script to inspect Wikipedia dataset from the existing tokenized data
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer
import traceback

# Location of existing tokenized data
tokenized_data_path = "/data/s4422090/wiki_data/wiki_tokens.npy"

def inspect_tokenized_data():
    """Inspect the already tokenized data"""
    try:
        print(f"Loading tokenized data from: {tokenized_data_path}")
        # Load the tokenized data
        tokens = np.load(tokenized_data_path)
        
        print(f"\nTokenized data shape: {tokens.shape}")
        print(f"Data type: {tokens.dtype}")
        
        # Print some statistics
        print(f"\nStatistics:")
        print(f"Mean token ID: {tokens.mean():.2f}")
        print(f"Min token ID: {tokens.min()}")
        print(f"Max token ID: {tokens.max()}")
        
        # Load tokenizer to decode the tokens
        print("\nLoading tokenizer to decode the tokens...")
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/gpt-neox-olmo-dolma-v1_5",
            cache_dir="/data/s4422090/huggingface_cache/transformers/tokenizers"
        )
        
        # Get the first 1024 tokens (full first sequence)
        first_sequence = tokens[0]
        
        # Print first 1000 token IDs
        print(f"\nFirst 1000 token IDs:")
        print(first_sequence[:1000])
        
        # Try to decode them
        text = tokenizer.decode(first_sequence[:1000].tolist())
        print(f"\nDecoded text from first 1000 tokens:")
        print(text)
        
        # Print the first 20 words (using tokenizer to get words)
        print(f"\nFirst 1000 tokens broken down by words:")
        all_words = []
        current_pos = 0
        max_display = 50  # Show the first 50 words/tokens with their IDs
        
        for i in range(min(1000, len(first_sequence))):
            # Decode single token to show word-by-word breakdown
            token_text = tokenizer.decode([first_sequence[i]])
            if current_pos < max_display:
                print(f"Token {i}: ID={first_sequence[i]}, Text='{token_text}'")
                current_pos += 1
            
            # Add to our list of words
            all_words.append(token_text)
        
        print(f"\nTotal words/tokens in first 1000 tokens: {len(all_words)}")
        
        # Print the actual batching information
        print("\nTraining batching information:")
        print(f"Number of sequences: {tokens.shape[0]}")
        print(f"Sequence length: {tokens.shape[1]}")
        print(f"Total tokens in dataset: {tokens.shape[0] * tokens.shape[1]}")
        
        # Get frequency distribution of tokens
        unique, counts = np.unique(tokens.flatten(), return_counts=True)
        sorted_indices = np.argsort(-counts)  # Sort in descending order
        
        print("\nTop 20 most frequent tokens:")
        for i in range(min(20, len(sorted_indices))):
            token_id = unique[sorted_indices[i]]
            count = counts[sorted_indices[i]]
            token_text = tokenizer.decode([token_id])
            print(f"Token ID: {token_id}, Count: {count}, Text: '{token_text}'")
        
    except Exception as e:
        print(f"Error inspecting tokenized data: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    inspect_tokenized_data() 