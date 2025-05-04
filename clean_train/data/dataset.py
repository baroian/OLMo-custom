"""
Dataset preparation and loading module.
"""

import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer
import torch
import itertools # Added for chain.from_iterable

from olmo_core.data import NumpyDatasetConfig, NumpyDatasetType
from olmo_core.data import TokenizerConfig, TokenizerName
from olmo_core.utils import seed_all


def _download_and_tokenize_wiki(wiki_data_path, sequence_length, total_tokens_with_margin, percent_of_articles):
    """
    Download and tokenize Wikipedia data efficiently using datasets.map,
    then concatenate using NumPy.

    Args:
        wiki_data_path: Path to save tokenized data
        sequence_length: Sequence length
        total_tokens_with_margin: Total tokens needed with margin
    """

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        cache_dir=os.path.join(os.environ["TRANSFORMERS_CACHE"], "tokenizers")
    )

    # Get tokenizer configuration from OLMo-core
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    eos_token_id = tokenizer_config.eos_token_id

    # Download Wikipedia dataset
    print("Downloading Wikipedia dataset...")
    wiki_dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    print(f"Downloaded full Wikipedia dataset with {len(wiki_dataset)} articles")

    #percent_of_articles = 0.01
    wiki_subset = wiki_dataset.select(range(int(len(wiki_dataset) * percent_of_articles)))
    print(f"Downloaded subset of Wikipedia dataset with {len(wiki_subset)} articles")

    # Define the tokenization function to be mapped
    def tokenize_function(examples):
        # Tokenize texts - IMPORTANT: Use the tokenizer directly on the batch
        tokenized_output = tokenizer(examples["text"], truncation=False, padding=False)
        # Add EOS token manually if needed (tokenizer might handle this depending on config)
        for ids in tokenized_output["input_ids"]:
            if not ids or ids[-1] != eos_token_id:
                ids.append(eos_token_id)
        return tokenized_output

    # Set number of processes for mapping (adjust based on CPU cores)
    # num_proc = max(1, os.cpu_count() // 2 if os.cpu_count() else 1) # Use half the CPU cores, default to 1 if count fails
    num_proc = 16 # Example: Reduced number based on previous discussion
    print(f"Tokenizing dataset using {num_proc} processes...")

    # Apply tokenization using map with batching
    # Define features explicitly to help HF manage memory/types for large datasets


    
    tokenized_features = Features({
        # Keep original columns you might need later, or remove them all
        # 'id': Value('string'),
        # 'url': Value('string'),
        # 'title': Value('string'),
        # 'text': Value('string'), # Remove 'text' if you don't need it after tokenization
        'input_ids': Sequence(Value('int32')),
        'attention_mask': Sequence(Value('int8')) # attention_mask added by tokenizer
    })


    tokenized_dataset = wiki_subset.map(
        tokenize_function,
        batched=True, # Process data in batches
        num_proc=num_proc, # Use multiple CPU cores
        remove_columns=wiki_subset.column_names,  # Remove original text columns to save memory
        desc="Tokenizing articles",
        features=tokenized_features # Optional but recommended for large datasets
    )

    print("Finished tokenization.")

    # Concatenate using intermediate NumPy arrays (Option 2)
    list_of_token_arrays = []
    print("Collecting tokens into NumPy arrays using batching...")
    batch_size = 10000  # Adjust batch size based on memory/performance trade-off
    num_examples = len(tokenized_dataset)
    collected_count = 0

    # Create an iterator that yields batches
    batched_iterator = tokenized_dataset.select_columns(["input_ids"]).iter(batch_size=batch_size)

    for batch in tqdm(batched_iterator, desc="Collecting tokens in batches", total=(num_examples + batch_size - 1) // batch_size):
        # batch is like {'input_ids': [ [tokens1], [tokens2], ..., [tokens_batch_size] ]}
        # Flatten the list of lists in the batch efficiently
        tokens_in_batch = list(itertools.chain.from_iterable(batch['input_ids']))
        if not tokens_in_batch: # Handle potential empty results from tokenizer
            continue

        # Convert the flattened list for this batch into a numpy array
        list_of_token_arrays.append(np.array(tokens_in_batch, dtype=np.int32))
        collected_count += len(tokens_in_batch)

        # Check periodically if we have enough tokens
        if collected_count >= total_tokens_with_margin:
            print(f"\nCollected enough tokens ({collected_count:,}), stopping collection early.")
            break

    if not list_of_token_arrays:
        print("Error: No token arrays were collected. Cannot proceed.")
        return

    print(f"Collected {collected_count:,} total tokens in {len(list_of_token_arrays)} arrays.")
    print("Concatenating NumPy arrays...")
    # Concatenate all the collected numpy arrays into one large array
    all_tokens_np = np.concatenate(list_of_token_arrays)
    # It's good practice to clear the intermediate list to free up memory sooner
    del list_of_token_arrays
    print(f"Total concatenated tokens: {len(all_tokens_np):,}")


    # --- Reshape and Save --- (Using the NumPy array 'all_tokens_np')

    # Calculate the number of sequences we can actually make from collected tokens
    num_sequences_possible = len(all_tokens_np) // sequence_length

    # Calculate the number of sequences theoretically needed based on the target
    sequences_needed = (total_tokens_with_margin + sequence_length - 1) // sequence_length

    # Determine the number of sequences to actually create (the minimum of possible and needed)
    num_sequences_to_create = min(num_sequences_possible, sequences_needed)

    if num_sequences_possible == 0:
         print("Error: Collected tokens are less than the sequence length. Cannot create any sequences.")
         return # Exit if no sequences can be formed

    if num_sequences_possible < sequences_needed:
        print(f"Warning: Only able to create {num_sequences_possible:,} sequences from collected tokens.")
        print(f"Target token count ({total_tokens_with_margin:,}) would ideally yield {sequences_needed:,} sequences.")
        print(f"Creating {num_sequences_to_create:,} sequences.")
    else:
         print(f"Collected enough tokens ({len(all_tokens_np):,}) to create the target {sequences_needed:,} sequences.")
         print(f"Creating {num_sequences_to_create:,} sequences.")


    # Take only the tokens needed for the complete sequences we are creating
    # Ensure we don't try to slice beyond the collected tokens if concatenation stopped early
    final_token_count = num_sequences_to_create * sequence_length
    # Slice the final NumPy array directly
    tokens_to_use = all_tokens_np[:final_token_count]
    # Optionally clear the large concatenated array if memory is extremely tight *after* slicing
    # del all_tokens_np

    if tokens_to_use.size == 0:
        print("Error: No tokens available after slicing. Cannot create sequences.")
        return

    # Reshape the final NumPy array directly
    sequences = tokens_to_use.reshape(num_sequences_to_create, sequence_length)

    print(f"Created {sequences.shape[0]:,} sequences of length {sequence_length}")

    # Save tokenized data
    np.save(wiki_data_path, sequences)
    print(f"Saved tokenized data to {wiki_data_path}") 