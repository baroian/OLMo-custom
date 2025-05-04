"""
Command line argument parsing module.
"""

import argparse


def parse_args_train():
    """
    Parse command line arguments for OLMo training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train an OLMo model on text data")
    
    # Training parameters
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--steps", type=int, default=100, 
                        help="Total training steps (default: 100)")
    parser.add_argument("--batch-size", type=int, default=2, 
                        help="Batch size (default: 2)")
    
    # Inference parameters
    parser.add_argument("--inference-interval", type=int, default=200, 
                       help="Run inference every N steps (default: 200)")
    parser.add_argument("--inference-prompt", type=str, default="Dutch is ", 
                       help="Prompt to use for inference (default: 'Dutch is ')")

    # wandb parameters
    parser.add_argument("--wandb-project", type=str, default="olmo-training",
                        help="Wandb project name (default: 'olmo-training')")
    parser.add_argument("--wandb-name", type=str, default="olmo-train",
                        help="Wandb run name (default: 'olmo-train')")
                        

    # data prep parameters
    parser.add_argument("--sequence-length", type=int, default=1024,
                        help="Sequence length for tokenization (default: 1024)")
                    
    parser.add_argument("--data-dir", type=str, default=None, 
                    help="Data directory to use")
    
    return parser.parse_args() 

def parse_args_data_prep():
    """
    Parse command line arguments for data preparation.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Prepare data for OLMo training")
    
    parser.add_argument("--sequence-length", type=int, default=1024,
                        help="Sequence length for tokenization (default: 1024)")
    parser.add_argument("--output-file", type=str, default="wiki_tokens.npy",
                        help="Output filename for tokenized data (default: wiki_tokens.npy)")
    parser.add_argument("--target-tokens", type=int, default=300_000,
                        help="Target number of tokens to collect (default: 3.5B)")
    parser.add_argument("--data-dir", type=str, default=None, 
                    help="Data directory to use")
    parser.add_argument("--percent-of-articles", type=float, default=0.01,
                    help="Percentage of articles to use (default: 0.01)")
    
    return parser.parse_args()