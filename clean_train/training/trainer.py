"""
Training module for OLMo models.
"""

import time
import wandb

from olmo_core.train.train_module.transformer.config import TransformerTrainModuleConfig
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import WandBCallback
from training.callbacks.inference_callback import InferenceCallback
from olmo_core.train.common import Duration



def run_training(
    model, 
    optim_config, 
    data_loader, 
    save_folder, 
    args, 
    tokenizer_config, 
    device, 
    sequence_length
):
    """
    Run the training process.
    
    Args:
        model: The OLMo model
        optim_config: Optimizer configuration
        data_loader: Data loader
        save_folder: Folder to save checkpoints
        args: Command line arguments
        tokenizer_config: Tokenizer configuration
        device: Training device 
        sequence_length: Sequence length
        
    Returns:
        tuple: (train_module_config, trainer)
    """
    # Calculate global batch size
    global_batch_size = args.batch_size * sequence_length
    
    # Set up log file for logging messages
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    log_file = f"{save_folder}/olmo_training_{timestamp}.log"
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_name}-{timestamp}",
        config={
            "model": "OLMo-190M",
            "batch_size": args.batch_size,
            "total_steps": args.steps,
            "sequence_length": sequence_length,
            "learning_rate": optim_config.lr,
            "inference_interval": args.inference_interval,
            "initialization_method": "normal",
            "weight_decay": optim_config.weight_decay,
            "optim_betas": optim_config.betas,
        }
    )
    
    # Create TransformerTrainModuleConfig
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=global_batch_size,
        max_sequence_length=sequence_length,
        optim=optim_config,
        compile_model=False,  # Set to False for compatibility
    )

    # Build train module from the config
    train_module = train_module_config.build(model=model, device=device)
    print("Train module built successfully")
    
    # Create inference callback
    inference_callback = InferenceCallback(
        model=model,
        tokenizer_config=tokenizer_config,
        inference_interval=args.inference_interval,
        inference_prompt=args.inference_prompt,
        log_file=log_file
    )


    # Configure trainer
    trainer_config = TrainerConfig(
        save_folder=save_folder,
        save_overwrite=True,
        metrics_collect_interval=10,
        cancel_check_interval=5,
        max_duration=Duration.steps(args.steps),
        device=str(device)
    ).with_callback(
        "wandb",
        WandBCallback(
            name=args.wandb_name, 
            entity=None,  
            project=args.wandb_project,
            enabled=True,
            cancel_check_interval=10,
        )
    ).with_callback(
        "inference",
        inference_callback
    )
    

    # Build trainer
    trainer = trainer_config.build(
        train_module=train_module,
        data_loader=data_loader
    )
    print("Trainer built successfully")

    # Run training
    print(f"Starting training for {args.steps} steps...")
    trainer.fit()
    print("Training complete!")

    