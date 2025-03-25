import os
import torch
import logging
import json
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from model.llm import PX1LLM
from torch.cuda.amp import autocast, GradScaler
import gc

# Get batch size from environment
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(model, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    # Training loop with memory optimization
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Generate smaller chunks of data
        for i in range(0, 100, BATCH_SIZE):
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Generate small batch of dummy data
            x = torch.randn(BATCH_SIZE, 768).to(device)
            y = torch.randn(BATCH_SIZE, 256).to(device)
            
            # Use mixed precision training
            with autocast():
                outputs = model(x)
                loss = torch.nn.functional.mse_loss(outputs, y)
            
            # Optimize with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if i % (BATCH_SIZE * 5) == 0:
                logger.info(f"Batch {i//BATCH_SIZE}, Loss: {loss.item():.4f}")
                logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            
            # Save checkpoint every 20 batches
            if i % (BATCH_SIZE * 20) == 0:
                save_checkpoint(model, epoch, i//BATCH_SIZE, loss.item())
            
            # Break if memory gets too high
            if torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                logger.warning("Memory usage too high, reducing batch")
                break

def save_checkpoint(model, epoch, batch, loss):
    checkpoint_dir = Path('/kaggle/working/checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f'model_e{epoch}_b{batch}.pt'
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def main():
    logger.info(f"Starting training with batch size {BATCH_SIZE}")
    model = PX1LLM()
    train_model(model)

if __name__ == "__main__":
    main()
