import torch
from tqdm import tqdm
import numpy as np

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    Trains a PyTorch model with progress bar and loss display.
    """
    model.train()
    
    # Move model to device
    model = model.to(device)
    
    # Initialize tqdm progress bar
    pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
    for epoch in pbar:
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # Move data to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            epoch_loss += loss.item() * batch_x.size(0)
        
        # Calculate average loss for the epoch
        epoch_loss = epoch_loss / len(train_loader.dataset)
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{epoch_loss:.6f}'})
    
    return model