import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from DP_timegan import TimeSeriesDataset
import torch_utils as tu
import subprocess
import gc

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, t):
        t_cpu = t.clone().detach().cpu()  # Ensure lengths are on CPU
        packed_input = nn.utils.rnn.pack_padded_sequence(x, t_cpu, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        y_hat_logit = self.fc(output[:, -1, :])  # Get the last output state
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat

def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data
    
    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      
    Returns:
      - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    device = tu.get_device()
    
    # Basic Parameters
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    no, seq_len, dim = ori_data.shape    
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = tu.extract_time(ori_data)
    generated_time, generated_max_seq_len = tu.extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
    
    # Build a post-hoc RNN discriminator network 
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128
    
    # Initialize the model, loss function, and optimizer
    model = Discriminator(input_dim=dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        tu.train_test_divide(ori_data, generated_data, ori_time, generated_time)

    real_dataset = TimeSeriesDataset(train_x, train_t)
    synthetic_dataset = TimeSeriesDataset(train_x_hat, train_t_hat)
    real_dataloader = DataLoader(real_dataset, batch_size=128, shuffle=True)
    synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=128, shuffle=True)
    
    # Training step
    model.train()
    for itt in range(iterations):
        for (X_mb, T_mb), (X_hat_mb, T_hat_mb) in zip(real_dataloader, synthetic_dataloader):
            X_mb, X_hat_mb = X_mb.to(device).to(torch.float32), X_hat_mb.to(device).to(torch.float32)
            
            # Train discriminator
            optimizer.zero_grad()
            
            y_logit_real, y_pred_real = model(X_mb, T_mb)
            y_logit_fake, y_pred_fake = model(X_hat_mb, T_hat_mb)
            
            d_loss_real = criterion(y_logit_real, torch.ones_like(y_logit_real))
            d_loss_fake = criterion(y_logit_fake, torch.zeros_like(y_logit_fake))
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer.step()

        if (itt + 1) % 500 == 0:
            print(f'iteration: {itt + 1}/{iterations}, Loss: {d_loss.item()}')
    
    # Test the performance on the testing set
    model.eval()
    test_x, test_x_hat = torch.tensor(test_x, dtype=torch.float32).to(device), torch.tensor(test_x_hat, dtype=torch.float32).to(device)
    test_t, test_t_hat = torch.tensor(test_t, dtype=torch.long), torch.tensor(test_t_hat, dtype=torch.long)  # Keep lengths on CPU
    
    with torch.inference_mode():
        y_logit_real_curr, y_pred_real_curr = model(test_x, test_t)
        y_logit_fake_curr, y_pred_fake_curr = model(test_x_hat, test_t_hat)
    
    y_pred_final = torch.cat((y_pred_real_curr, y_pred_fake_curr), axis=0).cpu().numpy()
    y_label_final = np.concatenate((np.ones(len(y_pred_real_curr)), np.zeros(len(y_pred_fake_curr))), axis=0)
    
    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    
    del y_logit_fake_curr, y_logit_real_curr, y_pred_fake_curr, y_pred_real_curr, real_dataset, real_dataloader, synthetic_dataset, synthetic_dataloader, model, optimizer
    del train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat
    gc.collect()
    return discriminative_score