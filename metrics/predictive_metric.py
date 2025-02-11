import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch_utils as tu
import subprocess
import gc

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, t):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, t, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        y_hat_logit = self.fc(output)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat

class TimeSeriesDataset(Dataset):
    def __init__(self, data, time, dim):
        self.data = data
        self.time = time
        self.dim = dim

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx][:-1, :(self.dim-1)], dtype=torch.float32)
        T = self.time[idx] - 1
        Y = torch.tensor(self.data[idx][1:, (self.dim-1)].reshape(-1, 1), dtype=torch.float32)
        return X, T, Y

def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.
    
    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      
    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    
    # Basic Parameters
    device = tu.get_device()
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    no, seq_len, dim = ori_data.shape
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = tu.extract_time(ori_data)
    generated_time, generated_max_seq_len = tu.extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
    
    # Build a post-hoc RNN predictive network 
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128
    
    # Initialize the model, loss function, and optimizer
    model = Predictor(input_dim=dim-1, hidden_dim=hidden_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())
    
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = tu.train_test_divide(ori_data, generated_data, ori_time, generated_time, train_rate=0.8)

    dataset = TimeSeriesDataset(train_x_hat, train_t_hat, dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Training using Synthetic dataset
    model.train()
    for itt in range(iterations):
        for (X_mb, T_mb, Y_mb) in dataloader:
            # idx = np.random.permutation(len(generated_data))
            # train_idx = idx[:batch_size]     
            
            # X_mb = [torch.tensor(generated_data[i][:-1, :(dim-1)], dtype=torch.float32).to(device) for i in train_idx]
            # T_mb = [generated_time[i] - 1 for i in train_idx]
            # Y_mb = [torch.tensor(generated_data[i][1:, (dim-1)].reshape(-1, 1), dtype=torch.float32).to(device) for i in train_idx]

            X_mb = X_mb.to(device)
            Y_mb = Y_mb.to(device)
            
            X_mb = nn.utils.rnn.pad_sequence(X_mb, batch_first=True)
            Y_mb = nn.utils.rnn.pad_sequence(Y_mb, batch_first=True)
            
            optimizer.zero_grad()
            y_pred = model(X_mb, T_mb)
            loss = criterion(y_pred, Y_mb)
            loss.backward()
            optimizer.step()

        if (itt + 1) % 500 == 0:
            print(f'iteration: {itt + 1}/{iterations}, Loss: {loss.item()}')
    
    # Test the trained model on the original data
    model.eval()
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]
    
    X_mb = [torch.tensor(ori_data[i][:-1, :(dim-1)], dtype=torch.float32).to(device) for i in train_idx]
    T_mb = [ori_time[i] - 1 for i in train_idx]
    Y_mb = [torch.tensor(ori_data[i][1:, (dim-1)].reshape(-1, 1), dtype=torch.float32).to(device) for i in train_idx]
    
    X_mb = nn.utils.rnn.pad_sequence(X_mb, batch_first=True)
    Y_mb = nn.utils.rnn.pad_sequence(Y_mb, batch_first=True)

    with torch.inference_mode():
        pred_Y_curr = model(X_mb, T_mb)
    
    predictive_score = torch.mean(torch.abs(Y_mb - pred_Y_curr))

    #new
    predictive_score = predictive_score.cpu().numpy()
    
    #predictive_score = MAE_temp / no
    del dataset, dataloader, model, optimizer, X_mb, T_mb, Y_mb
    gc.collect()
    return predictive_score
