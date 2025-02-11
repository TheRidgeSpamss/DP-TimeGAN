import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryRecall, BinarySpecificity, MulticlassAUROC, MulticlassAccuracy, MulticlassRecall, MulticlassSpecificity
from sklearn.metrics import roc_auc_score
import os
import sys
import gc

sys.path.append(os.path.abspath(".."))
import torch_utils as tu

class ConvGRU(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.input_dim = params["input_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.output_dim = params["output_dim"]
        self.num_layers = params["num_layers"]
        self.dropout_frac = params["dropout"]
        self.max_pool_padding = params["max_pool_padding"]
        self.seq_len = params["seq_len"]
        self.fc_layer_dim = params["fc_layer_dim"]

        self.dropout = nn.Dropout(p=self.dropout_frac)
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Conv1d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(p=self.dropout_frac),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=self.max_pool_padding),
        )

        self.flatten = nn.Flatten()

        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim)
        )

        self.output_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = x.transpose(1, 2)
        _, hn = self.gru(x)
        hn = hn[-1]

        x = self.dropout(hn)

        x = self.fc(x)

        x = self.output_fc(x)
        return x


class PatientDataset(Dataset):
    def __init__(self, data, labels, device, class_count):
        self.data = torch.from_numpy(data).to(device).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(device)
        self.labels = self.labels.to(torch.long) if class_count > 2 else self.labels.to(torch.float32)
        # self.labels = torch.from_numpy(labels).to(device).to(torch.float32) #for bcewithlogitsloss

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_downstream_model(data, labels, parameters, device):
    epoch_count = 100
    samples, seq_len, features = data.shape
    class_count = len(np.unique(labels))
    batch_size = 64
    
    model_params = {}
    model_params["input_dim"] = features
    model_params["hidden_dim"] = 24
    model_params["output_dim"] = class_count if class_count > 2 else 1
    model_params["num_layers"] = 3
    model_params["dropout"] = 0.6
    model_params["max_pool_padding"] = 1 if (seq_len % 2 == 1) else 0
    model_params["seq_len"] = seq_len
    model_params["fc_layer_dim"] = (((model_params["hidden_dim"] * (model_params["seq_len"] + 1)) // 2)) if (seq_len % 2 == 1) else ((model_params["hidden_dim"] * model_params["seq_len"]) // 2)

    model = ConvGRU(model_params).to(device)

    dataset = PatientDataset(data, labels, device, class_count)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss() if class_count > 2 else nn.BCEWithLogitsLoss()

    model.train()
    print("Start training")
    for epoch in range(epoch_count):
        for x_mb, y_mb in dataloader:
            optimizer.zero_grad()
            y_pred_mb = model(x_mb)
            if class_count <= 2:
                y_pred_mb = y_pred_mb.squeeze(1)
            loss = criterion(y_pred_mb, y_mb)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1}/{epoch_count}, Loss: {loss.item()}")
    
    return model

def downstream_model_metrics(data, labels, parameters):

    device = tu.get_device()

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=parameters["seed"])
    model = train_downstream_model(x_train, y_train, parameters, device)

    model.eval()
    with torch.inference_mode():
        x_test = torch.from_numpy(x_test).to(device).to(torch.float32)
        y_test = torch.from_numpy(y_test).to(device).to(torch.long)
        y_test_pred = model(x_test)

        classes = torch.unique(y_test).cpu().numpy()
        class_count = y_test_pred.shape[1]

        if class_count > 2:
            torch_accuracy_metric = MulticlassAccuracy(num_classes=class_count).to(device)
        else:
            y_test_pred = torch.sigmoid(y_test_pred) #for binary
            y_test_pred = y_test_pred.squeeze(1)
            torch_accuracy_metric = BinaryAccuracy().to(device)

        torch_accuracy = torch_accuracy_metric(y_test_pred, y_test)
        torch_accuracy = torch_accuracy.cpu().numpy().item()


    del model, x_train, x_test, y_train, y_test, data, labels
    gc.collect()
    print(f"Accuracy: {torch_accuracy}")

    return torch_accuracy

if __name__ == "__main__":
    parameters = {"hidden_dim": 24, "num_layers": 3, "seed": 42, "lr": 1e-3, "batch_size": 128}
    downstream_model_metrics("x", "y", parameters)
