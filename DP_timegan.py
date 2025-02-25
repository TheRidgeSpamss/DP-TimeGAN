'''
Based on the TimeGAN architecture: Jinsung Yoon, Daniel Jarrett, 
Mihaela van der Schaar, "Time-series Generative Adversarial Networks," Neural Information Processing Systems (NeurIPS), 2019.

Pytorch Implementation heavily influenced by the following github repository 
Title: TimeGAN_PytorchRebuild
From User: AlanDongMu
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from internal_networks import Embedder, Recovery, Generator, Supervisor, DP_Discriminator
import os
from opacus import PrivacyEngine
import torch_utils as tu
from torch.utils.data import Dataset, DataLoader
from opacus.validators import ModuleValidator

class TimeSeriesDataset(Dataset):
    def __init__(self, data, times):
        self.data = data
        self.times = times

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.times[idx]

class DP_Timegan:
    def __init__(self, original_data, parameters, checkpoint_file=None):
        self.device = tu.get_device()
        self.ori_data, self.min_val, self.max_val = tu.MinMaxScaler(original_data)
        self.ori_time, self.max_seq_len = tu.extract_time(self.ori_data)
        self.no, self.seq_len, self.z_dim = np.asarray(original_data).shape
        parameters["input_dim"] = self.z_dim
        self.parameters = parameters

        # Create the custom dataset and DataLoader
        self.dataset = TimeSeriesDataset(self.ori_data, self.ori_time)
        self.dataloader = DataLoader(self.dataset, batch_size=self.parameters["batch_size"], shuffle=False)
        self.data_iter = iter(self.dataloader)

        # If using Noise, define standard deviation
        self.noise_sd = parameters["noise_sd"]

        filename = checkpoint_file

        self.embedder = Embedder(self.parameters['input_dim'], self.parameters['hidden_dim'], self.parameters['num_layers']).to(self.device)
        self.recovery = Recovery(self.parameters['hidden_dim'], self.parameters['input_dim'], self.parameters['num_layers']).to(self.device)
        self.generator = Generator(self.parameters['input_dim'], self.parameters['hidden_dim'], self.parameters['num_layers']).to(self.device)
        self.supervisor = Supervisor(self.parameters['hidden_dim'], self.parameters['num_layers']).to(self.device)
        self.discriminator = DP_Discriminator(self.parameters['hidden_dim'], self.parameters['num_layers']).to(self.device)

        # Create and initialize optimizer.
        self.optim_embedder = torch.optim.Adam(self.embedder.parameters(), lr=self.parameters["lr"])
        self.optim_recovery = torch.optim.Adam(self.recovery.parameters(), lr=self.parameters["lr"])
        self.optim_generator = torch.optim.Adam(self.generator.parameters(), lr=self.parameters["lr"])
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.parameters["lr"])
        self.optim_supervisor = torch.optim.Adam(self.supervisor.parameters(), lr=self.parameters["lr"])

        # Set loss function
        self.MSELoss = torch.nn.MSELoss()
        self.BCELoss = torch.nn.BCELoss()

        # Initialize PrivacyEngine for the discriminator
        self.privacy_engine = PrivacyEngine(accountant='rdp')
        print('Compatibility:', self.privacy_engine.is_compatible(module=self.discriminator, optimizer=self.optim_discriminator, data_loader=self.dataloader))  #check compatibility
        self.epsilon = parameters["eps"]

        self.start_epoch = {'embedding': 0, 'supervisor': 0, 'joint': 0}
        if checkpoint_file:
            self.load_checkpoint(filename)

    def gen_batch(self):
        # Get a batch from the DataLoader
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        
        batch_data, batch_time = batch
        self.X = nn.utils.rnn.pad_sequence([data.clone().detach().float() for data in batch_data], batch_first=True).to(self.device)
        self.T = batch_time.clone().detach().long().to(self.device)  # Ensure integer type

        self.batch_size = self.X.size(0)

        # Random vector generation
        self.Z = torch.rand(self.batch_size, self.X.size(1), self.parameters['input_dim'], dtype=torch.float32).to(self.device)

    def forward_embedder_recovery(self):
        """Forward pass for embedding and recovery networks."""
        self.H = self.embedder(self.X)
        self.X_tilde = self.recovery(self.H)

    def forward_supervisor(self):
        """Forward pass for the supervisor network."""
        self.H = self.embedder(self.X)
        self.H_hat_supervise = self.supervisor(self.H)

    def forward_generator_discriminator(self):
        """Forward pass for generator and discriminator networks."""
        self.H = self.embedder(self.X)
        self.E_hat = self.generator(self.Z)
        self.H_hat = self.supervisor(self.E_hat)
        self.X_hat = self.recovery(self.H_hat)

        self.Y_real = self.discriminator(self.H)
        self.Y_fake = self.discriminator(self.H_hat)
        self.Y_fake_e = self.discriminator(self.E_hat)

        self.noisyY_real = self.discriminator(self.H + torch.normal(mean=0, std=self.noise_sd, size=self.H.size()).to(self.device))
        self.noisyY_fake = self.discriminator(self.H_hat + torch.normal(mean=0, std=self.noise_sd, size=self.H_hat.size()).to(self.device))
        self.noisyY_fake_e = self.discriminator(self.E_hat + torch.normal(mean=0, std=self.noise_sd, size=self.E_hat.size()).to(self.device))

    def gen_synth_data(self):
        self.Z = torch.randn(self.parameters["synth_size"], self.max_seq_len, self.parameters['input_dim']).to(self.device)
        self.E_hat = self.generator(self.Z)
        self.H_hat = self.supervisor(self.E_hat)
        self.X_hat = self.recovery(self.H_hat)

        return self.X_hat
    
    def train_embedder(self, join_train=False):
        self.embedder.train()
        self.recovery.train()
        self.optim_embedder.zero_grad(set_to_none=True) 
        self.optim_recovery.zero_grad(set_to_none=True)
        self.E_loss_T0 = self.MSELoss(self.X, self.X_tilde)
        self.E_loss0 = 10 * torch.sqrt(self.E_loss_T0)

        #self.G_loss_S = self.MSELoss(self.H[:, 1:, :], self.H_hat_supervise[:, :-1, :])
        self.E_loss = self.E_loss0 #+ 0.1 * self.G_loss_S
        self.E_loss.backward()
        self.optim_embedder.step()
        self.optim_recovery.step()

    def train_supervisor(self):
        # GS_solver
        self.generator.train()
        self.supervisor.train()
        self.optim_generator.zero_grad(set_to_none=True)
        self.optim_supervisor.zero_grad(set_to_none=True)
        self.G_loss_S = self.MSELoss(self.H[:, 1:, :], self.H_hat_supervise[:, :-1, :])
        self.G_loss_S.backward()
        self.optim_generator.step()
        self.optim_supervisor.step()

    def train_joint(self, X_batch):
        '''Note that opacus seem to require that forward passes for obtaining data
        are re-evalated after any gradient steps have been taken.
        '''
        X_batch = X_batch.float().to(self.device)
        batch_size = X_batch.size(0)
        self.H = self.embedder(X_batch)
        H_batch_size = self.H.size(0)
        self.Z = torch.randn(batch_size, X_batch.size(1), self.parameters['input_dim']).to(self.device)

        # 1. Train DP_Discriminator
        self.optim_discriminator.zero_grad(set_to_none=True)

        #Making Fake Data and Labels for DP_Discriminator
        raw_fake_data = self.generator(self.Z)
        supervised_fake_data = self.supervisor(raw_fake_data)
        self.X_hat = self.recovery(supervised_fake_data)        #recovering for statistics loss

        #Discriminating EACH time step in the sequence
        real_labels = torch.ones(H_batch_size, self.parameters["hidden_dim"]).to(self.device)
        fake_labels = torch.zeros(H_batch_size, self.parameters["hidden_dim"]).to(self.device)

        #Real, Fake, and Raw Fake Data into Discriminator, compute losses
        # real_output = self.discriminator(self.H)
        # d_real_loss = self.BCELoss(real_output, real_labels)

        # fake_output = self.discriminator(supervised_fake_data.detach())
        # d_fake_loss = self.BCELoss(fake_output, fake_labels)

        # raw_fake_output = self.discriminator(raw_fake_data.detach())
        # d_raw_fake_loss = self.BCELoss(raw_fake_output, fake_labels)
        real_output = self.discriminator(self.H)
        fake_output = self.discriminator(supervised_fake_data.detach())
        raw_fake_output = self.discriminator(raw_fake_data.detach())

        d_real_loss = self.BCELoss(real_output, torch.ones_like(real_output))
        d_fake_loss = self.BCELoss(fake_output, torch.zeros_like(fake_output))
        d_raw_fake_loss = self.BCELoss(raw_fake_output, torch.zeros_like(raw_fake_output))

        #Total DP_Discriminator Loss, backpropagate, step, zero gradients for opacus
        self.D_loss = d_real_loss + d_fake_loss + self.parameters["gamma"] * d_raw_fake_loss
        if self.D_loss > 0.15:
            self.D_loss.backward()
            self.optim_discriminator.step()
        self.optim_discriminator.zero_grad(set_to_none=True)

        #epoch_d_loss += d_loss.item()

        # 2. Train Generator
        self.optim_generator.zero_grad()
        self.optim_supervisor.zero_grad()

        #Generator Supervised Loss
        fake_output = self.discriminator(supervised_fake_data)
        self.G_loss_U = self.BCELoss(fake_output, torch.ones_like(fake_output))

        #Generator Unsupservised Loss
        raw_fake_output = self.discriminator(raw_fake_data)
        self.G_loss_U_e = self.BCELoss(raw_fake_output, torch.ones_like(raw_fake_output))

        #Generated Data Descriptive Temporal Statistics Loss
        mean_X = torch.mean(X_batch, dim=0)
        var_X = torch.var(X_batch, dim=0, unbiased=False)
        mean_X_hat = torch.mean(self.X_hat, dim=0)
        var_X_hat = torch.var(self.X_hat, dim=0, unbiased=False)
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(var_X_hat + 1e-6) - torch.sqrt(var_X + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs(mean_X_hat - mean_X))
        self.G_loss_V = G_loss_V1 + G_loss_V2
        
        #Continuing to Train Supervisor
        #self.H = self.embedder(X_batch)
        #supervised_real_latent_data = self.supervisor(self.H)
        #G_loss_S = self.MSELoss(supervised_real_latent_data[:, :-1, :], self.H[:, 1:, :])

        self.G_loss = self.G_loss_U + self.G_loss_U_e + 100*self.G_loss_V #+ 10*G_loss_S

        self.G_loss.backward()#retain_graph=True)
        self.optim_generator.step()  # Move this step immediately after the backward pass
    
    def reset_gradients(self):
        self.optim_embedder.zero_grad(set_to_none=True)
        self.optim_recovery.zero_grad(set_to_none=True)
        self.optim_generator.zero_grad(set_to_none=True)
        self.optim_supervisor.zero_grad(set_to_none=True)
        self.optim_discriminator.zero_grad(set_to_none=True)

    def initialize_privacy_engine(self):
        self.discriminator, self.optim_discriminator, self.dataloader = self.privacy_engine.make_private_with_epsilon(
            module=self.discriminator, 
            optimizer=self.optim_discriminator, 
            data_loader=self.dataloader,
            #noise_multiplier=1.3, 
            max_grad_norm=1.0,
            target_epsilon=self.epsilon,
            target_delta=1e-5,
            epochs=self.parameters["iterations"],
        )


    def reinitialize_discriminator(self):
        self.discriminator = DP_Discriminator(self.parameters['hidden_dim'], self.parameters['num_layers']).to(self.device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.parameters["lr"])


    def get_checkpoint_path(self, filename):
        script_dir = os.path.dirname(__file__)
        checkpoint_path = os.path.join(script_dir, 'Checkpoints', filename)
        return checkpoint_path
    

    def load_checkpoint(self, filename):
        #self.reinitialize_privacy_engine()      #need this to load the correct version of discrim - opacus version
        #^^^When integrating this file with the other timegan, will need to check it opt.use_dp is on here^^^
    
        filename = filename+'.pth'
        checkpoint_path = self.get_checkpoint_path(filename)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            self.embedder.load_state_dict(checkpoint['model_state_dict']['embedder'])
            self.recovery.load_state_dict(checkpoint['model_state_dict']['recovery'])
            self.generator.load_state_dict(checkpoint['model_state_dict']['generator'])
            self.supervisor.load_state_dict(checkpoint['model_state_dict']['supervisor'])
            #self.discriminator.load_state_dict(checkpoint['model_state_dict']['discriminator'])

            if self.start_epoch['joint'] > 0:
                self.initialize_privacy_engine()
                self.discriminator.load_state_dict(checkpoint['model_state_dict']['discriminator'])
                self.optim_discriminator.load_state_dict(checkpoint['optimizer_state_dict']['discriminator_optimizer'])
            else:
                self.discriminator.load_state_dict(checkpoint['model_state_dict']['discriminator'])
                self.optim_discriminator.load_state_dict(checkpoint['optimizer_state_dict']['discriminator_optimizer'])

            self.optim_generator.load_state_dict(checkpoint['optimizer_state_dict']['generator_optimizer'])
            self.optim_supervisor.load_state_dict(checkpoint['optimizer_state_dict']['supervisor_optimizer'])
            #self.optim_discriminator.load_state_dict(checkpoint['optimizer_state_dict']['discriminator_optimizer'])
            self.optim_embedder.load_state_dict(checkpoint['optimizer_state_dict']['embedder_optimizer'])
            self.optim_recovery.load_state_dict(checkpoint['optimizer_state_dict']['recovery_optimizer'])
            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"Resumed training from epoch {self.start_epoch}")
        else:
            print('No Checkpoint under given filename, beginning training.')

    def save_checkpoint(self, epoch, filename):
        """Saves checkpoints for each internal network.

        Inputs:
        - epoch: a dict with the current epoch number of each training phase 
        ie. {'embedding':__, 'supervisor':__, 'joint':__}
        - filename: The file name to save the checkpoint under"""

        #Create the Checkpoints directory if it does not exist
        filename = filename + '.pth'
        checkpoint_path = self.get_checkpoint_path(filename)
        #checkpoint_dir = os.path.join(checkpoint_path, '..')
        checkpt_dir = os.path.join(os.path.dirname(__file__),'Checkpoints')
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': {
                'embedder': self.embedder.state_dict(),
                'recovery': self.recovery.state_dict(),
                'generator': self.generator.state_dict(),
                'supervisor': self.supervisor.state_dict(),
                'discriminator': self.discriminator.state_dict()
            },
            'optimizer_state_dict': {
                'embedder_optimizer': self.optim_embedder.state_dict(),
                'recovery_optimizer': self.optim_recovery.state_dict(),
                'generator_optimizer': self.optim_generator.state_dict(),
                'supervisor_optimizer': self.optim_supervisor.state_dict(),
                'discriminator_optimizer': self.optim_discriminator.state_dict()
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")
