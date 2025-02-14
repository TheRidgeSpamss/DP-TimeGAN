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
from internal_networks import Embedder, Recovery, Generator, Supervisor, Discriminator
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
#import subprocess
import torch_utils as tu

class Timegan:
    def __init__(self, original_data, parameters, checkpoint_file):
        
        self.device = tu.get_device()
        #print(original_data)
        self.ori_data, self.min_val, self.max_val = tu.MinMaxScaler(original_data)
        self.ori_time, self.max_seq_len = tu.extract_time(self.ori_data)
        self.no, self.seq_len, self.z_dim = np.asarray(original_data).shape
        parameters["input_dim"] = self.z_dim
        self.parameters = parameters

        self.dataloader = DataLoader(self.ori_data, batch_size=parameters["batch_size"], shuffle=False)

        #If using Noise, define standard deviation
        self.noise_sd = parameters["noise_sd"]

        filename = checkpoint_file

        self.embedder = Embedder(parameters['input_dim'], parameters['hidden_dim'], parameters['num_layers']).to(self.device)
        self.recovery = Recovery(parameters['hidden_dim'], parameters['input_dim'], parameters['num_layers']).to(self.device)
        self.generator = Generator(parameters['input_dim'], parameters['hidden_dim'], parameters['num_layers']).to(self.device)
        self.supervisor = Supervisor(parameters['hidden_dim'], parameters['num_layers']).to(self.device)
        self.discriminator = Discriminator(parameters['hidden_dim'], parameters['num_layers']).to(self.device)

        # Create and initialize optimizer.
        self.optim_embedder = torch.optim.Adam(self.embedder.parameters(), lr=parameters["lr"])
        self.optim_recovery = torch.optim.Adam(self.recovery.parameters(), lr=parameters["lr"])
        self.optim_generator = torch.optim.Adam(self.generator.parameters(), lr=parameters["lr"])
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=parameters["lr"])
        self.optim_supervisor = torch.optim.Adam(self.supervisor.parameters(), lr=parameters["lr"])

        # Set loss function
        self.MSELoss = torch.nn.MSELoss()
        self.BCELoss = torch.nn.BCELoss()

        self.start_epoch = {'embedding':0, 'supervisor':0, 'joint':0}
        if checkpoint_file:
            self.load_checkpoint(filename)

    def gen_batch(self):
        self.Z = torch.rand(self.parameters["batch_size"], self.X.size(1), self.parameters['input_dim'], dtype=torch.float32).to(self.device)

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
        self.Z = torch.rand(self.parameters["synth_size"], self.max_seq_len, self.parameters['input_dim'], dtype=torch.float32).to(self.device)

        self.E_hat = self.generator(self.Z)
        self.H_hat = self.supervisor(self.E_hat)
        self.X_hat = self.recovery(self.H_hat)

        return self.X_hat
    
    def train_embedder(self, join_train=False):
        self.embedder.train()
        self.recovery.train()
        self.optim_embedder.zero_grad()
        self.optim_recovery.zero_grad()
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
        self.optim_generator.zero_grad()
        self.optim_supervisor.zero_grad()
        self.G_loss_S = self.MSELoss(self.H[:, 1:, :], self.H_hat_supervise[:, :-1, :])
        self.G_loss_S.backward()
        self.optim_generator.step()
        self.optim_supervisor.step()

    def train_generator(self):
        # G_solver
        self.optim_generator.zero_grad()
        self.optim_supervisor.zero_grad()
        #self.G_loss_U = self.BCELoss(self.Y_fake, torch.ones_like(self.Y_fake))
        #self.G_loss_U_e = self.BCELoss(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
        self.G_loss_U = -torch.mean(torch.log(self.Y_fake))   #, torch.ones_like(self.Y_fake))
        self.G_loss_U_e = -torch.mean(torch.log(self.Y_fake_e)) #self.BCELoss(self.Y_fake_e, torch.ones_like(self.Y_fake_e))

        #testing
        #self.G_loss_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(self.X_hat, [0])[1] + 1e-6) - torch.sqrt(
        #    torch.std(self.X, [0])[1] + 1e-6)))
        #self.G_loss_V2 = torch.mean(torch.abs((torch.mean(self.X_hat, [0])) - (torch.mean(self.X, [0]))))
        mean_X = torch.mean(self.X, dim=0)
        var_X = torch.var(self.X, dim=0, unbiased=False)
        mean_X_hat = torch.mean(self.X_hat, dim=0)
        var_X_hat = torch.var(self.X_hat, dim=0, unbiased=False)
        self.G_loss_V1 = torch.mean(torch.abs(torch.sqrt(var_X_hat + 1e-6) - torch.sqrt(var_X + 1e-6)))
        self.G_loss_V2 = torch.mean(torch.abs(mean_X_hat - mean_X))
        # end test of olf regiment


        self.G_loss_V = self.G_loss_V1 + self.G_loss_V2
        #self.G_loss_S = self.MSELoss(self.H_hat_supervise[:, :-1, :], self.H[:, 1:, :])
        self.G_loss = self.G_loss_U + \
                      self.parameters["gamma"] * self.G_loss_U_e + \
                      self.G_loss_V * 100 #+ \
                      #torch.sqrt(self.G_loss_S) * 100

        self.G_loss.backward(retain_graph=True)

        self.optim_generator.step()
        self.optim_supervisor.step()


    def train_discriminator(self):
        # D_solver
        self.discriminator.train()
        self.optim_discriminator.zero_grad(set_to_none=True)
        
        self.D_loss_real = self.BCELoss(self.noisyY_real, torch.ones_like(self.Y_real))
        self.D_loss_fake = self.BCELoss(self.noisyY_fake, torch.zeros_like(self.Y_fake))
        self.D_loss_fake_e = self.BCELoss(self.noisyY_fake_e, torch.zeros_like(self.Y_fake_e))
        
        self.D_loss = self.D_loss_real + \
                      self.D_loss_fake + \
                      self.parameters["gamma"] * self.D_loss_fake_e
        # Train discriminator (only when the discriminator does not work well)
        if self.D_loss > 0.15:
            self.D_loss.backward()
            self.optim_discriminator.step()


    def get_checkpoint_path(self, filename):
        script_dir = os.path.dirname(__file__)
        checkpoint_path = os.path.join(script_dir, 'Checkpoints', filename)
        return checkpoint_path
    

    def load_checkpoint(self, filename):
        filename = filename+'.pth'
        checkpoint_path = self.get_checkpoint_path(filename)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            self.embedder.load_state_dict(checkpoint['model_state_dict']['embedder'])
            self.recovery.load_state_dict(checkpoint['model_state_dict']['recovery'])
            self.generator.load_state_dict(checkpoint['model_state_dict']['generator'])
            self.supervisor.load_state_dict(checkpoint['model_state_dict']['supervisor'])
            self.discriminator.load_state_dict(checkpoint['model_state_dict']['discriminator'])
            #self.er_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['er_optimizer'])
            self.optim_generator.load_state_dict(checkpoint['optimizer_state_dict']['generator_optimizer'])
            self.optim_supervisor.load_state_dict(checkpoint['optimizer_state_dict']['supervisor_optimizer'])
            self.optim_discriminator.load_state_dict(checkpoint['optimizer_state_dict']['discriminator_optimizer'])
            self.optim_embedder.load_state_dict(checkpoint['optimizer_state_dict']['embedder_optimizer'])
            self.optim_recovery.load_state_dict(checkpoint['optimizer_state_dict']['recovery_optimizer'])
            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"Resumed training from epoch {self.start_epoch}")
        else:
            print('No Checkpoint under given filename, beginning training.')

#Test, currently not including losses in checkpoints (they are not loaded anywhere)   
#VERIFY THAT THIS WORKS
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
        checkpt_dir = os.path.join(os.path.dirname(__file__), 'Checkpoints')
        if not os.path.exists(checkpt_dir):
            os.makedirs(checkpt_dir)

        #checkpoint_path = os.path.join(checkpoint_dir, filename)

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
