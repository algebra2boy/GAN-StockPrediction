import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
data = pd.read_csv('preprocessed_stock_data.csv')
data['y'] = data['Close']

num_features = 32
x = data.iloc[:, :num_features].values
y = data.iloc[:, num_features].values

split = int(data.shape[0]* 0.8)
train_x, test_x = x[: split, :], x[split:, :]
train_y, test_y = y[: split, ], y[split: , ]

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

train_x = x_scaler.fit_transform(train_x)
test_x = x_scaler.transform(test_x)

train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
test_y = y_scaler.transform(test_y.reshape(-1, 1))

# VAE definition remains the same as before
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU()
                )
            )
            current_dim = hidden_dim
            
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        
        self.decoder_layers = nn.ModuleList()
        reversed_dims = list(reversed(hidden_dims))
        
        for i in range(len(reversed_dims) - 1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(reversed_dims[i], reversed_dims[i + 1]),
                    nn.ReLU()
                )
            )
        
        self.final_layer = nn.Sequential(
            nn.Linear(reversed_dims[-1], input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        for layer in self.decoder_layers:
            x = layer(x)
        return self.final_layer(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), z, mu, log_var

def time_series_window(x, y, window_size):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    n_samples = len(x) - window_size
    windowed_x = np.zeros((n_samples, window_size, x.shape[1]))
    windowed_y = np.zeros((n_samples, 1))
    windowed_y_gan = np.zeros((n_samples, window_size + 1, 1))
    
    for i in range(n_samples):
        windowed_x[i] = x[i:i + window_size]
        windowed_y[i] = y[i + window_size]
        windowed_y_gan[i] = y[i:i + window_size + 1]
    
    return (torch.from_numpy(windowed_x).float(),
            torch.from_numpy(windowed_y).float(),
            torch.from_numpy(windowed_y_gan).float())

def objective(trial):
    # Set device
    use_cuda = 1
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    
    # VAE hyperparameters with wider ranges for more features
    vae_params = {
        'hidden_dims': [
            trial.suggest_int('vae_hidden_1', 128, 1024),
            trial.suggest_int('vae_hidden_2', 128, 1024),
            trial.suggest_int('vae_hidden_3', 128, 1024),
            trial.suggest_int('vae_hidden_4', 32, 256)
        ],
        'latent_dim': trial.suggest_int('latent_dim', 10, 64),
        'learning_rate': trial.suggest_float('vae_lr', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('vae_batch_size', [32, 64, 128, 256, 512])
    }
    
    # Generator (GRU) hyperparameters
    gen_params = {
        'gru1_units': trial.suggest_int('g_gru1_units', 256, 2048),
        'gru2_units': trial.suggest_int('g_gru2_units', 128, 1024),
        'gru3_units': trial.suggest_int('g_gru3_units', 64, 512),
        'linear1_units': trial.suggest_int('g_linear1_units', 64, 256),
        'linear2_units': trial.suggest_int('g_linear2_units', 32, 128),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate_g', 1e-5, 1e-2, log=True)
    }
    
    # Discriminator (CNN) hyperparameters
    disc_params = {
        'conv1_filters': trial.suggest_int('d_conv1_filters', 16, 64),
        'conv2_filters': trial.suggest_int('d_conv2_filters', 32, 128),
        'conv3_filters': trial.suggest_int('d_conv3_filters', 64, 256),
        'linear1_units': trial.suggest_int('d_linear1_units', 128, 512),
        'kernel_size': trial.suggest_int('d_kernel_size', 3, 7, step=2),
        'leaky_slope': trial.suggest_float('d_leaky_slope', 0.01, 0.3),
        'dropout': trial.suggest_float('d_dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate_d', 1e-5, 1e-2, log=True)
    }

    # Train VAE
    model_vae = VAE(input_dim=num_features, 
                    hidden_dims=vae_params['hidden_dims'], 
                    latent_dim=vae_params['latent_dim']).to(device)
    
    optimizer_vae = torch.optim.Adam(model_vae.parameters(), lr=vae_params['learning_rate'])
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float()), 
                            batch_size=vae_params['batch_size'], 
                            shuffle=False)
    
    # VAE training loop
    num_epochs_vae = 300  # Adjusted for your implementation
    for epoch in range(num_epochs_vae):
        for (x, ) in train_loader:
            x = x.to(device)
            output, z, mu, logVar = model_vae(x)
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(output, x) + kl_divergence
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()
    
    # Generate VAE features
    model_vae.eval()
    with torch.no_grad():
        _, VAE_train_x, _, _ = model_vae(torch.from_numpy(train_x).float().to(device))
        _, VAE_test_x, _, _ = model_vae(torch.from_numpy(test_x).float().to(device))
    
    # Prepare combined features
    train_x_combined = np.concatenate((train_x, VAE_train_x.cpu().numpy()), axis=1)
    test_x_combined = np.concatenate((test_x, VAE_test_x.cpu().numpy()), axis=1)
    
    # Create windowed data
    train_x_slide, train_y_slide, train_y_gan = time_series_window(train_x_combined, train_y, 3)
    test_x_slide, test_y_slide, test_y_gan = time_series_window(test_x_combined, test_y, 3)

    class Generator(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.gru_1 = nn.GRU(input_size, gen_params['gru1_units'], batch_first=True)
            self.gru_2 = nn.GRU(gen_params['gru1_units'], gen_params['gru2_units'], batch_first=True)
            self.gru_3 = nn.GRU(gen_params['gru2_units'], gen_params['gru3_units'], batch_first=True)
            self.linear_1 = nn.Linear(gen_params['gru3_units'], gen_params['linear1_units'])
            self.linear_2 = nn.Linear(gen_params['linear1_units'], gen_params['linear2_units'])
            self.linear_3 = nn.Linear(gen_params['linear2_units'], 1)
            self.dropout = nn.Dropout(gen_params['dropout_rate'])

        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), gen_params['gru1_units']).to(x.device)
            out_1, _ = self.gru_1(x, h0)
            out_1 = self.dropout(out_1)
            
            h1 = torch.zeros(1, x.size(0), gen_params['gru2_units']).to(x.device)
            out_2, _ = self.gru_2(out_1, h1)
            out_2 = self.dropout(out_2)
            
            h2 = torch.zeros(1, x.size(0), gen_params['gru3_units']).to(x.device)
            out_3, _ = self.gru_3(out_2, h2)
            out_3 = self.dropout(out_3)
            
            out_4 = self.linear_1(out_3[:, -1, :])
            out_5 = self.linear_2(out_4)
            out = self.linear_3(out_5)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(4, disc_params['conv1_filters'], 
                                 kernel_size=disc_params['kernel_size'], 
                                 stride=1, padding='same')
            self.conv2 = nn.Conv1d(disc_params['conv1_filters'], 
                                 disc_params['conv2_filters'], 
                                 kernel_size=disc_params['kernel_size'], 
                                 stride=1, padding='same')
            self.conv3 = nn.Conv1d(disc_params['conv2_filters'], 
                                 disc_params['conv3_filters'], 
                                 kernel_size=disc_params['kernel_size'], 
                                 stride=1, padding='same')
            self.linear1 = nn.Linear(disc_params['conv3_filters'], disc_params['linear1_units'])
            self.linear2 = nn.Linear(disc_params['linear1_units'], disc_params['linear1_units'])
            self.linear3 = nn.Linear(disc_params['linear1_units'], 1)
            self.leaky = nn.LeakyReLU(disc_params['leaky_slope'])
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(disc_params['dropout'])

        def forward(self, x):
            conv1 = self.conv1(x)
            conv1 = self.leaky(conv1)
            conv2 = self.conv2(conv1)
            conv2 = self.leaky(conv2)
            conv3 = self.conv3(conv2)
            conv3 = self.leaky(conv3)
            flatten_x = conv3.reshape(conv3.shape[0], conv3.shape[1])
            out_1 = self.linear1(flatten_x)
            out_1 = self.leaky(out_1)
            out_1 = self.dropout(out_1)
            out_2 = self.linear2(out_1)
            out_2 = self.relu(out_2)
            out_2 = self.dropout(out_2)
            out_3 = self.linear3(out_2)
            return torch.sigmoid(out_3)

    # Initialize models
    modelG = Generator(train_x_combined.shape[1]).to(device)
    modelD = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(modelG.parameters(), lr=gen_params['learning_rate'], betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=disc_params['learning_rate'], betas=(0.0, 0.9))
    
    trainDataloader = DataLoader(TensorDataset(train_x_slide.to(device), train_y_gan.to(device)), 
                               batch_size=128, shuffle=False)
    
    # GAN training loop
    num_epochs = 165  # Adjusted for your implementation
    for epoch in range(num_epochs):
        for (x, y) in trainDataloader:
            fake_data = modelG(x)
            fake_data = torch.cat([y[:, :3, :], fake_data.reshape(-1, 1, 1)], axis=1)
            
            dis_real_output = modelD(y)
            real_labels = torch.ones_like(dis_real_output).to(device)
            lossD_real = criterion(dis_real_output, real_labels)
                        
            dis_fake_output = modelD(fake_data)
            fake_labels = torch.zeros_like(real_labels).to(device)
            lossD_fake = criterion(dis_fake_output, fake_labels)
            
            lossD = lossD_real + lossD_fake
            optimizerD.zero_grad()
            lossD.backward(retain_graph=True)
            optimizerD.step()
            
            output_fake = modelD(fake_data)
            lossG = criterion(output_fake, real_labels)
            modelG.zero_grad()
            lossG.backward()
            optimizerG.step()
    
    # Evaluation
    modelG.eval()
    with torch.no_grad():
        pred_y_test = modelG(test_x_slide.to(device))
        y_test_true = y_scaler.inverse_transform(test_y_slide)
        y_test_pred = y_scaler.inverse_transform(pred_y_test.cpu().numpy())
        
    rmse = math.sqrt(mean_squared_error(y_test_true, y_test_pred))
    return rmse

if __name__ == "__main__":

    # Load existing study instead of creating new
    study = joblib.load("fine_tuned_study.pkl")
    print("\nPrevious best value:", study.best_value)
    print("\nPrevious best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Continue optimization
    study.optimize(objective, n_trials=50)  # Run 50 more trials
    
    print("\nNew best value:", study.best_value)
    print("\nNew best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Save the updated study
    joblib.dump(study, "fine_tuned_study.pkl")

    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=50)

    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: ", trial.value)
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")

    # # Save the study
    # joblib.dump(study, "gru_cnn_study.pkl")