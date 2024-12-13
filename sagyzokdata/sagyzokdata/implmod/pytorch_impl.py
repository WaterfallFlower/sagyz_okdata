import torch
import torch.nn as nn

class TransactionAutoencoder(nn.Module):
    def _init_(self, input_dim, hidden_dim=64):
        super(TransactionAutoencoder, self)._init_()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

def train_autoencoder(data, epochs=10, lr=1e-3):
    input_dim = data.shape[1]
    model = TransactionAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    tensor_data = torch.tensor(data.values, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon = model(tensor_data)
        loss = criterion(recon, tensor_data)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def detect_anomalies(model, data, threshold=0.1):
    model.eval()
    with torch.no_grad():
        tensor_data = torch.tensor(data.values, dtype=torch.float32)
        recon = model(tensor_data)
        loss = (recon - tensor_data)**2
        mse = loss.mean(dim=1)
    anomalies = mse > threshold
    return anomalies