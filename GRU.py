import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.func import jacfwd

class DatasetW(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]  
        y = self.data[idx+1:idx+self.seq_length+1]  
        return x, y
    
def normalize_sincos(output):
    sin_pred1, cos_pred1 = output[:, :, 0], output[:, :, 1]
    sin_pred2, cos_pred2 = output[:, :, 2], output[:, :, 3]
    
    norm1 = torch.sqrt(sin_pred1**2 + cos_pred1**2)
    norm2 = torch.sqrt(sin_pred2**2 + cos_pred2**2)
    
    normalized_output = output.clone()
    normalized_output[:, :, 0] = sin_pred1 / norm1
    normalized_output[:, :, 1] = cos_pred1 / norm1
    normalized_output[:, :, 2] = sin_pred2 / norm2
    normalized_output[:, :, 3] = cos_pred2 / norm2
    
    return normalized_output

class GRU_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.Wz = nn.Parameter(torch.empty(hidden_size, input_size + hidden_size))
        self.Wr = nn.Parameter(torch.empty(hidden_size, input_size + hidden_size)) 
        self.Wh = nn.Parameter(torch.empty(hidden_size, input_size + hidden_size))
        
        nn.init.xavier_uniform_(self.Wz)
        nn.init.xavier_uniform_(self.Wr)
        nn.init.xavier_uniform_(self.Wh)
        
        self.bz = nn.Parameter(torch.ones(1, hidden_size) * 1.0)
        self.br = nn.Parameter(torch.zeros(1, hidden_size))
        self.bh = nn.Parameter(torch.zeros(1, hidden_size))
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        for m in self.linear:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    
    def forward(self, x, h_prev):
        h_prev_dropped = h_prev
        combined = torch.cat((x, h_prev_dropped), dim=1)
        
        z_t = torch.sigmoid(combined @ self.Wz.t() + self.bz)
        r_t = torch.sigmoid(combined @ self.Wr.t() + self.br)
        
        gate_input = torch.cat((r_t * h_prev_dropped, x), dim=1)
        h_hat = torch.tanh(gate_input @ self.Wh.t() + self.bh)
        
        h_t = (1 - z_t) * h_prev + z_t * h_hat
        
        output = self.linear(h_t)
        
        return output, h_t

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = GRU_Cell(input_size, hidden_size, output_size)

    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.shape
        h_t = h_0 if h_0 is not None else torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            out, h_t = self.rnn_cell(x[:, t, :], h_t)
            outputs.append(out)
        return torch.stack(outputs, dim=1), h_t

    def autonomous_step(self, h_prev, normalize_sincos):
        input_t = self.rnn_cell.linear(h_prev)
        if normalize_sincos:
            sin1, cos1 = input_t[:, 0], input_t[:, 1]
            norm1 = torch.sqrt(sin1**2 + cos1**2)
            input_t[:, 0] = sin1 / (norm1 + 1e-10)
            input_t[:, 1] = cos1 / (norm1 + 1e-10)
            
            sin2, cos2 = input_t[:, 2], input_t[:, 3]
            norm2 = torch.sqrt(sin2**2 + cos2**2)
            input_t[:, 2] = sin2 / (norm2 + 1e-10)
            input_t[:, 3] = cos2 / (norm2 + 1e-10)
        else:
            input_t = input_t
        _, h_next = self.rnn_cell(input_t, h_prev)
        return h_next   
     
    def lyapunov_exponents(self, initial_condition, n_steps, dt=1.0, num_lyaps=None,
                           warmup_steps=100, norm_freq=10, epsilon=1e-15, normalize_sincos = False):
        self.eval()
        device = next(self.parameters()).device
        
        h = torch.zeros(1, self.hidden_size, device=device, dtype=torch.float32)
        if isinstance(initial_condition, np.ndarray):
            initial_condition = torch.tensor(initial_condition, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            for t in range(min(warmup_steps, len(initial_condition))):
                input_t = initial_condition[t:t+1].view(1, -1)
                if normalize_sincos:
                    sin1, cos1 = input_t[:, 0], input_t[:, 1]
                    norm1 = torch.sqrt(sin1**2 + cos1**2)
                    input_t[:, 0] = sin1 / (norm1 + 1e-10)
                    input_t[:, 1] = cos1 / (norm1 + 1e-10)
                    
                    sin2, cos2 = input_t[:, 2], input_t[:, 3]
                    norm2 = torch.sqrt(sin2**2 + cos2**2)
                    input_t[:, 2] = sin2 / (norm2 + 1e-10)
                    input_t[:, 3] = cos2 / (norm2 + 1e-10)
                else:
                    input_t = input_t
                _, h = self.rnn_cell(input_t, h)
        
        delta = np.random.randn(self.hidden_size, num_lyaps)
        delta, _ = np.linalg.qr(delta, mode='reduced')
        delta = torch.tensor(delta, dtype=torch.float32, device=device)
        
        lyap_sums = np.zeros(num_lyaps, dtype=np.float32)
        count = 0
        
        for step in range(n_steps):
            h_detached = h.detach().requires_grad_(True)
            try:
                jac = jacfwd(self.autonomous_step)(h_detached, normalize_sincos).squeeze()
            except Exception as e:
                print(f"error in Jacobian: {e}")
                continue
            
            with torch.no_grad():
                J_delta = jac @ delta
                
                if (step + 1) % norm_freq == 0:
                    J_delta = J_delta + epsilon * torch.randn_like(J_delta)
                    Q, R = torch.linalg.qr(J_delta, mode='reduced')
                    diag_R = torch.diagonal(R)
                    signs = torch.sign(diag_R)
                    signs[signs == 0] = 1
                    Q = Q * signs.unsqueeze(0)
                    diag_R = diag_R * signs
                    abs_diag_R = torch.abs(diag_R)
                    
                    if (abs_diag_R < epsilon).any():
                        abs_diag_R = torch.clamp(abs_diag_R, min=epsilon)
                    
                    lyap_sums += torch.log(abs_diag_R).cpu().numpy()
                    count += 1
                    delta = Q
                else:
                    delta = J_delta
            
            with torch.no_grad():
                h = self.autonomous_step(h.detach(), normalize_sincos)
                
        total_time = count * norm_freq * dt
        lyap_exponents = lyap_sums / total_time
        return lyap_exponents
    
def predict_future(model, seed_data, future_steps, device, normalize_sincos_values = False):
        model.eval()
        d = seed_data.shape[1]
        if isinstance(seed_data, np.ndarray):
            seed_data = torch.tensor(seed_data, dtype=torch.float32).to(device)
        print(seed_data.shape)
        x = seed_data.view(1, -1, d)
        print(x.shape)
        with torch.no_grad():
            output, hidden = model(x[:, 0, :].unsqueeze(1))
            last_state = x[:, 0, :]
            for i in range(150):
                delta, hidden = model.rnn_cell(x[:, i, :], hidden)
                last_state = x[:, i, :]
                
            predictions = [last_state.unsqueeze(1)]
            
            for _ in range(future_steps - 1):
                out, hidden = model.rnn_cell(last_state, hidden)
                if normalize_sincos_values:
                    sin1, cos1 = out[:, 0], out[:, 1]
                    norm1 = torch.sqrt(sin1**2 + cos1**2)
                    out[:, 0] = sin1 / (norm1 + 1e-10)
                    out[:, 1] = cos1 / (norm1 + 1e-10)
                    
                    sin2, cos2 = out[:, 2], out[:, 3]
                    norm2 = torch.sqrt(sin2**2 + cos2**2)
                    out[:, 2] = sin2 / (norm2 + 1e-10)
                    out[:, 3] = cos2 / (norm2 + 1e-10)
                last_state = out
                predictions.append(last_state.unsqueeze(1))
                
            predictions = torch.cat(predictions, dim=1)
        
        pred_norm = predictions.squeeze().cpu().numpy()
        return pred_norm
    

def train_model(model, train_loader, criterion, optimizer, num_epochs, device, normalize_sincos_values = False):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        h = None
        print(f"Epoch {epoch} of {num_epochs}")
        for x, y in train_loader:
            x = x   
            y = y
            optimizer.zero_grad()
            if h is not None:
                h = h.detach()
                
            output, h = model(x.to(device))
            if normalize_sincos_values:
                normalized_output = normalize_sincos(output)
            else:
                normalized_output = output
            mse_loss = criterion(normalized_output, y[:, :, :].to(device)) 
            mse_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()
            total_loss += mse_loss.item()
