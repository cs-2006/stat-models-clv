import torch
import torch.nn as nn
import torch.optim as optim

class ParetoNBDModel(nn.Module):
    def __init__(self, init_params=None):
        """
        Initialize the Pareto/NBD model.
        The parameters r, alpha, s, beta are stored in log-space.
        Optionally, you can pass an init_params dict with keys 'r', 'alpha', 's', 'beta'.
        """
        super(ParetoNBDModel, self).__init__()
        if init_params is None:
            init_params = {'r': 1.0, 'alpha': 1.0, 's': 1.0, 'beta': 1.0}
        # Store parameters in log-space to enforce positivity
        self.log_r = nn.Parameter(torch.log(torch.tensor(init_params['r'], dtype=torch.float32)))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_params['alpha'], dtype=torch.float32)))
        self.log_s = nn.Parameter(torch.log(torch.tensor(init_params['s'], dtype=torch.float32)))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(init_params['beta'], dtype=torch.float32)))

    def forward(self, x, t_x, T):
        """
        Compute the log-likelihood for each customer.
        
        For customers with x > 0 (at least one transaction):
          log L = 
            log[gamma(r+x)] - log[gamma(r)] - log(x!) +
            r*log(alpha) + s*log(beta) -
            (r+x)*log(alpha+T) - (s+1)*log(beta+T-t_x) +
            log( _2F_1(s+1, r+x; r; (T-t_x)/(alpha+T)) )
        
        For customers with x == 0:
          log L = r*log(alpha) - r*log(alpha+T) + s*log(beta) - s*log(beta+T)
        
        Inputs:
          x   : Tensor of transaction counts.
          t_x : Tensor of recency (time of last transaction). For x == 0, t_x should be 0.
          T   : Tensor of total observation period.
        """
        # Ensure tensors are float
        x = x.float()
        t_x = t_x.float()
        T = T.float()
        
        # Convert parameters back to positive space
        r = torch.exp(self.log_r)
        alpha = torch.exp(self.log_alpha)
        s = torch.exp(self.log_s)
        beta = torch.exp(self.log_beta)
        
        # Prepare output tensor (same shape as x)
        ll = torch.empty_like(x)
        
        # Boolean masks for customers with x==0 and x>0
        mask0 = (x == 0)
        mask1 = (x > 0)
        
        # For customers with zero transactions: use the closed form
        # log L(0,0,T) = r*log(alpha) - r*log(alpha+T) + s*log(beta) - s*log(beta+T)
        if mask0.any():
            T0 = T[mask0]
            ll[mask0] = r * torch.log(alpha) - r * torch.log(alpha + T0) + \
                        s * torch.log(beta) - s * torch.log(beta + T0)
        
        # For customers with at least one transaction:
        if mask1.any():
            x1 = x[mask1]
            t_x1 = t_x[mask1]
            T1 = T[mask1]
            # First term: log[gamma(r+x)] - log[gamma(r)] - log(x!)
            gamma_term = torch.lgamma(r + x1) - torch.lgamma(r) - torch.lgamma(x1 + 1)
            # Second term: r*log(alpha) + s*log(beta)
            term2 = r * torch.log(alpha) + s * torch.log(beta)
            # Third term: -(r+x)*log(alpha+T)
            term3 = -(r + x1) * torch.log(alpha + T1)
            # Fourth term: -(s+1)*log(beta+T-t_x)
            term4 = -(s + 1) * torch.log(beta + T1 - t_x1)
            # z for hypergeometric: (T-t_x)/(alpha+T)
            z = (T1 - t_x1) / (alpha + T1)
            # Fifth term: log( _2F_1(s+1, r+x; r; z) )
            # (torch.special.hyp2f1 is available in recent PyTorch versions)
            hyp_val = torch.special.hyp2f1(s + 1, r + x1, r, z)
            term5 = torch.log(hyp_val)
            ll[mask1] = gamma_term + term2 + term3 + term4 + term5
        
        return ll

    def negative_log_likelihood(self, x, t_x, T):
        """
        Returns the total negative log-likelihood for a batch of customers.
        """
        ll = self.forward(x, t_x, T)
        # Return negative total log-likelihood
        return -ll.sum()


# === Example usage ===
if __name__ == "__main__":
    # Simulated (dummy) data for demonstration:
    # Suppose we have 10 customers. For each customer:
    # x: number of transactions, t_x: time of last transaction, T: total observation time.
    # (In practice, you’d use your actual data.)
    x = torch.tensor([0, 2, 1, 3, 0, 4, 1, 2, 0, 3], dtype=torch.float32)
    # For customers with no transactions, t_x should be 0.
    t_x = torch.tensor([0, 1.5, 1.0, 2.0, 0, 3.0, 1.0, 1.8, 0, 2.5], dtype=torch.float32)
    # Total observation period (assumed same for all or could be different)
    T = torch.tensor([3.0]*10, dtype=torch.float32)

    # Create the model 
    model = ParetoNBDModel()

    # Set up an optimizer 
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = model.negative_log_likelihood(x, t_x, T)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 200 == 0:
            print(f"Epoch {epoch+1}, Negative Log-Likelihood: {loss.item():.4f}")

    # Access the estimated parameters:
    r_est = torch.exp(model.log_r).item()
    alpha_est = torch.exp(model.log_alpha).item()
    s_est = torch.exp(model.log_s).item()
    beta_est = torch.exp(model.log_beta).item()

    print("\nEstimated Parameters:")
    print(f"r     = {r_est:.4f}")
    print(f"alpha = {alpha_est:.4f}")
    print(f"s     = {s_est:.4f}")
    print(f"beta  = {beta_est:.4f}")
