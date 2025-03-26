import torch
import torch.nn as nn
import torch.optim as optim

class BGNBDModel(nn.Module):
    def __init__(self, init_params=None):
        """
        Initialize the BG/NBD model.
        The parameters r, alpha, a, b are stored in log-space to ensure positivity.
        Optionally, you can pass an init_params dict with keys 'r', 'alpha', 'a', 'b'.
        """
        super(BGNBDModel, self).__init__()
        if init_params is None:
            init_params = {'r': 1.0, 'alpha': 1.0, 'a': 1.0, 'b': 1.0}
        self.log_r = nn.Parameter(torch.log(torch.tensor(init_params['r'], dtype=torch.float32)))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_params['alpha'], dtype=torch.float32)))
        self.log_a = nn.Parameter(torch.log(torch.tensor(init_params['a'], dtype=torch.float32)))
        self.log_b = nn.Parameter(torch.log(torch.tensor(init_params['b'], dtype=torch.float32)))
        
    def forward(self, x, t_x, T):
        """
        Compute the log-likelihood for each customer.
        
        For customers with no transactions (x == 0):
          log L(0,0,T) = r * log(alpha) - r * log(alpha+T) + log(b) - log(a+b)
        
        For customers with transactions (x > 0):
          log L(x,t_x,T) = 
             [log gamma(r+x) - log gamma(r) - log(x!)]
           + r * [log(alpha) - log(alpha+T)]
           + x * [log(T-t_x) - log(alpha+T)]
           + log(a) + log gamma(a+b) - log gamma(a) - log gamma(a+b+x) + log gamma(a+x)
           + log( {}_2F_1( r+x, a; a+b+x; (T-t_x)/(alpha+T) ) )
        
        Inputs:
          x   : Tensor of transaction counts.
          t_x : Tensor of recency (time of last transaction). For x == 0, t_x should be 0.
          T   : Tensor of total observation period.
        """
        # Ensure inputs are float tensors
        x = x.float()
        t_x = t_x.float()
        T = T.float()
        
        # Recover parameters in positive space
        r = torch.exp(self.log_r)
        alpha = torch.exp(self.log_alpha)
        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        
        ll = torch.empty_like(x)
        
        # Masks for customers with zero and at least one transaction
        mask0 = (x == 0)
        mask1 = (x > 0)
        
        # For customers with no transactions:
        if mask0.any():
            T0 = T[mask0]
            # log L = r*log(alpha) - r*log(alpha+T) + log(b) - log(a+b)
            ll[mask0] = r * (torch.log(alpha) - torch.log(alpha + T0)) + (torch.log(b) - torch.log(a + b))
        
        # For customers with at least one transaction:
        if mask1.any():
            x1 = x[mask1]
            t_x1 = t_x[mask1]
            T1 = T[mask1]
            
            # Term 1: Gamma ratio for transactions count
            term1 = torch.lgamma(r + x1) - torch.lgamma(r) - torch.lgamma(x1 + 1)
            
            # Term 2: Contribution from the transaction process
            term2 = r * (torch.log(alpha) - torch.log(alpha + T1))
            
            # Term 3: Contribution from the time component of transactions
            term3 = x1 * (torch.log(T1 - t_x1) - torch.log(alpha + T1))
            
            # Term 4: Contribution from the dropout process (using Beta-Gamma functions)
            term4 = torch.log(a) + torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(a + b + x1) + torch.lgamma(a + x1)
            
            # Term 5: Hypergeometric term, with z = (T-t_x)/(alpha+T)
            z = (T1 - t_x1) / (alpha + T1)
            hyp_val = torch.special.hyp2f1(r + x1, a, a + b + x1, z)
            term5 = torch.log(hyp_val)
            
            ll[mask1] = term1 + term2 + term3 + term4 + term5
        
        return ll

    def negative_log_likelihood(self, x, t_x, T):
        """
        Returns the total negative log-likelihood over a batch of customers.
        """
        ll = self.forward(x, t_x, T)
        return -ll.sum()


# === Example usage ===
if __name__ == "__main__":
    # Simulated (dummy) data for demonstration purposes:
    # Suppose we have 10 customers.
    # For each customer:
    # x: number of transactions, t_x: time of last transaction (0 for no transactions), T: total observation period.
    x = torch.tensor([0, 1, 2, 0, 3, 1, 0, 2, 1, 4], dtype=torch.float32)
    t_x = torch.tensor([0, 0.5, 1.0, 0, 1.5, 0.8, 0, 1.2, 0.6, 1.8], dtype=torch.float32)
    T = torch.tensor([2.0] * 10, dtype=torch.float32)

    model = BGNBDModel()

    # Set up an optimizer 
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = model.negative_log_likelihood(x, t_x, T)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}, Negative Log-Likelihood: {loss.item():.4f}")

    # Retrieve the estimated parameters:
    r_est = torch.exp(model.log_r).item()
    alpha_est = torch.exp(model.log_alpha).item()
    a_est = torch.exp(model.log_a).item()
    b_est = torch.exp(model.log_b).item()

    print("\nEstimated Parameters:")
    print(f"r     = {r_est:.4f}")
    print(f"alpha = {alpha_est:.4f}")
    print(f"a     = {a_est:.4f}")
    print(f"b     = {b_est:.4f}")
