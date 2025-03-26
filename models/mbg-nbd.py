import torch
import torch.nn as nn
import torch.optim as optim

class MBGNBDModel(nn.Module):
    def __init__(self, init_params=None):
        """
        Initialize the MBG/NBD model.
        The four parameters (r, alpha, a, b) are stored in log-space to ensure positivity.
        Optionally, you can provide initial values via a dictionary.
        """
        super(MBGNBDModel, self).__init__()
        if init_params is None:
            init_params = {'r': 1.0, 'alpha': 1.0, 'a': 1.1, 'b': 1.0}  # note: a is set > 1 for stability
        self.log_r = nn.Parameter(torch.log(torch.tensor(init_params['r'], dtype=torch.float32)))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_params['alpha'], dtype=torch.float32)))
        self.log_a = nn.Parameter(torch.log(torch.tensor(init_params['a'], dtype=torch.float32)))
        self.log_b = nn.Parameter(torch.log(torch.tensor(init_params['b'], dtype=torch.float32)))

    def forward(self, x, t_x, T):
        """
        Compute the log-likelihood for each customer based on their transaction history.
        
        Inputs:
          x   : Tensor of transaction counts.
          t_x : Tensor of recency (time of last transaction). For x==0, t_x should be 0.
          T   : Tensor of total observation time.
          
        The individual likelihood (integrated over heterogeneity) is given by:
        
          L(x,t_x,T) = (gamma(r+x)/gamma(r)) · alpha^r · { [B(a, b+x+1)/B(a, b)]·(alpha+T)^(–(r+x))
                                             + [B(a+1, b+x)/B(a, b)]·(alpha+t_x)^(–(r+x)) }.
                                             
        We compute this in log-space using the log-sum-exp trick.
        """
        # Ensure inputs are float tensors
        x   = x.float()
        t_x = t_x.float()
        T   = T.float()
        
        # Recover parameters in positive space
        r     = torch.exp(self.log_r)
        alpha = torch.exp(self.log_alpha)
        a     = torch.exp(self.log_a)
        b     = torch.exp(self.log_b)
        
        # Compute the common gamma ratio: log(gamma(r+x)) - log(gamma(r))
        log_gamma_ratio = torch.lgamma(r + x) - torch.lgamma(r)
        # Common term: r * log(alpha)
        common = log_gamma_ratio + r * torch.log(alpha)
        
        # Compute the beta-function ratios.
        # log[B(a, b+x+1)] - log[B(a, b)]
        logB1 = torch.lgamma(a) + torch.lgamma(b + x + 1) - torch.lgamma(a + b + x + 1)
        logB0 = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        beta_ratio1 = logB1 - logB0  # term for dropout after T
        
        # log[B(a+1, b+x)] - log[B(a, b)]
        logB2 = torch.lgamma(a + 1) + torch.lgamma(b + x) - torch.lgamma(a + b + x + 1)
        beta_ratio2 = logB2 - logB0  # term for dropout at t_x
        
        # Term 1: customer remains active at T (no dropout at time zero)
        log_term1 = common - (r + x) * torch.log(alpha + T) + beta_ratio1
        # Term 2: customer drops out at t_x (including dropout possibility at time zero)
        log_term2 = common - (r + x) * torch.log(alpha + t_x) + beta_ratio2
        
        # Use log-sum-exp to combine the two likelihood components
        stacked = torch.stack([log_term1, log_term2], dim=0)
        log_likelihood = torch.logsumexp(stacked, dim=0)
        
        return log_likelihood

    def negative_log_likelihood(self, x, t_x, T):
        """
        Returns the total negative log-likelihood for a batch of customers.
        """
        ll = self.forward(x, t_x, T)
        return -ll.sum()

    def active_probability(self, x, t_x, T):
        """
        Compute the probability that a customer is active at time T.
        Based on the model derivation, this is given by:
        
          P(Active|x,t_x,T) = 1 / { 1 + [gamma(a+1)gamma(b+x) / (gamma(a)gamma(b+x+1))]·((alpha+T)/(alpha+t_x))^(r+x) }
          
        The calculation is done in log-space.
        """
        x   = x.float()
        t_x = t_x.float()
        T   = T.float()
        
        r     = torch.exp(self.log_r)
        alpha = torch.exp(self.log_alpha)
        a     = torch.exp(self.log_a)
        b     = torch.exp(self.log_b)
        
        # Compute log factor: log(gamma(a+1)) + log(gamma(b+x)) - log(gamma(a)) - log(gamma(b+x+1))
        log_factor = torch.lgamma(a + 1) + torch.lgamma(b + x) - torch.lgamma(a) - torch.lgamma(b + x + 1)
        # And add the term (r+x)*[log(alpha+T) - log(alpha+t_x)]
        log_factor = log_factor + (r + x) * (torch.log(alpha + T) - torch.log(alpha + t_x))
        
        # Now, P(active) = 1 / (1 + exp(log_factor))
        p_active = 1.0 / (1.0 + torch.exp(log_factor))
        return p_active

    def expected_transactions(self, x, t_x, T, t):
        """
        Compute the expected number of future transactions in a period of length t 
        (i.e. in the interval (T, T+t]) for a customer with history (x, t_x, T).
        
        Here we use a formulation that combines the probability of being active with a
        hypergeometric term:
        
          E(Y(t)|x,t_x,T) = P(Active|x,t_x,T) · ((b+x)/(a-1)) · {}_2F_1( r+x, b+x+1; a+b+x; z )
          
        where z = t / (t + T + alpha).
        
        Note: This expression assumes a > 1.
        """
        x   = x.float()
        t_x = t_x.float()
        T   = T.float()
        t   = t.float()
        
        r     = torch.exp(self.log_r)
        alpha = torch.exp(self.log_alpha)
        a     = torch.exp(self.log_a)
        b     = torch.exp(self.log_b)
        
        p_active = self.active_probability(x, t_x, T)
        
        # z is defined as:
        z = t / (t + T + alpha)
        # Compute the hypergeometric term {}_2F_1(r+x, b+x+1; a+b+x; z)
        hyp_term = torch.special.hyp2f1(r + x, b + x + 1, a + b + x, z)
        
        # Expected future transactions (for customers who remain active)
        exp_trans = p_active * ((b + x) / (a - 1)) * hyp_term
        
        return exp_trans


# === Example usage ===
if __name__ == "__main__":
    # Suppose we have 10 customers with:
    # x: number of transactions,
    # t_x: time of the last transaction (0 if no transactions),
    # T: total observation period.
    x   = torch.tensor([0, 1, 2, 0, 3, 1, 0, 2, 1, 4], dtype=torch.float32)
    t_x = torch.tensor([0, 0.5, 1.0, 0, 1.5, 0.8, 0, 1.2, 0.6, 1.8], dtype=torch.float32)
    T   = torch.tensor([2.0]*10, dtype=torch.float32)
    
    model = MBGNBDModel(init_params={'r': 1.0, 'alpha': 4.0, 'a': 1.1, 'b': 1.0})
    
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = model.negative_log_likelihood(x, t_x, T)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 200 == 0:
            print(f"Epoch {epoch+1}, Negative Log-Likelihood: {loss.item():.4f}")
    
    r_est     = torch.exp(model.log_r).item()
    alpha_est = torch.exp(model.log_alpha).item()
    a_est     = torch.exp(model.log_a).item()
    b_est     = torch.exp(model.log_b).item()
    
    print("\nEstimated Parameters:")
    print(f"r     = {r_est:.4f}")
    print(f"alpha = {alpha_est:.4f}")
    print(f"a     = {a_est:.4f}")
    print(f"b     = {b_est:.4f}")
    
    # For demonstration, compute active probability and expected future transactions
    p_active = model.active_probability(x, t_x, T)
    print("\nProbability of being active for each customer:")
    print(p_active)
    
    t_future = torch.tensor([1.0]*10, dtype=torch.float32)
    exp_future = model.expected_transactions(x, t_x, T, t_future)
    print("\nExpected number of future transactions in the next time unit:")
    print(exp_future)
