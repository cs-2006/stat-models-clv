import torch
import torch.nn as nn
import torch.optim as optim

class GammaGammaModel(nn.Module):
    def __init__(self, init_params=None):
        """
        Initialize the Gamma-Gamma model.
        The parameters p, q, and v are stored in log-space to ensure positivity.
        You can pass an init_params dictionary with keys 'p', 'q', and 'v'.
        """
        super(GammaGammaModel, self).__init__()
        if init_params is None:
            init_params = {'p': 1.0, 'q': 1.0, 'v': 1.0}
        self.log_p = nn.Parameter(torch.log(torch.tensor(init_params['p'], dtype=torch.float32)))
        self.log_q = nn.Parameter(torch.log(torch.tensor(init_params['q'], dtype=torch.float32)))
        self.log_v = nn.Parameter(torch.log(torch.tensor(init_params['v'], dtype=torch.float32)))

    def forward(self, x, m):
        """
        Compute the log-likelihood for each customer.
        
        Inputs:
          x : Tensor of number of transactions (must be > 0).
          m : Tensor of average monetary value for the customer.
          
        The likelihood for a single customer is given by:
        
          L(p, q, v | x, m) = [gamma(p+q*x) / (gamma(p) gamma(q*x))] * (v^p m^{(q-1)x}) / (v+x*m)^(p+q*x)
          
        We compute the log-likelihood in a numerically stable way.
        """
        x = x.float()
        m = m.float()
        
        p = torch.exp(self.log_p)
        q = torch.exp(self.log_q)
        v = torch.exp(self.log_v)
        
        term1 = torch.lgamma(p + q * x)
        term2 = - torch.lgamma(p)
        term3 = - torch.lgamma(q * x)
        term4 = p * torch.log(v)
        term5 = (q - 1) * x * torch.log(m)
        term6 = - (p + q * x) * torch.log(v + x * m)
        
        ll = term1 + term2 + term3 + term4 + term5 + term6
        return ll

    def negative_log_likelihood(self, x, m):
        """
        Returns the total negative log-likelihood for a batch of customers.
        """
        ll = self.forward(x, m)
        return -ll.sum()

    def conditional_expected_value(self, x, m):
        """
        Compute the conditional expected monetary value for a customer,
        which is given by:
        
          E(m|x, m) = (p + q*x) / (v + x*m)
        
        Note: The input x should be > 0 (since monetary values are observed only for customers with transactions).
        """
        x = x.float()
        m = m.float()
        
        p = torch.exp(self.log_p)
        q = torch.exp(self.log_q)
        v = torch.exp(self.log_v)
        
        return (p + q * x) / (v + x * m)


if __name__ == "__main__":
    x = torch.tensor([1, 2, 3, 1, 4, 2, 1, 3, 2, 5], dtype=torch.float32)
    m = torch.tensor([10.0, 20.0, 15.0, 12.0, 25.0, 18.0, 11.0, 22.0, 19.0, 30.0], dtype=torch.float32)

    model = GammaGammaModel(init_params={'p': 1.0, 'q': 1.0, 'v': 1.0})
    
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = model.negative_log_likelihood(x, m)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}, Negative Log-Likelihood: {loss.item():.4f}")
    
    p_est = torch.exp(model.log_p).item()
    q_est = torch.exp(model.log_q).item()
    v_est = torch.exp(model.log_v).item()
    
    print("\nEstimated Parameters:")
    print(f"p = {p_est:.4f}")
    print(f"q = {q_est:.4f}")
    print(f"v = {v_est:.4f}")
    
    expected_m = model.conditional_expected_value(x, m)
    print("\nConditional Expected Monetary Values:")
    print(expected_m)
