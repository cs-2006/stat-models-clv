import torch
import torch.nn as nn
import torch.optim as optim
import torch.special  # for hyp2f1
import numpy as np
import pandas as pd


# 1) BG/NBD Model (Fully Implemented: Log-Likelihood + Expected Transactions)

class BGNBDModel(nn.Module):
    """
    BG/NBD model for transaction frequency:
      - r, alpha : Gamma-Poisson mixture for transaction rates
      - a, b     : Beta-Geometric mixture for dropout
    We implement:
      forward(x,t_x,T) -> log-likelihood for each customer
      expected_transactions(...) -> closed-form approximation of E[# of future txns]
    """
    def __init__(self, init_params=None):
        super(BGNBDModel, self).__init__()
        if init_params is None:
            init_params = {'r': 1.0, 'alpha': 1.0, 'a': 1.0, 'b': 1.0}
        self.log_r = nn.Parameter(torch.log(torch.tensor(init_params['r'], dtype=torch.float32)))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_params['alpha'], dtype=torch.float32)))
        self.log_a = nn.Parameter(torch.log(torch.tensor(init_params['a'], dtype=torch.float32)))
        self.log_b = nn.Parameter(torch.log(torch.tensor(init_params['b'], dtype=torch.float32)))

    def forward(self, x, t_x, T):
        """
        Return the log-likelihood (LL) for each customer under BG/NBD.

        If x == 0:
          log L(0,0,T) = r*log(alpha) - r*log(alpha + T) + log(b) - log(a+b)

        If x > 0:
          log L(x,t_x,T) =
             + logGamma(r+x) - logGamma(r) - logGamma(x+1)
             + r [log(alpha) - log(alpha+T)]
             + x [log(T - t_x) - log(alpha+T)]
             + log(a) + logGamma(a+b) - logGamma(a) - logGamma(a+b+x) + logGamma(a+x)
             + log( 2F1(r+x, a; a+b+x; (T - t_x)/(alpha+T)) )
        """
        # Ensure float
        x   = x.float()
        t_x = t_x.float()
        T   = T.float()

        # Recover parameters
        r = torch.exp(self.log_r)
        alpha = torch.exp(self.log_alpha)
        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)

        ll = torch.empty_like(x)

        mask0 = (x == 0)
        mask1 = (x > 0)

        # --- x=0 case ---
        if mask0.any():
            T0 = T[mask0]
            ll0 = (r * torch.log(alpha)
                   - r * torch.log(alpha + T0)
                   + torch.log(b)
                   - torch.log(a + b))
            ll[mask0] = ll0

        # --- x>0 case ---
        if mask1.any():
            x1   = x[mask1]
            t_x1 = t_x[mask1]
            T1   = T[mask1]

            term1 = ( torch.lgamma(r + x1)
                      - torch.lgamma(r)
                      - torch.lgamma(x1 + 1) )
            term2 = r * ( torch.log(alpha) - torch.log(alpha + T1) )
            term3 = x1 * ( torch.log(T1 - t_x1) - torch.log(alpha + T1) )
            term4 = ( torch.log(a)
                      + torch.lgamma(a + b)
                      - torch.lgamma(a)
                      - torch.lgamma(a + b + x1)
                      + torch.lgamma(a + x1) )
            z = (T1 - t_x1) / (alpha + T1)
            # hypergeometric
            hyp_val = torch.special.hyp2f1(r + x1, a, a + b + x1, z)
            term5 = torch.log(hyp_val + 1e-30)  # add small epsilon for safety

            ll1 = term1 + term2 + term3 + term4 + term5
            ll[mask1] = ll1

        return ll

    def negative_log_likelihood(self, x, t_x, T):
        """Sum of negative log-likelihood over all customers."""
        ll = self.forward(x, t_x, T)
        return -ll.sum()

    def expected_transactions(self, x, t_x, T, t_future=10.0):
        """
        Predict E[# of transactions in (T, T + t_future)] for a customer with
        history (x, t_x, T) under BG/NBD. We'll use a known approximation:

          1) Probability that the customer is alive at time T:
             p_alive = 1 / [ 1 + factor ]
             where factor = exp( log_factor ) as in Hardie notes.

          2) E[X(t_future)| x,t_x,T] ~ p_alive * ((r + x)/(alpha + T)) * t_future
             (This is a common simplified expression for BG/NBD.)

        For more exact formulas, see Hardie & Fader (2005).
        """
        x   = x.float()
        t_x = t_x.float()
        T   = T.float()

        r = torch.exp(self.log_r)
        alpha = torch.exp(self.log_alpha)
        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)

        # Probability alive
        # log_factor = logGamma(a+1) + logGamma(b+x) - logGamma(a) - logGamma(b+x+1)
        #           + (r+x)*[ log(alpha+T) - log(alpha+t_x) ]
        log_factor = ( torch.lgamma(a + 1)
                       + torch.lgamma(b + x)
                       - torch.lgamma(a)
                       - torch.lgamma(b + x + 1) )
        log_factor += (r + x) * ( torch.log(alpha + T) - torch.log(alpha + t_x + 1e-8) )
        factor = torch.exp(log_factor)
        p_alive = 1.0 / (1.0 + factor)

        # approximate expected future transactions
        return p_alive * ( (r + x) / (alpha + T + 1e-8) ) * t_future



# 2) Gamma–Gamma Model (Fully Implemented)

class GammaGammaModel(nn.Module):
    """
    Gamma–Gamma model for monetary value:
      p, q, v (all stored in log-space).
    forward() -> log-likelihood
    conditional_expected_value(x,m) -> E(m|x,m).
    """
    def __init__(self, init_params=None):
        super(GammaGammaModel, self).__init__()
        if init_params is None:
            init_params = {'p': 1.0, 'q': 1.0, 'v': 1.0}
        self.log_p = nn.Parameter(torch.log(torch.tensor(init_params['p'], dtype=torch.float32)))
        self.log_q = nn.Parameter(torch.log(torch.tensor(init_params['q'], dtype=torch.float32)))
        self.log_v = nn.Parameter(torch.log(torch.tensor(init_params['v'], dtype=torch.float32)))

    def forward(self, x, m):
        """
        Return log-likelihood for each customer:
          log L = logGamma(p + qx) - logGamma(p) - logGamma(qx)
                  + p log v + (q - 1)x log m - (p + qx) log(v + x m)
        Usually only applied to customers with x>0, but here we just do it for all.
        """
        x = x.float()
        m = m.float()

        p = torch.exp(self.log_p)
        q = torch.exp(self.log_q)
        v = torch.exp(self.log_v)

        # We'll define a small epsilon for safety
        eps = 1e-30

        term1 = torch.lgamma(p + q*x) - torch.lgamma(p) - torch.lgamma(q*x + eps)
        term2 = p * torch.log(v + eps)
        term3 = (q - 1) * x * torch.log(m + eps)
        term4 = - (p + q*x) * torch.log(v + x*m + eps)

        ll = term1 + term2 + term3 + term4
        return ll

    def negative_log_likelihood(self, x, m):
        ll = self.forward(x, m)
        return -ll.sum()

    def conditional_expected_value(self, x, m):
        """
        E(m| x, m) = (p + q*x) / (v + x*m).
        """
        x = x.float()
        m = m.float()
        p = torch.exp(self.log_p)
        q = torch.exp(self.log_q)
        v = torch.exp(self.log_v)
        eps = 1e-8
        return (p + q * x) / (v + x*m + eps)



# 3) Composite Model that Multiplies BG/NBD * Gamma–Gamma to Predict CLV

class CompositeCLVModel(nn.Module):
    """
    - We train it by comparing predicted CLV to a known 'regression_label'.
    - The forward pass = E[# future transactions] * E[monetary value].
    """
    def __init__(self, init_bgnbd=None, init_gg=None, t_future=10.0):
        super(CompositeCLVModel, self).__init__()
        self.bgnbd = BGNBDModel(init_params=init_bgnbd)
        self.ggamma = GammaGammaModel(init_params=init_gg)
        self.t_future = t_future

    def forward(self, x, t_x, T, m):
        # predicted # future transactions
        count_pred = self.bgnbd.expected_transactions(x, t_x, T, self.t_future)
        # predicted average value
        val_pred = self.ggamma.conditional_expected_value(x, m)
        return count_pred * val_pred

    def loss_mse(self, x, t_x, T, m, actual_spend):
        """
        MSE vs. the actual future spend (regression_label).
        """
        pred_spend = self.forward(x, t_x, T, m)
        return torch.mean((pred_spend - actual_spend)**2)



# 4) Data Preparation: parse (x, t_x, T) from 'days_before_lst', parse m from 'articles_ids_lst'

def parse_x_t_x_T(row):
    """
    Example parse from 'days_before_lst':
       x = len(days_before_lst)
       T = sum of days_before_lst
       t_x = last element
    """
    days_list = row['days_before_lst']
    if not isinstance(days_list, list) or len(days_list) == 0:
        return 0, 0.0, 0.0
    else:
        x = len(days_list)
        T = float(sum(days_list))
        t_x = float(days_list[-1])
        return x, t_x, T

def parse_avg_monetary_value(row):
    """
    Example parse from 'articles_ids_lst':
    We'll define a dummy approach: average value = 20 + 0.1 * (#articles).
    """
    arts = row['articles_ids_lst']
    if not isinstance(arts, list) or len(arts) == 0:
        return 20.0
    else:
        return 20.0 + 0.1*len(arts)

def build_tensor_dataset(df):
    """
    Convert the DataFrame to PyTorch Tensors for (x, t_x, T, m, regression_label).
    We dont use 'classification_label' or 'customer_id' or 'postal_code' as features.
    """
    x_list, t_x_list, T_list, m_list, spend_list = [], [], [], [], []
    for _, row in df.iterrows():
        x_val, tx_val, bigT_val = parse_x_t_x_T(row)
        m_val = parse_avg_monetary_value(row)
        # future spend is stored in regression_label
        clv = row.get('regression_label', 0.0)

        x_list.append(x_val)
        t_x_list.append(tx_val)
        T_list.append(bigT_val)
        m_list.append(m_val)
        spend_list.append(clv)

    x_ten = torch.tensor(x_list, dtype=torch.float32)
    t_x_ten = torch.tensor(t_x_list, dtype=torch.float32)
    T_ten = torch.tensor(T_list, dtype=torch.float32)
    m_ten = torch.tensor(m_list, dtype=torch.float32)
    spend_ten = torch.tensor(spend_list, dtype=torch.float32)

    return x_ten, t_x_ten, T_ten, m_ten, spend_ten



# 5) Example Pipeline: train_and_validate

def train_and_validate(train_df, val_df, test_df):
    """
    1) Build Tensors from each DataFrame
    2) Create CompositeCLVModel
    3) Train end-to-end to minimize MSE vs. regression_label
    4) Evaluate on val/test sets
    """
    # Convert to Tensors
    x_train, t_x_train, T_train, m_train, spend_train = build_tensor_dataset(train_df)
    x_val, t_x_val, T_val, m_val, spend_val = build_tensor_dataset(val_df)
    x_test, t_x_test, T_test, m_test, spend_test = build_tensor_dataset(test_df)

    # Create the composite model
    model = CompositeCLVModel(
        init_bgnbd={'r':1.0, 'alpha':1.0, 'a':1.0, 'b':1.0},
        init_gg={'p':1.0, 'q':1.0, 'v':1.0},
        t_future=10.0  # horizon for future transactions
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    n_epochs = 500
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = model.loss_mse(x_train, t_x_train, T_train, m_train, spend_train)
        loss.backward()
        optimizer.step()

        # check validation
        if (epoch+1) % 100 == 0:
            with torch.no_grad():
                val_preds = model.forward(x_val, t_x_val, T_val, m_val)
                val_loss = torch.mean((val_preds - spend_val)**2)
            print(f"Epoch {epoch+1}/{n_epochs}, Train MSE={loss.item():.4f}, Val MSE={val_loss.item():.4f}")

    # Final test evaluation
    with torch.no_grad():
        test_preds = model.forward(x_test, t_x_test, T_test, m_test)
        test_mse = torch.mean((test_preds - spend_test)**2).item()
    print(f"\n=== Final Test MSE: {test_mse:.4f} ===")

    # Print learned parameters
    print("\nLearned BG/NBD parameters:")
    print("r     =", torch.exp(model.bgnbd.log_r).item())
    print("alpha =", torch.exp(model.bgnbd.log_alpha).item())
    print("a     =", torch.exp(model.bgnbd.log_a).item())
    print("b     =", torch.exp(model.bgnbd.log_b).item())

    print("\nLearned Gamma–Gamma parameters:")
    print("p =", torch.exp(model.ggamma.log_p).item())
    print("q =", torch.exp(model.ggamma.log_q).item())
    print("v =", torch.exp(model.ggamma.log_v).item())


##############################################################################
# 6) DEMO
##############################################################################
# train_and_validate(train_df, val_df, test_df)


if __name__ == "__main__":
    # Minimal dummy data
    train_df = pd.DataFrame({
        'customer_id': ['C1','C2','C3'],
        'days_before_lst': [[10,20],[30],[15,5,5]],
        'articles_ids_lst': [[111,112],[999],[50,51,52]],
        'regression_label': [100.0, 30.0, 120.0],
        'classification_label': [1, 0, 1],  # not used
        'age': [25, 40, 35],               # not used here
        'postal_code': ['xyz','abc','def'] # not used
    })
    val_df = pd.DataFrame({
        'customer_id': ['C4'],
        'days_before_lst': [[10]],
        'articles_ids_lst': [[1234]],
        'regression_label': [50.0],
        'classification_label': [0],
        'age': [45],
        'postal_code': ['zzz']
    })
    test_df = pd.DataFrame({
        'customer_id': ['C5'],
        'days_before_lst': [[5,5]],
        'articles_ids_lst': [[9999,10000]],
        'regression_label': [80.0],
        'classification_label': [1],
        'age': [50],
        'postal_code': ['qqq']
    })

    print("=== Training on dummy data ===")
    train_and_validate(train_df, val_df, test_df)
