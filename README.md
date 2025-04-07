Args:
    device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
    kernel (str, optional): Kernel type ('linear' or 'rbf'). Defaults to 'linear'.
    sigma (float, optional): Sigma parameter for RBF kernel. If None, uses median distance. Defaults to None.
    standardize (bool, optional): Whether to standardize input features. Defaults to True.
"""

def __init__(self, device='cpu', kernel='linear', sigma=None, standardize=True):
    self.device = device
    self.kernel = kernel
    self.sigma = sigma
    self.standardize = standardize
    
def _gram_linear(self, X):
    """Compute Gram matrix for linear kernel."""
    return torch.mm(X, X.T)

def _gram_rbf(self, X):
    """Compute Gram matrix for RBF kernel."""
    X = X.reshape(X.shape[0], -1)
    pairwise_dists = torch.cdist(X, X, p=2).pow(2)  # Squared Euclidean distances
    
    if self.sigma is None:
        n = X.shape[0]
        if n <= 1:
            sigma = 1.0  # Handle edge cases
        else:
            # Extract upper triangle excluding diagonal
            triu_mask = torch.triu(torch.ones_like(pairwise_dists, dtype=torch.bool), diagonal=1)
            valid_dists = pairwise_dists[triu_mask]
            if valid_dists.numel() == 0:
                sigma = 1.0
            else:
                sigma = torch.sqrt(torch.median(valid_dists))
    else:
        sigma = torch.tensor(self.sigma, device=self.device)
    
    return torch.exp(-pairwise_dists / (2 * sigma**2))

def _center_kernel(self, K):
    """Center the kernel matrix."""
    n = K.shape[0]
    ones = torch.ones((n, n), device=self.device)
    identity = torch.eye(n, device=self.device)
    H = identity - ones / n
    return H @ K @ H

def __call__(self, X, Y):
    """Compute CKA similarity between X and Y.
    
    Args:
        X (torch.Tensor): Features of shape (n_samples, ...)
        Y (torch.Tensor): Features of shape (n_samples, ...)
        
    Returns:
        float: CKA similarity score
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    
    X = X.to(self.device)
    Y = Y.to(self.device)
    
    # Flatten features
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    
    # Standardize features
    if self.standardize:
        X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-5)
        Y = (Y - Y.mean(0, keepdim=True)) / (Y.std(0, keepdim=True) + 1e-5)
    
    # Compute Gram matrices
    if self.kernel == 'linear':
        gram_x = self._gram_linear(X)
        gram_y = self._gram_linear(Y)
    elif self.kernel == 'rbf':
        gram_x = self._gram_rbf(X)
        gram_y = self._gram_rbf(Y)
    else:
        raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    # Center kernels
    gram_x = self._center_kernel(gram_x)
    gram_y = self._center_kernel(gram_y)
    
    # Compute CKA
    hsic = (gram_x * gram_y).sum()
    normalization = torch.sqrt(gram_x.pow(2).sum() * torch.sqrt(gram_y.pow(2).sum())
    
    return (hsic / normalization).item()
