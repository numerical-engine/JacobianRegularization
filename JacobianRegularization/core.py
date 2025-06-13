import torch
import sys

def Jacobian_norm(x:torch.Tensor, z:torch.Tensor, n:int = 1)->torch.float:
    """Computes approximate Jacobian norm of the input tensor `x` with respect to `z`.

    Cite: <https://arxiv.org/pdf/1908.02729>

    Args:
        x (torch.Tensor): Input tensor which shape is (batch_size, input_dim, ...).
        z (torch.Tensor): Output tensor which shape is (batch_size, input_dim, ...).
        n (int, optional): Number of samples. Defaults to 1.
    Returns:
        torch.float: Approximate Jacobian norm.
    """
    assert x.requires_grad, "Input tensor x must have requires_grad=True to compute Jacobian norm."
    z = torch.flatten(z)
    batch_size = x.shape[0]; input_shape = x.shape[1:]; input_dim = torch.prod(torch.tensor(input_shape)).item()
    Jf = 0. # Initialize Jacobian norm

    for _ in range(n):
        random_vec = torch.randn(batch_size, *input_shape, device=x.device)
        random_vec /= torch.norm(random_vec, dim=1, keepdim=True)  # Normalize random vector
        random_vec = torch.flatten(random_vec)

        Jv = torch.dot(z, random_vec)
        dxJv = torch.autograd.grad(Jv, x, retain_graph=True, create_graph=True)[0]
        Jf = Jf + torch.norm(dxJv)
    
    Jf = Jf*input_dim/batch_size/n
    
    return Jf