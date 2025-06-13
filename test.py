import torch
import JacobianRegularization as JR

def test_function(x:torch.tensor)->torch.tensor:
    f1 = torch.sin(x[:,0])
    f2 = torch.cos(x[:,1])

    return torch.stack([f1, f2], dim=1)

x = torch.rand(100, 2)
x.requires_grad = True
z = test_function(x)

J = JR.Jacobian_norm(x, z, n=10)
print(J)