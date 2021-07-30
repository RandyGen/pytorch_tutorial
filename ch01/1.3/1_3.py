# unsolved
import torch

x1 = torch.tensor([1.], requires_grad=True)
x2 = torch.tensor([2.], requires_grad=True)
x3 = torch.tensor([3.], requires_grad=True)

z = (x1-2 * x2-1)**2 + (x2 * x3-1)**2 + 1
z.backward()

print(x1.grad)
# tensor([-8.])

print(x2.grad)
# tensor([46.])

print(x3.grad)
# tensor([20.])
