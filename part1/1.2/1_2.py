import torch

print(torch.tensor([0,1,2,3]))
# tensor([0, 1, 2, 3]) :type is long

print(torch.Tensor([0,1,2,3]))
# tensor([0., 1., 2., 3.]) :type is float

print('------------------------------')

print(torch.Tensor(2,3))
# tensor([[-4.9988e+32,  3.0746e-41,  0.0000e+00],
#         [ 0.0000e+00,  1.4013e-45,  0.0000e+00]])

print(torch.Tensor(3))
# tensor([0.0000e+00, 0.0000e+00, 1.4013e-45])

print('------------------------------')

print(torch.tensor(range(10)))
print(torch.arange(10))
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(torch.tensor([[0,1,2],[3,4,5]]))
print(torch.arange(6).reshape(2,3))
# tensor([[0, 1, 2],
#         [3, 4, 5]])

a = torch.arange(6).reshape(2,3)
print(a.reshape(3,2))
# tensor([[0, 1],
#         [2, 3],
#         [4, 5]])

print('------------------------------')

### 四則演算可
a = torch.arange(5)
print(a+2)
# tensor([2, 3, 4, 5, 6])

### 行列同士の四則演算可（各要素の演算）
a = torch.arange(6).reshape(2,3)
b = a+1
print(a+b)
# tensor([[ 1,  3,  5],
#         [ 7,  9, 11]])

### 内積
a0 = torch.tensor([1.,2.,3.,4.])
a1 = torch.tensor([5.,6.,7.,8.])
print(torch.dot(a0,a1))
print(torch.matmul(a0,a1))
# tensor(70.)

a0 = torch.tensor([1,2,3,4])
a1 = torch.arange(8).reshape(2,4)
print(torch.mv(a1,a0))
print(torch.matmul(a1,a0))
# tensor([20, 60])

a0 = torch.arange(8).reshape(2,4)
a1 = torch.arange(8).reshape(4,2)
print(torch.mm(a0,a1))
print(torch.matmul(a0,a1))
# tensor([[28, 34],
#         [76, 98]])

a0 = torch.arange(24).reshape(-1,2,4)
a1 = torch.arange(24).reshape(-1,4,2)
print(torch.bmm(a0,a1))
print(torch.matmul(a0,a1))
# tensor([[[  28,   34],
#          [  76,   98]],

#         [[ 428,  466],
#          [ 604,  658]],

#         [[1340, 1410],
#          [1644, 1730]]])

print('------------------------------')

a = torch.tensor([1.,2.,3.,4.])
print(torch.sin(a))
print(torch.log(a))
# tensor([ 0.8415,  0.9093,  0.1411, -0.7568])
# tensor([0.0000, 0.6931, 1.0986, 1.3863])

print('------------------------------')

a0 = torch.tensor([1,2,3,4])
print(a0.dtype)
print(a0.type())
# torch.int64
# torch.LongTensor

a1 = torch.tensor([1.,2.,3.,4.])
print(a1.dtype)
print(a1.type())
# torch.float32
# torch.FloatTensor

a0 = torch.tensor([1.,2.,3.,4.], dtype=torch.float)
print(a0.type())
# torch.FloatTensor

a1 = a0.type(torch.LongTensor)
print(a1.type())
# torch.LongTensor

a0 = torch.tensor([1,2,3])
print(a0.dtype)
# torch.int64

b0 = a0.numpy()
print(b0.dtype)
# int64

a2 = torch.from_numpy(b0)
print(a2.dtype)
# torch.int64

print('------------------------------')

a = torch.zeros(6).reshape(2,3)
b = torch.ones(6).reshape(2,3)
print(torch.cat([a,b]))
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [1., 1., 1.],
#         [1., 1., 1.]])

print(torch.cat([a,b],dim=1))
# tensor([[0., 0., 0., 1., 1., 1.],
#         [0., 0., 0., 1., 1., 1.]])

c = b+1
print(torch.stack([a,b,c]))
# tensor([[[0., 0., 0.],
#          [0., 0., 0.]],

#         [[1., 1., 1.],
#          [1., 1., 1.]],

#         [[2., 2., 2.],
#          [2., 2., 2.]]])

print('------------------------------')

a = torch.arange(6).reshape(2,3)
print(a.unsqueeze(0))
# tensor([[[0, 1, 2],
#          [3, 4, 5]]])

a = torch.arange(12).reshape(2,2,-1)
print(a)
# tensor([[[ 0,  1,  2],
#          [ 3,  4,  5]],

#         [[ 6,  7,  8],
#          [ 9, 10, 11]]])

print(a.permute(2,0,1))
# tensor([[[ 0,  3],
#          [ 6,  9]],

#         [[ 1,  4],
#          [ 7, 10]],

#         [[ 2,  5],
#          [ 8, 11]]])