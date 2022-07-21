from torch import dtype
import torch

X = torch.tensor([1,2,3,4,5], dtype=torch.float32)
Y = torch.tensor([2,4,6,8,10], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
def loss(y,ypred):
    return ((y - ypred)**2).mean()

def forward(x):
    return w * x

epochs = 1000
lr = 0.01

for epoch in range(epochs):

    ypred = forward(X)

    l = loss(Y,ypred)

    l.backward() # calculate gradient value --> dl/dw

    with torch.no_grad():
        w -= (lr * w.grad)
        w.grad.zero_()
    
    print(f"epoch: {epoch+1:.3f} w: {w:.3f} loss: {l:.8f}")