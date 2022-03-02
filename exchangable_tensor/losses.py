import torch

EPS = 1e-12

def mse(predicted, target, mask=None):
    if mask is not None:
        ans = torch.sum(torch.pow(predicted - target, 2) * mask).item() / torch.sum(mask).item()
        return torch.tensor([ans])
    else:
        ans = torch.mean(torch.pow(predicted - target, 2))
        return torch.tensor([ans])

def softmax(x, dim=-1):
    m = torch.clamp(torch.max(x, dim=dim, keepdim=True)[0], min=0.) 
    exps = torch.exp(x - m)
    return exps / (torch.sum(exps, dim=dim, keepdim=True))

def ce(predicted, target, mask=None):
    if mask is not None:
        #print(softmax(predicted))
        return torch.sum(-target * torch.log(EPS + softmax(predicted)) * mask)/ (torch.sum(mask))
    else:
        return torch.mean(target * torch.log(softmax(predicted)))

