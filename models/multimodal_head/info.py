import torch

C = torch.log(2*torch.tensor(torch.pi))/2
Info = lambda x: torch.norm(x)**2/2 + x.size(-1)*C
pMInfo = lambda x,y: x.size(-1)*C - torch.sum(x*y, dim=-1)
InfoDist = lambda x,y: (Info(x) - pMInfo(x,y)/2)/Info(x+y)