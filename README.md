# AutoEncSets  
[paper](https://arxiv.org/abs/1803.02879)  
Fixed Point : 
- Add `norm = torch.zeros(ind_max).to(input.device).index_add_(0, index, torch.ones_like(index).float()) + self.eps` to line 197 in exchangeable_tensor/sp_layers.py 
