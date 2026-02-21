import torch, pickle

class GaussianDistribution:
    def __init__(self, dim=4096):
        self.mean = torch.zeros(dim) # 차원별 평균 
        self.var = torch.zeros(dim) # 차원별 분산
        self.n = 0 # 누적된 샘플의 개수 

    def update(self, new_data):
        new_data = new_data.view(-1, new_data.size(-1)) # torch.Size([8, 4096])
        batch_size = new_data.size(0)
        
        if self.n == 0:
            self.mean = new_data.mean(dim=0)
            self.var = new_data.var(dim=0, unbiased=False)
            self.n = self.n+batch_size 
        else:
            old_mean = self.mean.clone()
            total_samples = self.n + batch_size
            self.mean = old_mean + (new_data.mean(dim=0) - old_mean) * (batch_size / total_samples)
            
            new_var = new_data.var(dim=0, unbiased=False)
            mean_diff = new_data.mean(dim=0) - old_mean
            self.var = (self.n * self.var + batch_size * new_var + 
                        (self.n * batch_size / total_samples) * (mean_diff ** 2)) / total_samples
            
            self.n = total_samples

    def get_params(self):
        return self.mean, self.var, self.n

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def copy(self):
        new_dist = GaussianDistribution()
        new_dist.mean = self.mean.clone()
        new_dist.var = self.var.clone()
        new_dist.n = self.n
        return new_dist
    
    def __str__(self):
        return f"Gaussian Distribution:\nMean: {self.mean.to('cpu').float().numpy()}\nVariance: {self.var.to('cpu').float().numpy()}\nCount: {self.n}"

def to_device_dtype(dist: GaussianDistribution, device=None, dtype=None):
    if device is not None:
        dist.mean = dist.mean.to(device)
        dist.var = dist.var.to(device)
    if dtype is not None:
        dist.mean = dist.mean.to(dtype=dtype)
        dist.var = dist.var.to(dtype=dtype)
    return dist

def dist_mean_cpu(dist: GaussianDistribution) -> torch.Tensor:
    return dist.mean.detach().float().cpu()