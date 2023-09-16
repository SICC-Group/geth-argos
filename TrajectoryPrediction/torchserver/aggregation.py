import torch
import numpy as np
from control_params import params as cp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _median(updates: torch.Tensor):
    values_upper, _ = updates.median(dim=0)
    values_lower, _ = (-updates).median(dim=0)
    res = (values_upper - values_lower) / 2
    return res.detach().numpy().tolist()


def aggregate(accepted_gradients: list):
    gradients = torch.from_numpy(np.array(accepted_gradients)).to(
        dtype=torch.float
    )

    if cp['AGGREGATION'] == 'multi-Krum':
        pass

    if cp['AGGREGATION'] == 'GeoMed':
        pass

    if cp['AGGREGATION'] == 'AutoGM':
        pass
    
    if cp['AGGREGATION'] == 'Median':
        return _median(gradients)
    
    if cp['AGGREGATION'] == 'TrimmedMean':
        pass
    
    if cp['AGGREGATION'] == 'CenteredClipped':
        pass
    
    if cp['AGGREGATION'] == 'Clustering':
        pass
    
    if cp['AGGREGATION'] == 'ClippedClustering':
        pass
    
    if cp['AGGREGATION'] == 'DnC':
        pass
    
    if cp['AGGREGATION'] == 'SignGuard':
        pass