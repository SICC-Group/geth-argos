import random
import torch

from control_params import params as cp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(cp['RANDOM_SEED'])


def _alie_attack(
    updates: torch.Tensor,
    num_of_byzantine: int,
    negative_indices = None,
    **kwargs,
):
    num_of_clients = kwargs["num_of_clients"]
    s = torch.floor_divide(num_of_clients, 2) + 1 - num_of_byzantine
    cdf_value = (num_of_clients - num_of_byzantine - s) / (
        num_of_clients - num_of_byzantine
    )
    dist = torch.distributions.normal.Normal(
        torch.tensor(0.0), torch.tensor(1.0)
    )
    z_max = dist.icdf(cdf_value)

    mean = updates.mean(dim=0)
    std = updates.std(dim=0)

    if kwargs["aggregation"] == "signGuard":
        if negative_indices is None:
            num_elements = len(std)
            num_negate = num_elements // 2
            negative_indices = random.sample(range(num_elements), num_negate)
        std[negative_indices] *= -1
    
    res = mean - std * z_max
    
    return res


def _ipm_attack(
    updates: torch.Tensor,
    scale: float = 1.0,
    **kwargs,
):
    mean = updates.mean(dim=0)
    if kwargs["aggregation"] == "multiKrum":
        scale = 0.1
    res = -scale * mean

    return res


def _noise_attack(
    updates: torch.Tensor,
    mu: float = 0.1,
    sigma: float = 0.1,
    **kwargs
):
    return torch.normal(mu, sigma, size=updates[0].shape)


def _last_round(
    mu: float = 0.0,
    sigma: float = 0.1,
    add_noise: bool = True,
    **kwargs,
):
    last_round = torch.tensor(kwargs["last_round"], dtype=torch.float)
    if add_noise:
        last_round = (
            last_round + torch.normal(mu, sigma, size=last_round.shape)
        )
    return last_round


adversaries = {
    "ALIE": _alie_attack,
    "IPM": _ipm_attack,
    "noise": _noise_attack,
    "last-round": _last_round,
}


def attack(
    benign_gradients: list,
    num_of_byzantine: int,
    adversary: str,
    aggregation: str,
    num_of_clients: int,
    last_round: list,
):
    if adversary in adversaries:
        gradients = torch.tensor(benign_gradients, dtype=torch.float)
        res = adversaries[adversary](
            updates=gradients,
            num_of_byzantine=num_of_byzantine,
            aggregation=aggregation,
            num_of_clients=num_of_clients,
            last_round=last_round,
        )
    else:
        raise NotImplementedError(
            f"The adversary {adversary} has not been implemented."
        )
    return res.detach().numpy().tolist()



# if isinstance(all_gradients[aggregation_name], list):
#     gradients = all_gradients[aggregation_name]
#     print("aggregation: {}, {} attack gradient0: {}".format(
#         aggregation_name, dico['attack'], gradients[:1]
#     ))
# elif isinstance(all_gradients[aggregation_name], str):
#     if all_gradients[aggregation_name] == "ALIE":
#         gradients = _alie_attack(
#             NUM_OF_ROBOTS, num_of_byzantine, all_gradients,
#             aggregation=aggregation_name
#         )
#     elif all_gradients[aggregation_name] == "IPM":
#         gradients = _ipm_attack(
#             NUM_OF_ROBOTS, num_of_byzantine, all_gradients,
#             aggregation=aggregation_name
#         )
#     else:
#         raise NotImplementedError(
#             "This attack has not been implemented."
#         )
#     print("aggregation: {}, {} attack gradient0: {}".format(
#         aggregation_name, all_gradients[aggregation_name], gradients[:1]
#     ))
# else:
#     raise NotImplementedError(
#         "This attack has not been implemented."
#     )