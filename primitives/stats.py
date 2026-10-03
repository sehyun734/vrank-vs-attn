import torch
from torch import Tensor
from torch.nn.functional import pad


def calc_basis(
    value: Tensor,
    energy_ratio: float,
) -> Tensor:
    # 실제 value는 noise 때문에 거의 항상 full rank.
    # 논문대로 energy 누적 합이 energy_ratio를 채우는 방향으로만.
    _, singular, basis = torch.linalg.svd(value.float(), full_matrices=False)
    energy = singular.square()
    return basis[: int((energy.cumsum(0) / energy.sum() < energy_ratio).sum()) + 1]


def calc_new_ratio(
    value: Tensor,
    basis: Tensor,
) -> float:
    # 논문대로 basis 밖 성분의 norm을 전체 norm으로 나눔.
    value = value.float()
    return ((value - value @ basis.T @ basis).norm() / value.norm()).item()


def calc_entropy(
    prob: Tensor,
) -> Tensor:
    # sink나 prefix를 잘라낸 attn은 합이 1보다 작으므로 다시 정규화.
    prob = prob.float() / prob.float().sum(-1, keepdim=True)
    return -(prob * prob.clamp_min(1e-12).log()).sum(-1)


def calc_jsd(
    prob_a: Tensor,
    prob_b: Tensor,
) -> Tensor:
    prob_a = prob_a.float() / prob_a.float().sum(-1, keepdim=True)
    prob_b = prob_b.float() / prob_b.float().sum(-1, keepdim=True)
    return calc_entropy((prob_a + prob_b) / 2) - (calc_entropy(prob_a) + calc_entropy(prob_b)) / 2


def calc_hidden_rank(
    hiddens: list[Tensor],
    mass_ratio: float,
) -> int:
    # 논문대로 각 sample의 마지막 레이어 hidden의 singular 정규화 후 평균.
    # 정규화는 sample마다 singular 크기가 다르고 누적합 비교에서는 합이 1이어야 하기 때문.
    # 또한 sample마다 길이가 다르므로 0 padding.
    # value의 새로운 방향이 마지막 레이어까지 이어지는지 확인하기 위함.
    spectra = []
    for hidden in hiddens:
        singular = torch.linalg.svdvals(hidden.float())
        spectra.append(pad(singular / singular.sum(), (0, hidden.shape[-1] - len(singular))))
    spectrum = torch.stack(spectra).mean(0)
    return int((spectrum.cumsum(0) < mass_ratio).sum()) + 1
