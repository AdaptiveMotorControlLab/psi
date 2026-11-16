import torch

def derangement(
    n: int,
    groups: torch.Tensor | None = None,
    device : torch.device | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    assert n > 1, "Derangement is only defined for n > 1"

    if groups is None:
        identity = torch.arange(n, device=device)

        while True:
            donor = torch.randperm(n, device=device, generator=generator)
            if not torch.any(donor == identity):
                return donor

    groups = groups.to(device=device)
    if groups.shape != (n,):
        raise ValueError(f"Groups tensor must have shape ({n},), got {groups.shape}.")

    donor = torch.empty(n, dtype=torch.long, device=device)
    for group in groups.unique():
        members = torch.where(groups == group)[0]
        if len(members) < 2:
            raise ValueError(f"Cannot derange a singleton transport group.")

        cycle = members[torch.randperm(len(members), device=device, generator=generator)]
        donor[cycle] = cycle.roll(1)

    return donor
