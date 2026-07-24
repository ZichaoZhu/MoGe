"""Minimal dense-PyTorch DDP probe used to separate NCCL from SpConv failures."""

import os
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs


def main() -> None:
    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(backend=os.environ.get("MOGE3_DDP_BACKEND", "nccl"))
        ]
    )
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    inputs = torch.randn(4, 8, device=accelerator.device)
    loss = model(inputs).square().mean()
    accelerator.backward(loss)
    optimizer.step()
    gathered = accelerator.gather(loss.detach()[None])
    if accelerator.is_main_process:
        print(
            {
                "world_size": accelerator.num_processes,
                "losses": gathered.cpu().tolist(),
            }
        )


if __name__ == "__main__":
    main()
