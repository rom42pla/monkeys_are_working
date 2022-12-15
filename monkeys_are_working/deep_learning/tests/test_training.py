def test_devices():
    from typing import List, Dict, Any, Union

    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import TensorDataset, DataLoader

    from ..training import Trainer

    # generates dummy inputs
    dataset_train, dataset_val = TensorDataset(torch.rand([4, 3, 512, 512])), \
        TensorDataset(torch.rand([2, 3, 512, 512]))
    dataloader_train, dataloader_val = DataLoader(dataset_train, batch_size=2), \
        DataLoader(dataset_val, batch_size=2)

    # defines the step and loss functions
    def step_fn(
            model: nn.Module,
            batch: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(batch, list):
            batch = torch.cat(batch, dim=0)
        return model["decoder"](
            model["encoder"](batch)
        )

    def loss_fn(
            target: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
            input: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]],
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(target, list):
            target = torch.cat(input, dim=0)
        if isinstance(input, list):
            input = torch.cat(input, dim=0)
        return F.mse_loss(input, target)

    for device in ["cuda", "cpu"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        # builds the model and the optimizer
        model = nn.ModuleDict({
            "encoder": nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=256,
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(256),
                nn.GELU(),
            ),
            "decoder": nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                ),
                nn.Conv2d(in_channels=256, out_channels=3,
                          kernel_size=1, stride=1, bias=True),
                nn.Sigmoid(),
            ),
        }).to(device)
        optimizer = AdamW(model.parameters())

        # tries 2 epochs
        Trainer().fit(model=model, optimizer=optimizer,
                      step_fn=step_fn,
                      loss_fn=loss_fn,
                      dataloader_train=dataloader_train,
                      dataloader_val=dataloader_val,
                      device=device,
                      max_epochs=2)
