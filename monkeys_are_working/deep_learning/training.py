import gc

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from os import makedirs
from os.path import isdir
from typing import Optional, Union, Dict

import torch
from torch import nn, autocast

from monkeys_are_working.deep_learning.common import parse_device, transfer_batch_to_device


class Trainer(nn.Module):
    def __init__(
            self,
            logs_path: Optional[str] = None,
    ):
        super().__init__()

        assert logs_path is None or isinstance(logs_path, str)
        self.logs_path = logs_path
        if self.logs_path is not None and not isdir(self.logs_path):
            makedirs(self.logs_path)

        self.logs_queue = {}
        self.logs = []
        self.free_memory()

    def fit(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            step_fn,
            loss_fn,
            dataloader_train,
            dataloader_val: Optional[DataLoader],
            max_epochs: int = 50,
            precision: int = 32,
            device: str = "auto"
    ) -> Dict[str, Union[nn.Module]]:
        device = parse_device(device)
        model = model.to(device)

        assert precision in {16, 32}

        # todo fix mixed precision speed
        scaler = torch.cuda.amp.GradScaler(enabled=False if precision == 32 else True)
        for epoch in range(max_epochs):
            epoch_stats = {
                "train": {},
                "val": {},
            }

            # training
            epoch_stats["train"] = self.train_epoch(
                model=model,
                optimizer=optimizer,
                step_fn=step_fn,
                loss_fn=loss_fn,
                scaler=scaler,
                dataloader=dataloader_train,
                precision=precision,
                device=device,
                i_epoch=epoch,
            )

            # validation
            if dataloader_val:
                epoch_stats["val"] = self.val_epoch(
                    model=model,
                    step_fn=step_fn,
                    loss_fn=loss_fn,
                    dataloader=dataloader_val,
                    precision=precision,
                    device=device,
                    i_epoch=epoch,
                )

            for phase in epoch_stats.keys():
                self.logs += [{
                    "phase": phase,
                    "epoch": epoch,
                    **{
                        k: torch.stack(v).mean().detach().item()
                        for k, v in epoch_stats[phase].items()
                    }
                }]
            self.free_memory()
        return {
            "model": model,
            "optimizer": optimizer,
        }

        # if (epoch == max_epochs - 1) or ((epoch != 0) and (epoch % plot_interval == 0)):
        #     self.plot_training_stats()

    # def encode_labels(self, labels: List[str]):
    #     labels_encoded = torch.asarray([self.labels[label] for label in labels]).to(self.device).long()
    #     return labels_encoded

    def train_epoch(
            self,
            model,
            optimizer,
            step_fn,
            loss_fn,
            scaler,
            dataloader,
            precision: int = 32,
            device: str = "auto",
            i_epoch: Optional[int] = None,
    ):
        model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        stats = {}
        for i_batch, batch in pbar:
            # transfer the inputs to the correct device
            batch = transfer_batch_to_device(batch, device=device)

            with autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16,
                          enabled=False if precision == 32 else True):
                outs = step_fn(model, batch)
            loss = loss_fn(outs, batch)
            loss = loss["loss"] if isinstance(loss, dict) else loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # updates the stats
            if isinstance(outs, torch.Tensor):
                if "outs" not in stats:
                    stats["outs"] = [loss]
                else:
                    stats["outs"] += [loss]
            elif isinstance(outs, dict):
                for k in outs.keys():
                    if isinstance(outs[k], torch.Tensor):
                        if k not in stats:
                            stats[k] = [outs[k]]
                        else:
                            stats[k] += [outs[k]]
            else:
                raise Exception(f"unrecognized model outputs of type '{type(outs)}'")

            # updates the progress bar
            pbar.set_description(
                f"epoch {i_epoch if isinstance(i_epoch, int) else 'undefined'} (train)\t\t" +
                ", ".join(f"{k}={torch.stack(v).mean():.3f}"
                          for k, v in stats.items() if len(v[0].shape) == 0))

            self.free_memory()
        return stats

    def val_epoch(
            self,
            model,
            step_fn,
            loss_fn,
            dataloader,
            precision: int = 32,
            device: str = "auto",
            i_epoch: Optional[int] = None,
    ):
        model.eval()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        stats = {}
        for i_batch, batch in pbar:
            # transfer the inputs to the correct device
            batch = transfer_batch_to_device(batch, device=device)

            # forward pass
            with torch.no_grad(), autocast(device_type=device,
                                           dtype=torch.float16 if device == "cuda" else torch.bfloat16,
                                           enabled=False if precision == 32 else True):
                outs = step_fn(model, batch)
            loss = loss_fn(outs, batch)

            # updates the stats
            if isinstance(outs, torch.Tensor):
                if "outs" not in stats:
                    stats["outs"] = [loss]
                else:
                    stats["outs"] += [loss]
            elif isinstance(outs, dict):
                for k in outs.keys():
                    if isinstance(outs[k], torch.Tensor):
                        if k not in stats:
                            stats[k] = [outs[k]]
                        else:
                            stats[k] += [outs[k]]
            else:
                raise Exception(f"unrecognized model outputs of type '{type(outs)}'")

            # updates the progress bar
            pbar.set_description(
                f"epoch {i_epoch if isinstance(i_epoch, int) else 'undefined'} (val)\t\t" +
                ", ".join(f"{k}={torch.stack(v).mean():.3f}"
                          for k, v in stats.items() if len(v[0].shape) == 0))

            self.free_memory()
        return stats

    def get_logs(self) -> pd.DataFrame:
        df = pd.DataFrame(self.logs)
        return df

    def free_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
