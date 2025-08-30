# yo.py
import torch
import torch.nn as nn
from ultralytics import YOLO

def slim_conv(conv: nn.Conv2d, keep_idx):
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=len(keep_idx),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )
    new_conv.weight.data = conv.weight.data[keep_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_idx].clone()
    return new_conv

def slim_bn(bn: nn.BatchNorm2d, keep_idx):
    new_bn = nn.BatchNorm2d(len(keep_idx))
    new_bn.weight.data = bn.weight.data[keep_idx].clone()
    new_bn.bias.data   = bn.bias.data[keep_idx].clone()
    new_bn.running_mean = bn.running_mean[keep_idx].clone()
    new_bn.running_var  = bn.running_var[keep_idx].clone()
    return new_bn

def prune_module(module: nn.Module, ratio=0.5):
    for name, child in list(module.named_children()):
        # Recurse
        prune_module(child, ratio)

        if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
            # Rank channels by importance (L1 norm)
            w = child.weight.detach().abs().mean(dim=(1,2,3))
            n_keep = max(1, int(len(w) * (1 - ratio)))
            keep_idx = torch.argsort(w, descending=True)[:n_keep]

            # Replace Conv
            new_conv = slim_conv(child, keep_idx)
            setattr(module, name, new_conv)

            # Try replacing next BN
            next_name, next_mod = None, None
            siblings = list(module.named_children())
            for i, (n, m) in enumerate(siblings):
                if n == name and i + 1 < len(siblings):
                    if isinstance(siblings[i+1][1], nn.BatchNorm2d):
                        next_name, next_mod = siblings[i+1]
                        break
            if next_mod is not None:
                new_bn = slim_bn(next_mod, keep_idx)
                setattr(module, next_name, new_bn)

def prune_model(model: nn.Module, ratio=0.5):
    prune_module(model, ratio)
    return model

if __name__ == "__main__":
    model_path = "best.pt"
    y = YOLO(model_path)
    net = y.model

    before = sum(p.numel() for p in net.parameters())
    net = prune_model(net, ratio=0.7)  # try 0.3 ~ 0.7
    after  = sum(p.numel() for p in net.parameters())

    print(f"Params: {before/1e6:.2f}M -> {after/1e6:.2f}M")

    torch.save({"model": net.state_dict()}, "manual_pruned.pt")
    print("Saved: manual_pruned.pt")
