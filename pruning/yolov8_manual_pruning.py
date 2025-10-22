# prune_best_bottlenecks.py
# Dependency-safe pruning for YOLOv8 on Ultralytics 8.3.209 (no extra packages).
# Strategy: prune only the internal mid-channels of Bottleneck blocks.
# - Keeps every module's external I/O channels unchanged => graph stays valid.
# - Reduces params/FLOPs meaningfully without concat/skip mismatches.

import torch
import torch.nn as nn
from ultralytics import YOLO

PRUNE_RATIO = 0.5          # prune 50% of mid-channels inside each Bottleneck (adaptable)
IMG_SIZE    = 320
INPUT_PT    = r"pruning/EE_backbone_neck.pt"
OUTPUT_PT   = r"pruning/EE_backbone_neck_pruned.pt"

# Ultralytics Conv wrapper: has .conv (nn.Conv2d), .bn (nn.BatchNorm2d), .act
def _slice_conv2d(conv: nn.Conv2d, keep_out=None, keep_in=None) -> nn.Conv2d:
    """Create a new Conv2d with selected out/in channels from an existing Conv2d."""
    w = conv.weight.data
    if keep_in is not None:
        w = w[:, keep_in, :, :]
    if keep_out is not None:
        w = w[keep_out, :, :, :]

    new = nn.Conv2d(
        in_channels=w.shape[1],
        out_channels=w.shape[0],
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups if conv.groups == 1 else 1,  # keep safe (YOLOv8 uses groups=1 here)
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )
    new.weight = nn.Parameter(w.clone())
    if conv.bias is not None:
        if keep_out is None:
            new.bias = nn.Parameter(conv.bias.data.clone())
        else:
            new.bias = nn.Parameter(conv.bias.data[keep_out].clone())
    return new

def _slice_bn(bn: nn.BatchNorm2d, keep_idx) -> nn.BatchNorm2d:
    """Slice BatchNorm to a subset of channels."""
    new = nn.BatchNorm2d(len(keep_idx))
    new.weight.data = bn.weight.data[keep_idx].clone()
    new.bias.data   = bn.bias.data[keep_idx].clone()
    new.running_mean = bn.running_mean[keep_idx].clone()
    new.running_var  = bn.running_var[keep_idx].clone()
    return new

def prune_bottleneck_mid(bneck, ratio=0.5):
    """
    Prune the internal mid-channels of a Ultralytics Bottleneck:
      - cv1: 1x1 Conv expands from Cin -> Cmid
      - cv2: 3x3 Conv from Cmid -> Cout (often Cout=Cin)
    Keep Bottleneck I/O (Cin, Cout) unchanged; only shrink Cmid.
    """
    # Access inner convs/bn via Conv wrapper
    cv1, cv2 = bneck.cv1, bneck.cv2
    c_mid = cv1.conv.out_channels  # original mid width
    if c_mid <= 2:                  # nothing to prune
        return False, c_mid, c_mid

    # Importance per INPUT channel of cv2 (shape [in]) => mean over out,k,k
    # cv2.conv.weight: [out_channels, in_channels, k, k]
    W = cv2.conv.weight.detach()
    importance = W.abs().mean(dim=(0, 2, 3))  # per in_channel of cv2
    n_keep = max(1, int(c_mid * (1 - ratio)))
    keep_idx = torch.argsort(importance, descending=True)[:n_keep]
    keep_idx = keep_idx.sort().values  # keep order stable

    # Slice cv1 (OUT = mid) and its BN
    new_cv1_conv = _slice_conv2d(cv1.conv, keep_out=keep_idx, keep_in=None)
    new_cv1_bn   = _slice_bn(cv1.bn, keep_idx)
    cv1.conv = new_cv1_conv
    cv1.bn   = new_cv1_bn

    # Slice cv2 (IN = mid); OUT stays same
    new_cv2_conv = _slice_conv2d(cv2.conv, keep_in=keep_idx, keep_out=None)
    cv2.conv = new_cv2_conv
    # cv2.bn stays same (its num_features = Cout, unchanged)

    return True, c_mid, n_keep

def prune_model_bottlenecks(model: nn.Module, ratio=0.5):
    """
    This function , pruning mid-channels inside each Bottleneck.
    Also descends into C2f blocks to reach their Bottlenecks (m list).
    """
    from ultralytics.nn.modules.block import Bottleneck
    changed = 0
    before_after = []

    def visit(mod: nn.Module):
        nonlocal changed, before_after
        for name, child in mod.named_children():
            # Dive into C2f and others
            visit(child)
            # If this is a Bottleneck, prune its mid channels
            if isinstance(child, Bottleneck):
                ok, old_cmid, new_cmid = prune_bottleneck_mid(child, ratio)
                if ok:
                    changed += 1
                    before_after.append((old_cmid, new_cmid))

    visit(model)
    return changed, before_after

def main():
    y = YOLO(INPUT_PT)
    net = y.model.eval()
    params_before = sum(p.numel() for p in net.parameters())/1e6
    print(f"Params before: {params_before:.2f}M")

    changed, cmids = prune_model_bottlenecks(net, ratio=PRUNE_RATIO)
    params_after = sum(p.numel() for p in net.parameters())/1e6

    print(f"Pruned bottlenecks: {changed}")
    if cmids:
        total_delta = sum(o - n for (o, n) in cmids)
        print(f"Total mid-channel reduction across bottlenecks: {int(total_delta)}")

    print(f"Params: {params_before:.2f}M → {params_after:.2f}M")

    # Forward sanity check (full model, preserves graph)
    with torch.no_grad():
        _ = net(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
    print("Forward pass OK")

    # Save YOLO-compatible checkpoint
    y.model = net
    y.save(OUTPUT_PT)
    print(f"Saved: {OUTPUT_PT}")

if __name__ == "__main__":
    main()
