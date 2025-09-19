import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv , Concat
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect

class YOLOv8n_EE(nn.Module):
    """
    YOLOv8n model with Early-Exit branches to allow fast inference when confident detections occur at shallower layers.
    Args:
        model_yaml (string): path to model's yaml file we will define model's nc, yaml, stride, and names by this file.
    """
    def __init__(self,nc , model_yaml):

        model = YOLO(model_yaml)
        #we should pass the yaml, stride, and names of our model to new model again aim at preserving its vital values for training
        self.nc = nc
        self.yaml = model.yaml
        self.stride = model.stride
        self.names = model.names
        #outputs is an array which will save early exit outputs count for each layer aim to analyse them and data flow.
        self.outputs = [0,0,0,0,0,0]

        super().__init__()
        # Backbone: progressively extract features and reduce spatial resolution

        #First Block -----------------------------------------------------------
        # Stage 0: 3×3 conv, stride 2, reduces H×W by 2 → feature map P1/2
        self.conv0 = Conv(3, 64, k=3, s=2)
        # Stage 1: 3×3 conv, stride 2, reduces H×W by 2 → P2/4
        self.conv1 = Conv(64, 128, k=3, s=2)
        # Feature extraction block (C2f) at P2 resolution
        self.c2f2  = C2f(128, 128, n=3, shortcut=True, g=1, e=1.0)
        # Early-exit detection head at P2
        self.exit0 = Detect(nc=self.nc, ch=[128])

        #Second Block -----------------------------------------------------------
        # Stage 2: conv → P3/8
        self.conv3 = Conv(128, 256, k=3, s=2)
        # C2f block at P3
        self.c2f4  = C2f(256, 256, n=6, shortcut=True, g=1, e=1.0)
        # Early-exit detection head at P3
        self.exit1 = Detect(nc=self.nc, ch=[256])

        #Third Block -------------------------------------------------------------
        # Stage 3: conv → P4/16
        self.conv5 = Conv(256, 512, k=3, s=2)
        # C2f block at P4
        self.c2f6  = C2f(512, 512, n=6, shortcut=True, g=1, e=1.0)
        # Early-exit detection head at P4
        self.exit2 = Detect(nc=self.nc, ch=[512])

        #Final Block of Backbone --------------------------------------------------
        # Stage 4: conv → P5/32
        self.conv7 = Conv(512, 1024, k=3, s=2)
        # C2f block at P5 for high-level features
        self.c2f8  = C2f(1024,1024, n=3, shortcut=True, g=1, e=1.0)
        self.exit3 = Detect(nc=self.nc, ch=[1024])
        # Spatial Pyramid Pooling Fast (SPPF) combines multi-scale context
        self.sppf9 = SPPF(1024,1024, k=5)

        # Neck + Final Head: feature fusion and multi-scale detection
        # Upsample P5 → P4 and fuse with P4 features
        self.up10    = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat11= Concat(dimension=1)
        self.c2f12   = C2f(1024+512, 512, n=3, shortcut=True, g=1, e=0.5)
        self.exit4 = Detect(nc=self.nc, ch=[512])

        # Upsample fused P4 → P3 and fuse with P3 features
        self.up13    = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat14= Concat(dimension=1)
        self.c2f15   = C2f(512+256, 256, n=3, shortcut=True, g=1, e=0.5)

        # Downsample P3 → P4 and fuse with earlier P4 features
        self.down16  = Conv(256, 256, k=3, s=2)
        self.concat17= Concat(dimension=1)
        self.c2f18   = C2f(256+512, 512, n=3, shortcut=True, g=1, e=0.5)

        # Downsample P4 → P5 and fuse with P5 features
        self.down19  = Conv(512, 512, k=3, s=2)
        self.concat20= Concat(dimension=1)
        self.c2f21   = C2f(512+1024,1024, n=3, shortcut=True, g=1, e=0.5)

        # Final detection head on three scales: P3, P4, P5
        self.detect_final = Detect(nc=self.nc, ch=[256,512,1024])

    def forward(self, x, exit_conf_threshold=0.25):
        """
        Forward pass with optional early exits.
        Args:
            x (Tensor): Input image tensor.
            exit_conf_threshold (float, optional): Confidence threshold for early exit.
        Returns:
            Depending on training or inference mode, returns tuple of all outputs or a single detection.
        """
        # --- Backbone forward ---
        x0 = self.conv0(x)                    # P1
        x1 = self.conv1(x0)                   # P2
        x2 = self.c2f2(x1)                    # feature at P2
        pred0 = self.exit0([x2])              # early-exit head 0

        x3 = self.conv3(x2)                   # P3
        x4 = self.c2f4(x3)
        pred1 = self.exit1([x4])              # early-exit head 1

        x5 = self.conv5(x4)                   # P4
        x6 = self.c2f6(x5)
        pred2 = self.exit2([x6])              # early-exit head 2

        x7 = self.conv7(x6)                   # P5
        x8 = self.c2f8(x7)
        x9 = self.sppf9(x8)                   # enriched P5
        pred3 = self.exit3([x9])              # early-exit head 3

        # --- Neck: top-down and bottom-up feature fusion ---
        u10 = self.up10(x9)                   # upsample P5 → P4
        f11 = self.concat11([u10, x6])       # fuse with P4
        x12= self.c2f12(f11)
        pred4 = self.exit4([x12])              # early-exit neck 1

        u13 = self.up13(x12)                  # upsample → P3
        f14= self.concat14([u13, x4])        # fuse with P3
        x15= self.c2f15(f14)

        d16= self.down16(x15)                 # downsample → P4
        f17= self.concat17([d16, x12])       # fuse
        x18= self.c2f18(f17)

        d19= self.down19(x18)                 # downsample → P5
        f20= self.concat20([d19, x9])        # fuse
        x21= self.c2f21(f20)

        # Final detection on three feature levels
        final = self.detect_final([x15, x18, x21])

        if self.training:
            # Return all heads for loss computation during training
            return (pred0, pred1, pred2, pred3, pred4, final)

        # Inference: check early exits if threshold provided
        if exit_conf_threshold is not None:
            # If confidence at first exit > threshold, return early
            if pred0[0][:,4].max() > exit_conf_threshold:
                self.outputs[0] -=-1
                return pred0
            if pred1[0][:,4].max() > exit_conf_threshold:
                self.outputs[1] -=-1
                return pred1
            if pred2[0][:,4].max() > exit_conf_threshold:
                self.outputs[2] -=-1
                return pred2
            if pred3[0][:,4].max() > exit_conf_threshold:
                self.outputs[3] -=-1
                return pred3
            if pred4[0][:,4].max() > exit_conf_threshold:
                self.outputs[4] -=-1
                return pred4
        # Otherwise, return final multi-scale predictions
        self.outputs[4] -=-1
        return final
