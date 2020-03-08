import torch.nn as nn
from .helper import init, make_standard_block
import torch

from .MobileNetV2 import InvertedResidual


class PAFModel(nn.Module):
    def __init__(self, backend, backend_outp_feats, n_joints, n_paf, n_stages=6): #backend_outp_feats=128,n_joints=19,n_paf=34
        super(PAFModel, self).__init__()
        assert (n_stages > 0)
        self.backend = backend        
        for i in range(n_stages):
            if i == 0:
                stages = [Stage_paf(backend_outp_feats, n_paf, True)]
            elif i < n_stages-1:
                stages.append(Stage_paf(backend_outp_feats, n_paf, False))
            else:
                stages.append(Stage_joints(backend_outp_feats, n_paf, n_joints))
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        img_feats = self.backend(x)
        cur_feats = img_feats
        paf_outs = []
        for i, stage in enumerate(self.stages):
            if i < len(self.stages)-1:
                paf_out = stage(cur_feats)            
                paf_outs.append(paf_out)
                cur_feats = torch.cat([img_feats, paf_out], 1)
            else:
                heatmap_out = stage(cur_feats)
        return paf_outs, heatmap_out

class Stage_paf(nn.Module):
    def __init__(self, backend_outp_feats, n_paf, stage1):
        super(Stage_paf, self).__init__()
        inp_feats = backend_outp_feats
        if stage1:
            self.block1 = make_block_stage1(inp_feats, n_paf)
        else:
            inp_feats = backend_outp_feats + n_paf
            self.block1 = make_block_stage2(inp_feats, n_paf)
        init(self.block1)

    def forward(self, x):
        y1 = self.block1(x)
        return y1

class Stage_joints(nn.Module):
    def __init__(self, backend_outp_feats, n_paf, n_joints):
        super(Stage_joints, self).__init__()
        inp_feats = backend_outp_feats + n_paf
        self.block1 = make_block_stage2(inp_feats, n_joints)
        init(self.block1)

    def forward(self, x):
        y1 = self.block1(x)
        return y1

def make_block_stage1(inp_feats, output_feats):
    layers = [InvertedResidual(inp_feats, 128, 1, 6), #inp_feats:128
              InvertedResidual(128, 128, 1, 6),
              InvertedResidual(128, 128, 1, 6),
              make_standard_block(128, 512, 1, 1, 0)]
    layers += [nn.Conv2d(512, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)

def make_block_stage2(inp_feats, output_feats):
    layers = [InvertedResidual(inp_feats, 128, 1, 1, kernel_size=7),
              InvertedResidual(128, 128, 1, 1, kernel_size=7),
              InvertedResidual(128, 128, 1, 1, kernel_size=7),
              InvertedResidual(128, 128, 1, 1, kernel_size=7),
              InvertedResidual(128, 128, 1, 1, kernel_size=7),
              make_standard_block(128, 128, 1, 1, 0)]
    layers += [nn.Conv2d(128, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)