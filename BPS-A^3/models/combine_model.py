import torch as th
import torch.nn as nn
from .EGAT import DiEGAT
from .models_2D2 import TrafficTransformer
from functools import partial

class triatten(nn.Module):
    def __init__(self, num_class=12, m=0.7):
        super().__init__()
        self.m = m
        self.nb_class = num_class
        self.pre_model = TrafficTransformer(
        img_size=40, patch_size=2, in_chans=1, embed_dim=192, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_class)
        self.feat_dim = self.pre_model.head.in_features
        self.gnn = DiEGAT(ndim_in=192, ndim_out=192, edim=192, dropout=0.2, num_class=num_class)

    def forward(self, g, idx):
        if self.training:
            cls_feats = self.pre_model.forward_features(g.edata['bytes'][idx])
            g.edata['cls_feat'][idx] = cls_feats
        else:
            cls_feats = g.edata['cls_feat'][idx]
        # cls_logit = self.classifier(cls_feats)
        cls_logit = self.pre_model.head(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        sage_logit = self.gnn(g, g.ndata['node_feat'], g.edata['cls_feat'])[idx]
        sage_pred = th.nn.Softmax(dim=1)(sage_logit)
        pred = (sage_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred




