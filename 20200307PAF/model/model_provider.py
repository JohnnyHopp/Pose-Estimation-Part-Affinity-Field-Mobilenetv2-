from .vgg import VGG
from .mobilenet import MobileNet
from .paf_model import PAFModel
import torch.nn as nn
import torch

def parse_criterion(criterion):
    if criterion == 'l1':
        return nn.L1Loss(size_average = False)
    elif criterion == 'mse':
        return nn.MSELoss(size_average = False)
    else:
        raise ValueError('Criterion ' + criterion + ' not supported')

def create_model(opt):
    if opt.model == 'mobilenet':
        backend = MobileNet()
        backend_feats = 128
    else:
        raise ValueError('Model ' + opt.model + ' not available.')
    model = PAFModel(backend,backend_feats,n_joints=19,n_paf=34,n_stages=6)
    if not opt.loadModel=='none':
#        model = torch.load(opt.loadModel, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(opt.loadModel, map_location=lambda storage, loc: storage))
        print('Loaded model from '+opt.loadModel)
    criterion_hm = parse_criterion(opt.criterionHm)
    criterion_paf = parse_criterion(opt.criterionPaf)
    return model, criterion_hm, criterion_paf

    
def create_test_model(opt):
    if opt.model == 'mobilenet':
        backend = MobileNet()
        backend_feats = 128
    else:
        raise ValueError('Model ' + opt.model + ' not available.')
    model = PAFModel(backend,backend_feats,n_joints=14,n_paf=24,n_stages=6)
    if not opt.loadModel=='none':
        model = torch.load(opt.loadModel, map_location=lambda storage, loc: storage)
        print('Loaded model from '+opt.loadModel)
    return model

def create_optimizer(opt, model):
    return torch.optim.Adam(model.parameters(), opt.LR)