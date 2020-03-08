import torch
from data_process.data_loader_provider import create_data_loaders,create_testdata_loaders
from model.model_provider import create_model, create_optimizer
from training.train_net import train_net
from testing.test_net import test_net

import argparse
import os

class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('-expID', default='default', help='Experiment ID')
        self.parser.add_argument('-data', default='../data', help='Input data folder')
        self.parser.add_argument('-nThreads', default=0, type=int, help='Number of threads')
        self.parser.add_argument('-expDir', default='../exp', help='Experiments directory')
        self.parser.add_argument('-scaleAugFactor', default=0.25, type=float, help='Scale augment factor')
        self.parser.add_argument('-rotAugProb', default=0.4, type=float, help='Rotation augment probability')
        self.parser.add_argument('-flipAugProb', default=0.5, type=float, help='Flip augment probability')
        self.parser.add_argument('-rotAugFactor', default=30, type=float, help='Rotation augment factor')
        self.parser.add_argument('-colorAugFactor', default=0.2, type=float, help='Colo augment factor')
        self.parser.add_argument('-imgSize', default=368, type=int, help='Number of threads')
        self.parser.add_argument('-hmSize', default=46, type=int, help='Number of threads')
        self.parser.add_argument('-DEBUG', type=int, default=0, help='Debug')
        self.parser.add_argument('-sigmaPAF', default=5, type=int, help='Width of PAF')
        self.parser.add_argument('-sigmaHM', default=7, type=int, help='Std. of Heatmap')
        self.parser.add_argument('-variableWidthPAF', dest='variableWidthPAF', action='store_true', help='Variable width PAF based on length of part')
        self.parser.add_argument('-dataset', default='coco', help='Dataset')
        self.parser.add_argument('-model', default='mobilenet', help='Model')
        self.parser.add_argument('-batchSize', default=1, type=int, help='Batch Size')
        self.parser.add_argument('-LR', default=1e-3, type=float, help='Learn Rate')
        self.parser.add_argument('-nEpoch', default=1000, type=int, help='Number of Epochs')
        self.parser.add_argument('-dropLR', type=float, default=50, help='Drop LR')
        self.parser.add_argument('-valInterval', type=int, default=1, help='Val Interval')
        self.parser.add_argument('-loadModel', default='model_prediction.pth', help='Load pre-trained')
        self.parser.add_argument('-mode', default='predict', help='Train mode or prediction mode, (train or predict)')
        self.parser.add_argument('-vizOut', dest='vizOut', action='store_true', help='Visualize output?')
        self.parser.add_argument('-criterionHm', default='mse', help='Heatmap Criterion')
        self.parser.add_argument('-criterionPaf', default='mse', help='PAF Criterion')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.saveDir = os.path.join(self.opt.expDir, self.opt.expID)
        if self.opt.DEBUG > 0:
            self.opt.nThreads = 1

        if not os.path.exists(self.opt.saveDir):
            os.makedirs(self.opt.saveDir)

        return self.opt
    
def main():
    opt = Opts().parse()    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders
    if opt.mode == 'train':
        train_loader, val_loader = create_data_loaders(opt)
    elif opt.mode == 'predict':
        test_loader = create_testdata_loaders(opt)

    # Create nn
    model, criterion_hm, criterion_paf = create_model(opt)
    model = model.to(device)
    criterion_hm = criterion_hm.to(device)
    criterion_paf = criterion_paf.to(device)
    # Create optimizer
    optimizer = create_optimizer(opt, model)

    # train/ predict
    if opt.mode == 'train':
        train_net(train_loader, val_loader, model, criterion_hm, criterion_paf, optimizer, opt.nEpoch,
                  opt.valInterval, opt.LR, opt.dropLR, opt.saveDir, opt.vizOut)
    elif opt.mode == 'predict':
       test_net(test_loader, model, opt)


if __name__ == '__main__':
    main()
