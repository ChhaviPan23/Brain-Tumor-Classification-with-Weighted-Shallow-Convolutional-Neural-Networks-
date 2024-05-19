import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import pickle
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from model import NetworkImageNet as Network
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
  init_process_group(backend="nccl")
  torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--weights',type=str,default=None,help='location of weights of fine-tuned network')
parser.add_argument('--num_classes',type=int,default=3,help='number of target classes')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = args.num_classes


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.to(torch.int64), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def add_noise(img_tensor, mean=0, std=1.5):
  noise = torch.randn_like(img_tensor) * std + mean
  noisy_img_tensor = img_tensor + noise
  return noisy_img_tensor

class BrainTumorDataset(Dataset):
  def __init__(self, images, labels):
    # images
    self.X = images
    # labels
    self.y = labels

    # Transformation for converting original image array to an image and then convert it to a tensor
    self.transform = transforms.Compose([transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    self.transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(90),
        transforms.ToTensor()
    ])

    self.transform2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(180),
        transforms.ToTensor()
    ])

    self.transform3 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.functional.vflip,
        transforms.ToTensor()
    ])

    self.transform4 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.functional.vflip,
        transforms.ToTensor()
    ])

    self.transform5 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_noise(x))
    ])


  def __len__(self):
    # return length of image samples
    return 6*len(self.X)

  def __getitem__(self, idx):
    idx = idx//6
    r = idx%6
    if(r==0):
      data = self.transform(self.X[idx])
    if(r==1):
      data = self.transform1(self.X[idx])
    if(r==2):
      data = self.transform2(self.X[idx])
    if(r==3):
      data = self.transform3(self.X[idx])
    if(r==4):
      data = self.transform4(self.X[idx])
    if(r==5):
      data = self.transform5(self.X[idx])

    labels = torch.zeros(3, dtype=torch.float32)
    labels[int(self.y[idx])-1] = 1.0

    return data,labels
  


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  ddp_setup()

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
  model = model.to(int(os.environ["LOCAL_RANK"]))
  model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    model.module.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  file = open("training_data.pickle",'rb')
  training_data = pickle.load(file)
  file.close()

  Xt = []
  yt = []
  features = None
  labels = None
  label = []

  for features,labels in training_data:
    Xt.append(features)
    yt.append(labels)


  X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.32, shuffle=True)
  X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
  train_set = BrainTumorDataset(X_train, y_train)
  valid_set = BrainTumorDataset(X_valid, y_valid)
  test_set = BrainTumorDataset(X_test, y_test)

  train_queue = DataLoader(train_set, batch_size=16, shuffle=False, pin_memory=True, num_workers=2, sampler=DistributedSampler(train_set))
  valid_queue = DataLoader(valid_set, batch_size=16, shuffle=False, pin_memory=True, num_workers=2, sampler=DistributedSampler(valid_set))
  test_queue = DataLoader(test_set, batch_size=16, shuffle=False, pin_memory=True, num_workers=2,sampler = DistributedSampler(test_set))
  vgg19_model = torch.nn.Sequential(torchvision.models.vgg19(weights=None).to(int(os.environ["LOCAL_RANK"])), torch.nn.Linear(1000, 3))
  weights_path = "weights_trained.pth" 
  checkpoint = torch.load(weights_path)
  vgg19_model.load_state_dict(checkpoint)
  vgg19_model.eval()

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

  best_acc_top1 = 0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer,vgg19_model)
    logging.info('train_acc %f', train_acc)

    valid_acc_top1, valid_obj = infer(valid_queue, model, criterion,vgg19_model)
    logging.info('valid_acc_top1 %f', valid_acc_top1)

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True

    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': best_acc_top1,
      'optimizer' : optimizer.state_dict(),
      }, is_best, args.save)
  
  test_acc_top1, valid_obj = infer(test_queue, model, criterion,vgg19_model)
  logging.info('test_acc_top1 %f', test_acc_top1)
   
  destroy_process_group() 

def train(train_queue, model, criterion, optimizer,vgg_model):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  #top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    target = target.cuda(non_blocking=True)
    input = input.cuda()
    input = Variable(input)
    target = Variable(target)
    
    with torch.no_grad():
      feature_maps = vgg_model[0].features[1](vgg_model[0].features[0](input))
      feature_maps = vgg_model[0].features[4](vgg_model[0].features[3](vgg_model[0].features[2](feature_maps)))
      feature_maps = vgg_model[0].features[6](vgg_model[0].features[5](feature_maps))
      feature_maps = vgg_model[0].features[9](vgg_model[0].features[8](vgg_model[0].features[7](feature_maps)))
      input1 = feature_maps
      feature_maps = vgg_model[0].features[11](vgg_model[0].features[10](feature_maps))
      feature_maps = vgg_model[0].features[13](vgg_model[0].features[12](feature_maps))
      feature_maps = vgg_model[0].features[15](vgg_model[0].features[14](feature_maps))
      feature_maps = vgg_model[0].features[18](vgg_model[0].features[17](vgg_model[0].features[16](feature_maps)))
      input2 = feature_maps
    
    optimizer.zero_grad()
    logits, logits_aux = model(input1,input2)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm(model.module.parameters(), args.grad_clip)
    optimizer.step()

    prec1 = utils.accuracy(logits, target)
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1, n)
    #top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion,vgg_model):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  #top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(non_blocking=True)
    
    with torch.no_grad():
      feature_maps = vgg_model[0].features[1](vgg_model[0].features[0](input))
      feature_maps = vgg_model[0].features[4](vgg_model[0].features[3](vgg_model[0].features[2](feature_maps)))
      feature_maps = vgg_model[0].features[6](vgg_model[0].features[5](feature_maps))
      feature_maps = vgg_model[0].features[9](vgg_model[0].features[8](vgg_model[0].features[7](feature_maps)))
      input1 = feature_maps
      feature_maps = vgg_model[0].features[11](vgg_model[0].features[10](feature_maps))
      feature_maps = vgg_model[0].features[13](vgg_model[0].features[12](feature_maps))
      feature_maps = vgg_model[0].features[15](vgg_model[0].features[14](feature_maps))
      feature_maps = vgg_model[0].features[18](vgg_model[0].features[17](vgg_model[0].features[16](feature_maps)))
      input2 = feature_maps
    logits, _ = model(input1,input2)
    loss = criterion(logits, target)

    prec1= utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1, n)
    #top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 
