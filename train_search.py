import os
import sys
import time
import glob
import pickle
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.models 
from torchvision import transforms
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
  init_process_group(backend="nccl")
  torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

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


parser = argparse.ArgumentParser("CHHAVA")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--model_path', type=str, default='MODEL_SAVED', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.25, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--weights',type=str,default=None,help='location of weights of fine-tuned network')
parser.add_argument('--num_classes',type=int,default=3,help='number of target classes')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def add_noise(img_tensor, mean=0, std=1.5):
  noise = torch.randn_like(img_tensor) * std + mean
  noisy_img_tensor = img_tensor + noise
  return noisy_img_tensor

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
  logging.info('gpu device = %d' % int(os.environ["LOCAL_RANK"]))
  logging.info("args = %s", args)

  classes = args.num_classes
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, classes, args.layers, criterion)
  model = model.to(int(os.environ["LOCAL_RANK"]))
  model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  


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

  train_queue = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, pin_memory=True,num_workers=2,)
  valid_queue = torch.utils.data.DataLoader(valid_set, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)
  test_queue = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)
  vgg19_model = torch.nn.Sequential(torchvision.models.vgg19(weights=None).to(int(os.environ["LOCAL_RANK"])), torch.nn.Linear(1000, 3))
  weights_path = "weights_trained.pth" 
  checkpoint = torch.load(weights_path)
  vgg19_model.load_state_dict(checkpoint)
  vgg19_model.eval()
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.module.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.module.alphas, dim=-1))
    print(F.softmax(model.module.alphas_pool, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,vgg19_model)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion,vgg19_model)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

  destroy_process_group()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,vgg_model):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  #top3 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)
    
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
    
    with torch.no_grad():
      feature_maps = vgg_model[0].features[1](vgg_model[0].features[0](input_search))
      feature_maps = vgg_model[0].features[4](vgg_model[0].features[3](vgg_model[0].features[2](feature_maps)))
      feature_maps = vgg_model[0].features[6](vgg_model[0].features[5](feature_maps))
      feature_maps = vgg_model[0].features[9](vgg_model[0].features[8](vgg_model[0].features[7](feature_maps)))
      input_search1 = feature_maps
      feature_maps = vgg_model[0].features[11](vgg_model[0].features[10](feature_maps))
      feature_maps = vgg_model[0].features[13](vgg_model[0].features[12](feature_maps))
      feature_maps = vgg_model[0].features[15](vgg_model[0].features[14](feature_maps))
      feature_maps = vgg_model[0].features[18](vgg_model[0].features[17](vgg_model[0].features[16](feature_maps)))
      input_search2 = feature_maps
        
    architect.step(input1,input2, target, input_search1,input_search2, target_search, lr, optimizer, unrolled=args.unrolled)
    

    optimizer.zero_grad()
    logits = model(input1,input2)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1 = utils.accuracy(logits, target, topk=(1,))
    
    objs.update(loss.data, n)
    top1.update(prec1, n)
    #top3.update(prec3.data, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f', step, objs.avg, top1.avg) 
      #, top3.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion,vgg_model):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter() 
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

    logits = model(input1,input2)
    loss = criterion(logits, target)

    prec1 = utils.accuracy(logits, target)
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1, n)
    #top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f', step, objs.avg, top1.avg)  #, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 
