import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C):
    super(Cell, self).__init__()
    self.preprocess0 = FactorizedReduce(C_prev_prev,C)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    genotype_normal = genotype.normal[:-1]
    genotype_reduce = genotype.normal[-1]
    op_names, indices = zip(*genotype_normal)
    reduction_op_names = [genotype_reduce[0]]
    reduction_indices = [0]
    concat = genotype.normal_concat
    self._compile(C, op_names, indices,reduction_op_names,reduction_indices, concat)

  def _compile(self, C, op_names, indices,reduction_op_names,reduction_indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops += [op]

    self.reduction_ops = nn.ModuleList()
    for name in reduction_op_names:
      stride=2
      op = OPS_POOL[name](stride)
      self.reduction_ops+=[op]

    self._indices = indices
    self._reduction_indices = reduction_indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s] 
    reduction_states = [torch.cat([states[i] for i in self._concat], dim=1)]
    for i in range(1):
      op = self.reduction_ops[0]
      s = reduction_states[0]
      s = op(s)
      reduction_states.append(s)
    
    return reduction_states[-1]



class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self.drop_path_prob = 0

    C_prev_prev, C_prev, C_curr = C, C, C
    self.stem0 = nn.Sequential(
      nn.Conv2d(128, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    self.stem1 = nn.Sequential(nn.Conv2d(256,C_curr,3,padding=1,bias=False),
                               nn.BatchNorm2d(C_curr)
    )

    self.cells = nn.ModuleList()
    for i in range(layers):
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr)
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(1)
    self.classifier = nn.Sequential(nn.Linear(4096,1000),nn.Linear(1000,num_classes))

  def forward(self, input1,input2):
    logits_aux = None
    s0 = self.stem0(input1)
    s1 = self.stem1(input2)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux