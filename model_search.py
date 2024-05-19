
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES,PRIMITIVES_POOL
from genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class choosePool(nn.Module):
  def __init__(self,stride): 
    super(choosePool, self).__init__()
    self._ops = nn.ModuleList()
    for pool in PRIMITIVES_POOL:
      op = OPS_POOL[pool](stride)
      self._ops.append(op)

  def forward(self,x,weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C):
    super(Cell, self).__init__()
    self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._reduction_ops = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 1
        op = MixedOp(C, stride)
        self._ops.append(op)
    
    stride = 2
    op = choosePool(stride)
    self._reduction_ops.append(op)

  def forward(self, s0, s1, weights,weights_alpha):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    final_state =  torch.cat(states[-self._multiplier:], dim=1)
    
    offset = 0
    reduction_states = [final_state]
    for i in range(1):
      s = sum(self._reduction_ops[offset+j](h, weights_alpha[offset+j]) for j, h in enumerate(reduction_states))
      offset += len(reduction_states)
      reduction_states.append(s)
    
    return reduction_states[-1]

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=2):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem1 = nn.Sequential(
      nn.Conv2d(128, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    self.stem2 = nn.Sequential(nn.Conv2d(256,C_curr,3,padding=1,bias=False),
                               nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr)
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(p=0.25)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input1,input2):
    s0 = self.stem1(input1)
    s1 = self.stem2(input2)
    for i, cell in enumerate(self.cells):
      weights = F.softmax(self.alphas, dim=-1)
      weights_pool = F.softmax(self.alphas_pool,dim=-1)
      s0, s1 = s1, cell(s0, s1, weights,weights_pool)
    out = self.global_pooling(s1)
    out = self.dropout(out)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input1,input2, target):
    logits = self(input1,input2)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_pool = Variable(1e-3*torch.randn(1, 2).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas,self.alphas_pool
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights,weights_pool):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      W = weights_pool.copy()
      k_best = None
      for k in range(len(W)):
        if k_best is None or W[0][k] > W[0][k_best]:
            k_best = k
      gene.append((PRIMITIVES_POOL[k_best], 0))

      return gene

    gene_normal = _parse(F.softmax(self.alphas, dim=-1).data.cpu().numpy(),F.softmax(self.alphas_pool, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
    )
    return genotype
