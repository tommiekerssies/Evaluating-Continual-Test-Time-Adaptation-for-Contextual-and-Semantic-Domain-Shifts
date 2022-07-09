
# TODO: img_size parameter that is also used in cotta transform code
# TODO: add run/sh scripts to repo
# %%
from argparse import ArgumentParser
from copy import deepcopy
import random
import os
import numpy as np
import torch
from torch.utils.data import RandomSampler, DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.modules import SyncBatchNorm
import torch.distributed as dist
import core50
import domainnet
from torch.utils.data.distributed import DistributedSampler
import wandb
from IPython import get_ipython


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def get_model(load_saved_model, find_unused_parameters=False):
  model = dataset_.Model(device)
  
  if load_saved_model and args.model is not None:
    state_dict = torch.load(os.path.join(args.path, args.model), map_location=device)
    model.load_state_dict(state_dict)
  
  model = model.to(device)
  
  if distributed:
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, [dist.get_rank()], dist.get_rank(), find_unused_parameters=find_unused_parameters)
  
  return model

def get_loader(domains, include_train_data, include_val_data):
  dataset = dataset_.Dataset(root=args.path, transform=dataset_.val_transform, domains=domains)
  
  num_train = int(len(dataset) * dataset_.train_ratio)
  indices = np.random.RandomState(seed=args.seed).permutation(len(dataset))
  if not include_train_data:
    dataset = Subset(dataset, indices[num_train:])
  elif not include_val_data:
    dataset = Subset(dataset, indices[:num_train])

  if distributed:
    sampler = DistributedSampler(dataset, seed=args.seed, shuffle=True)
  else:
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed))

  return DataLoader(dataset=dataset,
                    batch_size=args.batch_size,
                    sampler=sampler,
                    num_workers=args.num_workers,
                    pin_memory=True)

def eval(model, loader, log_as=None):  
  correct = torch.tensor(0, device=device)
  total = torch.tensor(0, device=device)
  acc = None

  for image, label in loader:
    image, label = image.to(device).float(), label.to(device)
    output = model(image)
    pred = torch.max(output, dim=1).indices
    correct += torch.sum(pred == label)
    total += label.size(0)
    
    if log_as:
      intermediate_correct = correct.detach().clone()
      intermediate_total = total.detach().clone()
      
      if distributed:
        dist.all_reduce(intermediate_correct)
        dist.all_reduce(intermediate_total)
      
      if is_master:	
        acc = float(intermediate_correct / intermediate_total)
        wandb.log({log_as: acc})
  
  return acc if acc else float(correct / total)

parser = ArgumentParser()
parser.add_argument("--method", type=str)
parser.add_argument("--path", type=str, help="Path where data and models should be stored")
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--lr", type=float, help="Main learning rate")
parser.add_argument("--seed", type=int)
# TODO: try different number of workers
parser.add_argument("--num_workers", default=16, type=int, help="Workers number for torch Dataloader")
parser.add_argument("--model", type=str, help="Load this model")
parser.add_argument("--dataset", type=str)
parser.add_argument("--sources", type=str, nargs='+')
parser.add_argument("--targets", type=str, nargs='+')
parser.add_argument("--mt_alpha", type=float)
parser.add_argument("--rst_m", type=float)

if is_notebook():
  # Add these dummy arguments so code can be run as notebook
  parser.add_argument("--ip")
  parser.add_argument("--stdin")
  parser.add_argument("--control")
  parser.add_argument("--hb")
  parser.add_argument("--domain.signature_scheme")
  parser.add_argument("--Session.signature_scheme")
  parser.add_argument("--domain.key")
  parser.add_argument("--Session.key")
  parser.add_argument("--shell")
  parser.add_argument("--transport")
  parser.add_argument("--iopub")
  parser.add_argument("--f")

args = parser.parse_args()

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

distributed = False
if torch.cuda.is_available():
    if "LOCAL_RANK" in os.environ:
      distributed = True
      dist.init_process_group(backend="nccl")
      device = torch.device(f"cuda:{dist.get_rank()}")
      torch.cuda.set_device(device)  # without this, somehow pytorch will start multiple processes on GPU 0
    else:
      device = torch.device("cuda")
else:
    if "LOCAL_RANK" in os.environ:
      raise Exception("distributed but no cuda available")
    device = torch.device("cpu")

is_master = not is_notebook() and (not distributed or dist.get_rank() == 0)

if is_master:
  wandb.init(project="CTTAVR")
  wandb.config.update(args)
  wandb.config.world_size = dist.get_world_size() if distributed else 1

if args.dataset == "CORe50":
  dataset_ = core50
elif args.dataset == "DomainNet-126":
  dataset_ = domainnet