
# %%
from argparse import ArgumentParser
import random
import os
import numpy as np
import torch
import itertools
from torch.utils.data import RandomSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import core50
import domainnet
from torch.utils.data.distributed import DistributedSampler
import wandb
from statistics import mean

# TODO: img_size parameter that is also used in cotta transform code
# TODO: add example run/sh script to repo

parser = ArgumentParser()
parser.add_argument("--path", type=str, help="Path where data and models should be stored", default="./")
parser.add_argument("--max_epochs", type=int)
# TODO: try different bs
parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="Main learning rate")
parser.add_argument("--seed", default=0, type=int)
# TODO: try different number of workers
parser.add_argument("--num_workers", default=8, type=int, help="Workers number for torch Dataloader")
parser.add_argument("--cycles", default=1, type=int, help="Number of adaptation cycles")
parser.add_argument("--model", type=str, help="Load this model", default="best_real_2020.pth.tar")
parser.add_argument("--dataset", type=str, default="DomainNet-126")
parser.add_argument("--eval", default=True, type=bool)
parser.add_argument("--sources", type=str, nargs='+', default=["real"])
parser.add_argument("--targets", type=str, nargs='+', default=["painting"])

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
    else:
      device = torch.device("cuda")
else:
    device = torch.device("cpu")

if (not distributed or dist.get_rank() == 0):
  wandb.init(project="CTTAVR")
  wandb.config.update(args)
  wandb.config.world_size = dist.get_world_size() if distributed else 1

if args.dataset == "CORe50":
  dataset_ = core50
elif args.dataset == "DomainNet-126":
  dataset_ = domainnet

def get_model(load_saved_model):
  model = dataset_.Model(device)
  if load_saved_model and args.model is not None:
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
  model = model.to(device)
  if distributed:
    model = DDP(model, [dist.get_rank()], dist.get_rank())
  return model

def get_target_loaders():
  domain_loaders = {}
  for domain in args.targets:
    dataset = dataset_.Dataset(root=args.path, transform=dataset_.val_transform, domains=[domain])
    if distributed:
      sampler = DistributedSampler(dataset, seed=args.seed, shuffle=True)
    else:
      sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed))
    domain_loaders[domain] = DataLoader(dataset=dataset,
                                        batch_size=args.batch_size,
                                        sampler=sampler,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
  return domain_loaders

def train(model, loader, criterion, optimizer):
  model = model.train()
  total_loss = torch.tensor(0., device=device)
  total_correct = torch.tensor(0, device=device)
  total_images = torch.tensor(0, device=device)
  for image, label in loader:
    image, label = image.to(device).float(), label.to(device)
    output = model(image)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    pred = torch.max(output, dim=1).indices
    total_loss += (loss.item() / len(loader))
    total_correct += (torch.sum(pred == label))
    total_images += label.size(0)
  if distributed:
    dist.all_reduce(total_loss)
    dist.all_reduce(total_correct)
    dist.all_reduce(total_images)
  return total_loss, total_correct / total_images

def eval(model, eval_mode=True, stop_permutation=1, reset=False):
  if not args.eval:
    return None
  
  if (not distributed or dist.get_rank() == 0):
    wandb.watch(model)
  
  if eval_mode:
    model = model.eval()
  else:
    model = model.train()
  
  permutations = list(itertools.permutations(args.targets))[:stop_permutation]
  results = {}
  for permutation in permutations:
    results[str(permutation)] = {}
    if reset:
      model.reset()
    for cycle in range(args.cycles):
      results[str(permutation)][f"cycle_{cycle}"] = {}
      target_loaders = get_target_loaders()
      for i_domain, domain in enumerate(permutation):
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)
        loader = target_loaders[domain]
        for image, label in loader:
          image, label = image.to(device).float(), label.to(device)
          output = model(image)
          pred = torch.max(output, dim=1).indices
          correct += torch.sum(pred == label)
          total += label.size(0)
          intermediate_correct = correct.detach().clone()
          intermediate_total = total.detach().clone()
          if distributed:
            dist.all_reduce(intermediate_correct)
            dist.all_reduce(intermediate_total)
          results[str(permutation)][f"cycle_{cycle}"][f"{i_domain}_{domain}"] = float(intermediate_correct / intermediate_total)
          if (not distributed or dist.get_rank() == 0):	
            wandb.log(results[str(permutation)][f"cycle_{cycle}"])

  def values(d):
    for val in d.values():
      if isinstance(val, dict):
        yield from values(val)
      else:
        yield val

  return mean(list(values(results)))