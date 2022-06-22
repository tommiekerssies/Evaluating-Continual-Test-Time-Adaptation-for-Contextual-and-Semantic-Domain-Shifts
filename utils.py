
# %%
from argparse import ArgumentParser
import random
import os
import numpy as np
import torch
import itertools
from torch.nn import Linear
from torch.utils.data import RandomSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.models import resnet18
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomHorizontalFlip,
)
from core50 import CORE50
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# TODO: add test sessions argument to use for training and TTA

parser = ArgumentParser()
parser.add_argument("--path", default="./", type=str, help="Path where data and models should be stored")
parser.add_argument("--max_epochs", type=int, help="Batch size")
# TODO: try different bs
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="Main learning rate")
parser.add_argument("--seed", default=0, type=int)
# TODO: try different number of workers
parser.add_argument("--num_workers", default=10, type=int, help="Workers number for torch Dataloader")
parser.add_argument("--cycles", default=1, type=int, help="Number of adaptation cycles")
parser.add_argument("--model", default="6530566_epoch_15_train_acc_0.9959_val_acc_0.6567.model", type=str, help="Load this model")
parser.add_argument("--train_sessions", nargs='+')
parser.add_argument("--val_sessions", nargs='+')

# Add these dummy arguments so code can be run as notebook
parser.add_argument("--ip")
parser.add_argument("--stdin")
parser.add_argument("--control")
parser.add_argument("--hb")
parser.add_argument("--Session.signature_scheme")
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

if dist.get_rank() == 0:
  print("World size: " + str(dist.get_world_size()))

if not distributed or dist.get_rank() == 0: 
  print(args)

all_sessions = set(range(1,12))

if args.val_sessions is None:
  if args.train_sessions is None:
    exit()
  args.val_sessions = all_sessions - set(args.train_sessions)

if args.train_sessions is None:
  args.train_sessions = all_sessions - set(args.val_sessions)

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = Compose(
    [ToTensor(), RandomHorizontalFlip(), normalize]
)
val_transform = Compose([ToTensor(), normalize])

def get_model(load_saved_model):
  model = resnet18(pretrained=True)
  num_ftrs = model.fc.in_features
  # TODO: try the simpler task of predicting the 10 categories
  model.fc = Linear(num_ftrs, 50)
  if load_saved_model and args.model is not None:
    model.load_state_dict(torch.load(args.model, map_location=device))
  model = model.to(device)
  if distributed:
    model = DDP(model, [dist.get_rank()], dist.get_rank())
  return model

def get_val_session_loaders():
  session_loaders = {}
  for session in args.val_sessions:
    dataset = CORE50(root=args.path, transform=val_transform, sessions=[session])
    if distributed:
      sampler = DistributedSampler(dataset, seed=args.seed, shuffle=True)
    else:
      sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed))
    session_loaders[session] = DataLoader(dataset=dataset,
                                          batch_size=args.batch_size,
                                          sampler=sampler,
                                          num_workers=args.num_workers,
                                          pin_memory=True)
  return session_loaders

def eval(model, stop_permutation=1, reset=False):
  permutations = list(itertools.permutations(args.val_sessions))[:stop_permutation]
  results = np.full(shape=(len(permutations), args.cycles, len(args.val_sessions)), fill_value=None)
  for i_permutation, permutation in enumerate(permutations):
    if reset:
      model.reset()
    for cycle in range(args.cycles):
      val_session_loaders = get_val_session_loaders()
      for i_session, session in enumerate(permutation):
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)
        loader = val_session_loaders[session]
        for image, label in loader:
          image, label = image.to(device).float(), label.to(device)
          output = model(image)
          pred = torch.max(output, dim=1).indices
          correct = torch.add(correct, torch.sum(pred == label))
          total = torch.add(total, label.size(0))
        dist.all_reduce(correct)
        dist.all_reduce(total)
        acc = float(correct / total)
        results[i_permutation][cycle][i_session] = acc
        if not distributed or dist.get_rank() == 0:
          print(results)
  return np.mean(results)