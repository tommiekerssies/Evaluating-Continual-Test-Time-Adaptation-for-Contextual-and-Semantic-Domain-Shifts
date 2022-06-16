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

parser = ArgumentParser()
parser.add_argument("--path", default="./", type=str, help="Path where data and models should be stored")
# TODO: try different bs
parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="Main learning rate")
parser.add_argument("--seed", default=0, type=int)
# TODO: try different number of workers
parser.add_argument("--num-workers", default=4, type=int, help="Workers number for torch Dataloader")
parser.add_argument("--cycles", default=10, type=int, help="Number of adaptation cycles")
parser.add_argument("--model", type=str, help="Load this model")

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
print(args)

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    distributed = True
    dist.init_process_group(backend="nccl")
else:
    device = "cpu"
    distributed = False

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = Compose(
    [ToTensor(), RandomHorizontalFlip(), normalize]
)
val_transform = Compose([ToTensor(), normalize])

test_sessions = (3, 7, 10)
test_permutations = list(itertools.permutations(test_sessions))

def get_model():
  model = resnet18(pretrained=True)
  num_ftrs = model.fc.in_features
  # TODO: try the simpler task of predicting the 10 categories
  model.fc = Linear(num_ftrs, 50)
  if args.model:
    model.load_state_dict(torch.load(args.model, map_location=device))
  model = model.to(device)
  if distributed:
    model = DDP(model, [local_rank], local_rank)
  return model

def get_test_session_loaders():
  test_session_loaders = {}
  for test_session in test_sessions:
    dataset = CORE50(root=args.path, train=False, transform=val_transform, test_session=test_session)
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed))
    test_session_loaders[test_session] = DataLoader(dataset=dataset,
                                                    batch_size=args.batch_size,
                                                    sampler=sampler,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)
  return test_session_loaders

def get_test_results_matrix():
  return np.full(shape=(len(test_permutations), args.cycles, len(test_sessions)), fill_value=None)
