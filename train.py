#%%
import utils
import time
import torch
from torch import Generator
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.distributed as dist
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from tqdm import tqdm
import datetime
from core50 import CORE50

train_dataset = CORE50(root=utils.args.path, train=True, download=True, transform=utils.train_transform)
val_dataset   = CORE50(root=utils.args.path, train=False, transform=utils.val_transform)

if utils.distributed:
  train_sampler = DistributedSampler(train_dataset, seed=utils.args.seed, shuffle=True)
  val_sampler   = DistributedSampler(val_dataset, seed=utils.args.seed, shuffle=True)
else:
  train_sampler = RandomSampler(train_dataset, generator=Generator().manual_seed(utils.args.seed))
  val_sampler   = RandomSampler(val_dataset, generator=Generator().manual_seed(utils.args.seed))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=utils.args.batch_size,
                          sampler=train_sampler,
                          num_workers=utils.args.num_workers,
                          pin_memory=True)
val_loader   = DataLoader(dataset=val_dataset,
                          batch_size=utils.args.batch_size,
                          sampler=val_sampler,
                          num_workers=utils.args.num_workers,
                          pin_memory=True)

criterion = CrossEntropyLoss()
optimizer = Adam(utils.model.parameters(), lr=utils.args.lr)

epoch = 0
filename = None
best_val_acc = 0.
while True:
  epoch_start = time.time()
  if utils.distributed:
    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch)

  model = utils.get_model().train()
  train_loss = 0.
  train_correct = 0
  train_total = 0
  for image, label in tqdm(train_loader, total=len(train_loader)):
    image, label = image.to(utils.device).float(), label.to(utils.device)
    output = model(image)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_loss += loss.item() / len(train_loader)
    pred = torch.max(output, dim=1).indices
    train_correct += torch.sum(pred == label)
    train_total += label.size(0)
  train_acc = train_correct / train_total

  model = model.eval()
  val_correct = 0
  val_total = 0
  for image, label in tqdm(val_loader, total=len(val_loader)):
    image, label = image.to(utils.device).float(), label.to(utils.device)
    output = model(image)
    pred = torch.max(output, dim=1).indices
    val_correct += torch.sum(pred == label)
    val_total += label.size(0)
  val_acc = val_correct / val_total

  print("""
    rank: {}, 
    epoch {}, 
    train loss: {:.4f}, 
    train accuracy: {:.8f},
    val accuracy: {:,.8f}
  """.format(
    utils.local_rank,
    epoch,
    train_loss,
    train_acc,
    val_acc,
  ))

  if utils.distributed and dist.get_rank() == 0 and val_acc > best_val_acc:
    best_val_acc = val_acc
    if filename:
      os.remove(filename)
    if "SLURM_JOB_ID" in os.environ:
      filename = os.environ["SLURM_JOB_ID"]
    else:
      filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename += f"_epoch_{str(int(epoch))}"
    filename += f"_train_acc_{str(float(train_acc))}"
    filename += f"_val_acc_{str(float(val_acc))}"
    filename += ".model"
    torch.save(model.module.state_dict(), filename)
  
  epoch += 1