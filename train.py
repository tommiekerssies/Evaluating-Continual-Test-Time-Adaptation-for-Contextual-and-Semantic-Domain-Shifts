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
from core50 import CORE50

def train(model, train_loader):
  model = model.train()
  train_loss = 0.
  train_correct = 0
  train_total = 0
  for image, label in train_loader:
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
  dist.all_reduce(train_correct)
  dist.all_reduce(train_total)
  return train_loss, train_correct / train_total

train_dataset = CORE50(root=utils.args.path, sessions=utils.args.train_sessions, download=True, transform=utils.train_transform)

if utils.distributed:
  train_sampler = DistributedSampler(train_dataset, seed=utils.args.seed, shuffle=True)
else:
  train_sampler = RandomSampler(train_dataset, generator=Generator().manual_seed(utils.args.seed))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=utils.args.batch_size,
                          sampler=train_sampler,
                          num_workers=utils.args.num_workers,
                          pin_memory=True)

criterion = CrossEntropyLoss()
model = utils.get_model(load_saved_model=False)
optimizer = Adam(model.parameters(), lr=utils.args.lr)

epoch = 0
best_val_acc = utils.eval(model)
if not utils.distributed or dist.get_rank() == 0:
  print("""epoch {}, val accuracy: {}""".format(epoch, best_val_acc))
filename = None
while utils.args.max_epochs is None or epoch < utils.args.max_epochs:
  epoch += 1
  epoch_start = time.time()
  if utils.distributed:
    train_sampler.set_epoch(epoch)

  train_loss, train_acc = train(model, train_loader)
  val_acc = utils.eval(model)

  if not utils.distributed or dist.get_rank() == 0:
    print("""
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

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      if filename:
        os.remove(filename)
      filename = f"sessions_{','.join(utils.args.train_sessions)}"
      filename += f"_epoch_{str(int(epoch))}"
      filename += f"_train_acc_{str(float(train_acc))}"
      filename += f"_val_acc_{str(float(val_acc))}"
      if "SLURM_JOB_ID" in os.environ:
        filename += f"_jobid_{os.environ['SLURM_JOB_ID']}"
      filename += ".model"
      torch.save(model.module.state_dict(), filename)