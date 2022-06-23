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
from core50 import CORE50

dataset = CORE50(root=utils.args.path, sessions=utils.args.train_sessions, transform=utils.train_transform)

if utils.distributed:
  sampler = DistributedSampler(dataset, seed=utils.args.seed, shuffle=True)
else:
  sampler = RandomSampler(dataset, generator=Generator().manual_seed(utils.args.seed))

loader = DataLoader(dataset=dataset,
                    batch_size=utils.args.batch_size,
                    sampler=sampler,
                    num_workers=utils.args.num_workers,
                    pin_memory=True)

criterion = CrossEntropyLoss()
model = utils.get_model(load_saved_model=False)
optimizer = Adam(model.parameters(), lr=utils.args.lr)

epoch = 0
best_val_acc = utils.eval(model)
if not utils.distributed or dist.get_rank() == 0:
  print("""starting val accuracy: {}""".format(best_val_acc))
filename = None
while utils.args.max_epochs is None or epoch < utils.args.max_epochs:
  epoch_start = time.time()
  if utils.distributed:
    sampler.set_epoch(epoch)

  train_loss, train_acc = utils.train(model, loader, criterion, optimizer)
  val_acc = utils.eval(model)

  if not utils.distributed or dist.get_rank() == 0:
    print("""
      epoch {}, 
      train loss: {:.4f}, 
      train accuracy: {:.8f},
      val accuracy: {:,.8f}
    """.format(
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
  
  epoch += 1