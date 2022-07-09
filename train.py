#%%
import utils
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.distributed as dist
import os
import wandb

model_path = os.path.join(utils.args.path, utils.args.model)
if os.path.exists(model_path):
  raise Exception("Model already exists at {}".format(model_path))

train_loader = utils.get_loader(domains=utils.args.sources, include_train_data=True, include_val_data=False)
val_loader = utils.get_loader(domains=utils.args.sources, include_train_data=False, include_val_data=True)
criterion = CrossEntropyLoss()
model = utils.get_model(load_saved_model=False)
optimizer = Adam(model.parameters(), lr=utils.args.lr)
epoch = 0

best_val_acc = utils.eval(model.eval(), val_loader)
if utils.is_master:
  wandb.log({"start_val_acc": best_val_acc})

if utils.is_master:
  wandb.watch(model)

while utils.args.epochs is None or epoch < utils.args.epochs:
  if utils.distributed:
    train_loader.sampler.set_epoch(epoch)

  total_loss = torch.tensor(0., device=utils.device)
  total_correct = torch.tensor(0, device=utils.device)
  total_images = torch.tensor(0, device=utils.device)

  for image, label in train_loader:
    image, label = image.to(utils.device).float(), label.to(utils.device)
    model = model.train()
    output = model(image)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    pred = torch.max(output, dim=1).indices
    total_loss += (loss.item() / len(train_loader))
    total_correct += (torch.sum(pred == label))
    total_images += label.size(0)
  
  if utils.distributed:
    dist.all_reduce(total_loss)
    dist.all_reduce(total_correct)
    dist.all_reduce(total_images)
  
  val_acc = utils.eval(model.eval(), val_loader)

  if utils.is_master:
    wandb.log({"epoch": epoch, "train_loss": total_loss, "train_acc": total_correct / total_images, "val_acc": val_acc})

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      if os.path.exists(model_path):
        os.remove(model_path)
      torch.save(model.module.state_dict(), model_path)
  
  epoch += 1