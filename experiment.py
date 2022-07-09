# %%
from copy import deepcopy
from statistics import mean
import numpy as np
import wandb
from torch.optim import Adam

target_acc = None

try:
  import utils

  if utils.args.method == "source":
    model = utils.get_model(load_saved_model=True).eval()

  elif utils.args.method == "bn":
    model = utils.get_model(load_saved_model=True).train()

  elif utils.args.method == "tent":
    import tent
    
    model = utils.get_model(load_saved_model=True, find_unused_parameters=True).train()
    model = tent.configure_model(model)
    tent.check_model(model)

    params = tent.Tent.collect_params(model)
    optimizer = Adam(params, lr=utils.args.lr)
    model = tent.Tent(model, optimizer)

  elif utils.args.method == "cotta":
    import cotta

    model = utils.get_model(load_saved_model=True)
    model = cotta.configure_model(model)
    cotta.check_model(model)

    params = cotta.CoTTA.collect_params(model)
    optimizer = Adam(params, lr=utils.args.lr)
    model = cotta.CoTTA(model, optimizer, utils.device, mt_alpha=utils.args.mt_alpha, rst_m=utils.args.rst_m)

  source_val_loader = utils.get_loader(domains=utils.args.sources, include_train_data=False, include_val_data=True)
  target_domain_loaders = []
  for domain in utils.args.targets:
    target_domain_loaders.append(utils.get_loader([domain], include_train_data=True, include_val_data=True))

  def get_source_acc():
    return utils.eval(deepcopy(model), source_val_loader, log_as=','.join(utils.args.sources))
  
  epoch = 0
  target_acc = []
  source_acc = [get_source_acc()]
  while utils.args.epochs is None or epoch < utils.args.epochs:
    epoch += 1
    
    target_acc.append([])
    for i, domain in enumerate(utils.args.targets):
      if utils.distributed:
        target_domain_loaders[i].sampler.set_epoch(epoch)  
      
      target_acc[-1].append(utils.eval(model, target_domain_loaders[i], log_as=domain))
    
    source_acc.append(get_source_acc())
    
    if utils.is_master:	
      wandb.log({
        "epoch": epoch, 
        "source_acc": source_acc, 
        "target_acc": target_acc, 
        "target_acc_mean": mean([acc for accs in target_acc for acc in accs]),
        "forget_rate": source_acc[0] - source_acc[-1]}
      )

except Exception as e:
  if utils.is_master:
    path=f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
    
    wandb.finish(exit_code=1, quiet=True)
    
    if not np.any(target_acc):
      wandb.Api().run(path).delete()
    
    raise e