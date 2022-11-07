# %%
from copy import deepcopy
from statistics import mean
from torch.optim import Adam
import wandb
import torch
import os

adaptation_model_name = "model.pth.tar"
target_accs = []
source_accs = []
epoch = 0

try:
  import utils

  if utils.args.resume_run:
    adaptation_model = torch.load(wandb.restore(adaptation_model_name))
    print(adaptation_model)
    
  elif utils.args.method == "source":
    adaptation_model = utils.get_model(load_saved_model=True).eval()

  elif utils.args.method == "bn":
    adaptation_model = utils.get_model(load_saved_model=True).train()

  elif utils.args.method == "tent":
    import tent
    
    model = utils.get_model(load_saved_model=True, find_unused_parameters=True).train()
    model = tent.configure_model(model)
    tent.check_model(model)

    params = tent.Tent.collect_params(model)
    optimizer = Adam(params, lr=utils.args.lr)
    adaptation_model = tent.Tent(model, optimizer)

  elif utils.args.method == "cotta":
    import cotta

    model = utils.get_model(load_saved_model=True)
    model = cotta.configure_model(model)
    cotta.check_model(model)

    params = cotta.CoTTA.collect_params(model)
    optimizer = Adam(params, lr=utils.args.lr)
    adaptation_model = cotta.CoTTA(model, optimizer, utils.device, mt_alpha=utils.args.mt_alpha, rst_m=utils.args.rst_m)

  if utils.is_master:
    wandb.watch(adaptation_model)

  source_val_loader = utils.get_loader(domains=utils.args.sources, include_train_data=False, include_val_data=True)
  target_domain_loaders = []
  for domain in utils.args.targets:
    target_domain_loaders.append(utils.get_loader([domain], include_train_data=True, include_val_data=True))

  log_intermediate_results = utils.args.epochs == 1

  def get_source_acc(temp):
    return utils.eval(deepcopy(adaptation_model), source_val_loader, log_as=temp + ','.join(utils.args.sources) if log_intermediate_results else None)
  
  source_accs.append(get_source_acc("start "))
  while utils.args.epochs is None or epoch < utils.args.epochs:
    epoch += 1
    
    target_acc = []
    for i, domain in enumerate(utils.args.targets):
      if utils.distributed:
        target_domain_loaders[i].sampler.set_epoch(epoch)  
      
      target_acc.append(utils.eval(adaptation_model, target_domain_loaders[i], log_as=domain if log_intermediate_results else None))
    
    source_accs.append(get_source_acc("end "))
    target_accs.append(target_acc)
    
    if utils.is_master:	
      torch.save(adaptation_model, os.path.join(wandb.run.dir, adaptation_model_name))
      wandb.log({
        "epoch": epoch, 
        "source_acc": source_accs, 
        "target_acc": target_accs, 
        "mean_old_temp": mean([acc for accs in target_accs for acc in accs] + source_accs),
        "target_acc_mean": mean([acc for accs in target_accs for acc in accs]),
        "forget_rate": source_accs[0] - source_accs[-1]
      })

except Exception as e:
  if utils.is_master:
    path=f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
    
    wandb.finish(exit_code=1, quiet=True)
    
    if len(target_accs) == 0 and not utils.args.resume_run:
      wandb.Api().run(path).delete()
    
    raise e