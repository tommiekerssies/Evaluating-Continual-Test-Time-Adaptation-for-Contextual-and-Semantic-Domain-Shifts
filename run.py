import utils
from torch.optim import Adam
import wandb

if utils.args.method == "source":
  model = utils.get_model(load_saved_model=True).eval()

elif utils.args.method == "bn":
  model = utils.get_model(load_saved_model=True).train()

elif utils.args.method == "tent":
  import tent
  
  model = utils.get_model(load_saved_model=True, find_unused_parameters=True).train()
  model = tent.configure_model(model)
  tent.check_model(model)

  params, _ = tent.collect_params(model)
  optimizer = Adam(params, lr=utils.args.lr)
  model = tent.Tent(model, optimizer)
  
mean_acc = utils.eval(model, log_intermediate_results=True)
if utils.is_master:
  wandb.log({"mean": mean_acc})