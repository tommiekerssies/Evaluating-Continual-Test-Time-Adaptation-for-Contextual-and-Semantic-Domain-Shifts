# %%
import utils

model = utils.get_model(load_saved_model=True).train() # use training mode for BN, so the model will use batch statistics
utils.eval(model)