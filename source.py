# %%
import utils

model = utils.get_model(load_saved_model=True).eval()
utils.eval(model)