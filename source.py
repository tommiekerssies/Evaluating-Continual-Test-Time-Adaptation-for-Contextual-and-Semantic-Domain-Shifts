# %%
import utils

model = utils.get_model().eval()
utils.eval(model, reset=False, stop_permutation=1)