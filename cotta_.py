# %%
import utils
import torch
from tqdm import tqdm
import cotta

results = utils.get_test_results_matrix()
for i_permutation, permutation in enumerate(utils.test_permutations):
  model = utils.get_model()
  model = cotta.configure_model(model)
  params, _ = cotta.collect_params(model)
  optimizer = torch.optim.Adam(params, lr=utils.args.lr)
  model = cotta.CoTTA(model, optimizer, utils.device)
  for cycle in range(utils.args.cycles):
    test_session_loaders = utils.get_test_session_loaders()
    for i_session, session in enumerate(permutation):
      correct = 0
      total = 0
      loader = test_session_loaders[session]
      for image, label in tqdm(loader, total=len(loader)):
        image, label = image.to(utils.device).float(), label.to(utils.device)
        output = model(image)
        pred = torch.max(output, dim=1).indices
        correct += torch.sum(pred == label)
        total += label.size(0)
      acc = float(correct / total)
      results[i_permutation][cycle][i_session] = acc
      print(results)