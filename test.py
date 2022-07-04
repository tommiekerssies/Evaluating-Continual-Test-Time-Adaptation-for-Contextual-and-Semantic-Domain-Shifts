from statistics import mean
results = {'test': {'joo': 3}, 'jo': 2}
def values(d):
    for val in d.values():
      if isinstance(val, dict):
        yield from values(val)
      else:
        yield val

print(mean(list(values(results))))