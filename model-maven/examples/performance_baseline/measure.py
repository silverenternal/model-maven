import time, resource, json
start = time.perf_counter()
import torch
import model
imported = time.perf_counter()
torch.set_num_threads(4)
m = model.Model().eval()
constructed = time.perf_counter()
state = torch.load('weights.pt', map_location='cpu', weights_only=True)
loaded = time.perf_counter()
m.load_state_dict(state)
applied = time.perf_counter()
x = model.batch()
prepared = time.perf_counter()
with torch.no_grad(): y = m(x)
forward = time.perf_counter()
with torch.no_grad(): y = m(x)
warm_forward = time.perf_counter()
print(json.dumps({'import_s': imported-start, 'construct_s':constructed-imported, 'weight_read_s':loaded-constructed, 'load_state_s':applied-loaded, 'input_s':prepared-applied, 'forward_s':forward-prepared, 'warm_forward_s':warm_forward-forward, 'total_s':forward-start, 'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'output_shape':list(y.shape), 'torch':torch.__version__, 'threads':torch.get_num_threads()}))
