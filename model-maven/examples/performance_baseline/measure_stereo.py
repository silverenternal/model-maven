import json, resource, sys, time
start = time.perf_counter()
import torch
from adapter import model, batch
imported = time.perf_counter()
torch.set_num_threads(4)
device = sys.argv[1]
if device.startswith('cuda'):
    torch.cuda.set_device(int(device.split(':')[1]))
    torch.cuda.reset_peak_memory_stats()
start_model = time.perf_counter()
net = model().eval().to(device)
if device.startswith('cuda'): torch.cuda.synchronize()
model_done = time.perf_counter()
inputs = batch()
args = [value.to(device) for value in inputs['args']]
if device.startswith('cuda'): torch.cuda.synchronize()
input_done = time.perf_counter()
times = []
with torch.inference_mode():
    for _ in range(2):
        before = time.perf_counter()
        result = net(*args, **inputs['kwargs'])
        if device.startswith('cuda'): torch.cuda.synchronize()
        times.append(time.perf_counter() - before)
print(json.dumps({'device':device,'device_name':torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU','torch':torch.__version__,'import_seconds':imported-start,'construct_and_transfer_seconds':model_done-start_model,'input_seconds':input_done-model_done,'forward_seconds':times,'output_shape':list(result.shape),'parameters':sum(x.numel() for x in net.parameters()),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.startswith('cuda') else None,'weight':'random, no pretrained weights'}))
