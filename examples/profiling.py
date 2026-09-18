"""Profile a model and export a Perfetto-compatible trace."""

import everyo as eo
from everyo.profiler import profile

model = eo.Sequential(eo.Linear(128, 256, seed=0), eo.ReLU(), eo.Linear(256, 10, seed=1)).eval()
inputs = eo.randn(64, 128, seed=0)

with profile(record_shapes=True) as result:
    for _ in range(10):
        model(inputs)

for row in result.summary():
    print(row)
result.export_chrome_trace("everyo-trace.json")
