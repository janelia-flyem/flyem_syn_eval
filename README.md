# flyem\_syn\_eval

computes precision-recall curves for synapse prediction.  predicted
and ground-truth synapses can be read from json files or pulled from
DVID.

## installation

install conda via [miniconda or anaconda](https://conda.io/docs/download.html)

## example
```python
from flyem_syn_eval import eval
import numpy as np
import matplotlib.pyplot as plt

results = eval.evaluate_pr(
    eval.Tbar_Info('fpl_unet_pred_cx1_2_baseline',
                   'emdata2:8000','cb7dc','roi_cx1_2'),
    eval.cx_synapse_groundtruth('2'),
    conf_thresholds=np.arange(0.7,0.98,0.02))
plt.plot(results.rr, results.pp, 'r.-')
plt.show()
```
