# IBL-Serotonin

## Instructions
Follow these instructions to set up an IBL environment which can load the data through the ONE interface: https://github.com/int-brain-lab/iblenv. The data are not publicly available yet, you will need IBL access.

Requirements:
- zetapy
- psychofit (for psychometric functions)
- torch 

To install these requirements activate the ``iblenv`` and run the following commands
```
pip install zetapy
pip install torch
git clone https://github.com/cortex-lab/psychofit
conda develop ./psychofit
```

