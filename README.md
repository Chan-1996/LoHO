# LoHO

### Installation
```bash
Python==3.9.18
torch==2.2.1 
transformers==4.28.1
datasets==2.14.4 
spops
```

### RoBERTa-large experiments
```bash
cd medium_models

#run inter-layer hybrid optimization with sgd
bash loho_sgd_inter.sh

#run inter-layer hybrid optimization with Adam
bash loho_adam_inter.sh

#run intra-layer hybrid optimization with Adam
bash loho_adam_intra.sh
```
You can modify the task name and hyperparamters in these scripts.

### OPT-13B experiments
```bash
cd large_models

#run inter-layer hybrid optimization with sgd
bash loho_sgd_inter.sh

#run inter-layer hybrid optimization with Adam
bash loho_adam_inter.sh

#run intra-layer hybrid optimization with Adam
bash loho_adam_intra.sh
```
You can modify the task name and hyperparamters in these scripts.
