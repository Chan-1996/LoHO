from tasks import MultiRCDataset, RTEDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
            '/data/hybrid-opt-FT/opt-13b', use_fast=False
        )

MultiRC = MultiRCDataset()
RTE = RTEDataset()

mrc_samples = MultiRC.samples['train']
rte_samples = RTE.samples['train']

mrc_max_len = 0
rte_max_len = 0

for s in mrc_samples:
    # print(s.data)
    tokens = tokenizer.tokenize(s.data['paragraph'] + s.data['question'] + s.data['answer'])
    if len(tokens) > mrc_max_len:
        mrc_max_len = len(tokens)

for s in rte_samples:
    print(s.data)
    tokens = tokenizer.tokenize(s.data['premise'] + s.data['hypothesis'])
    if len(tokens) > rte_max_len:
        rte_max_len = len(tokens)

print(mrc_max_len)
print(rte_max_len)
