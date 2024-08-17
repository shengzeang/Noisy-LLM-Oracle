import sys
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import json

with open("abstract_simple_cora.json", "r") as json_file:
    data = json.load(json_file)
# "citeseer"  "cora" "products" "wikics"
data = data["cora"]
sequences = []
for k, v in data.items():
    print(k)
    sequences.append(v)


PATH_TO_LLM = ""
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_LLM)
model = AutoModel.from_pretrained(PATH_TO_LLM)
tokenizer.pad_token = tokenizer.eos_token

tokens = tokenizer(sequences, padding=True,
                   truncation=True, return_tensors="pt")
output = model(**tokens)


def pooling(memory_bank, seg, pooling_type):
    seg = torch.unsqueeze(seg, dim=-1).type_as(memory_bank)
    memory_bank = memory_bank * seg
    if pooling_type == "mean":
        features = torch.sum(memory_bank, dim=1)
        features = torch.div(features, torch.sum(seg, dim=1))
    elif pooling_type == "last":
        features = memory_bank[torch.arange(memory_bank.shape[0]), torch.squeeze(
            torch.sum(seg != 0, dim=1).type(torch.int64) - 1), :]
    elif pooling_type == "max":
        features = torch.max(memory_bank + (seg - 1) * sys.maxsize, dim=1)[0]
    else:
        features = memory_bank[:, 0, :]
    return features


feat = pooling(output['last_hidden_state'],
               tokens['attention_mask'], pooling_type="mean")

similarity_feature = torch.mm(feat, feat.t())
norm = torch.norm(similarity_feature, 2, 1, keepdim=True).add(1e-8)
similarity_feature = torch.div(similarity_feature, norm)
similarity_feature = torch.div(similarity_feature, norm.t())
similarity_feature = F.normalize(similarity_feature, p=1, dim=1)
torch.save(similarity_feature, 'mistral8x7b_cora_sim_simple.pt')
