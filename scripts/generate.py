#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="6"

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import Dataset, DataLoader
from modelling_led import LEDForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import CitationTextGenerationDataset, CitationTextGenerationDatasetNoCitationType


device = "cuda"
max_input_length = 16384
max_output_length = 1024


def process_data_to_model_inputs(batch, special_tokens=['[Dominant]', '[Reference]'], length=None):
    # tokenize the inputs and labels
    
    additional_special_tokens_lookup = {token: idx for token, idx in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)}
    special_token_ids = set([additional_special_tokens_lookup[token] for token in special_tokens])
    special_token_ids.add(tokenizer.mask_token_id)
    
    inputs = tokenizer(
        batch["source"],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=True 
    )
    outputs = tokenizer(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
        add_special_tokens=True 
    )
    if length:
        batch["length"] = length
    else:
        batch["length"] = [sum(x) for x in outputs.attention_mask]


    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    for i_batch in range(len(batch["input_ids"])):
        for i_token in range(len(batch["input_ids"][0])):
            if batch["input_ids"][i_batch][i_token] in special_token_ids:
                batch["global_attention_mask"][i_batch][i_token] = 1
            
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    return batch

path = sys.argv[1]


tokenizer = AutoTokenizer.from_pretrained(path)
# special_tokens = ['<doc>','</doc>', '[BOS]', '[Mask]', '[Dominant]', '[Reference]', '[B_Dominant]',  '[E_Dominant]', '[B_Reference]', '[E_Reference]']
# additional_special_tokens = {'additional_special_tokens': special_tokens}
# tokenizer.add_special_tokens(additional_special_tokens)

model = LEDForConditionalGeneration.from_pretrained(
    path
)
model = model.to(device).half()
model.eval()

def get_sentence_token_count(sent):
    return tokenizer.tokenize(sent).__len__()

def get_context(source):
    return source.split("\n\n")[0]

def run_model(batch, model, length=None):
    processed_batch = process_data_to_model_inputs(
        batch, 
        special_tokens=['[Dominant]', '[Reference]'],
        length=length
    )
    processed_batch_cuda = {}
    for key in ["input_ids", "attention_mask", "global_attention_mask", "labels", "length"]:
        processed_batch_cuda[key] = torch.tensor(processed_batch[key]).to(device)
    model_kwargs = {'decoder_length' : processed_batch_cuda["length"].unsqueeze(0)}
    predicted_abstract_ids = model.generate(
        processed_batch_cuda["input_ids"], 
        attention_mask=processed_batch_cuda["attention_mask"], 
        global_attention_mask=processed_batch_cuda["global_attention_mask"],
        **model_kwargs,
        output_scores=True,
        return_dict_in_generate = True,
        num_return_sequences = 1,
        
    )
    out = tokenizer.batch_decode(predicted_abstract_ids.sequences, skip_special_tokens=True)
    target = batch["target"]
    return out, target, predicted_abstract_ids.sequences_scores



def run_length_controlled_prediction(
    source, target = "dummy target", length=None
):
    """high level function for length controlled generation"""
    test_data = {"source": source, "target": target, "id": "test_123"}

    test_data_list = [test_data]
    
    for batch in DataLoader(test_data_list, batch_size = 1, shuffle=False):
        out, _, _ = run_model(
            batch, model,
            length=length
        )
        break
    return out[0]

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#                                                                    x
#  source format                                                     x   
#  related work context [MASK] related work context \n\n             x                                           x   
#  repeat this for all the cited papar and append to the above text  x
#       [B_Dominant] or [B_Referece]                                 x
#       citation mark                                                x   
#       </s> title                                                   x   
#       | abstract of cited paper                                    x   
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# source = """A simple (and yet effective) baseline for zero-shot translation is pivoting that chain-translates, first to a pivot language, then to a target (Cohn and Lapata, 2007; Wu and Wang, 2007; Utiyama and Isahara, 2007) .\nDespite being a pipeline, pivoting gets better as the supervised models improve, which makes it a strong baseline in the zero-shot setting.\nCheng et al. (2017) proposed a joint pivoting learning strategy that leads to further improvements.\nLu et al. (2018) and Arivazhagan et al. (2018) proposed different techniques to obtain "neural interlingual" representations that are passed to the decoder.\n [Mask] \n\n [B_Dominant] Sestorain et al. (2018) </s> [E_Dominant] [B_Dominant] (He et al., 2016) </s> Dual Learning for Machine Translation | While neural machine translation (NMT) is making good progress in the past two years, tens of millions of bilingual sentence pairs are needed for its training. However, human labeling is very costly. To tackle this training data bottleneck, we develop a dual-learning mechanism, which can enable an NMT system to automatically learn from unlabeled data through a dual-learning game. This mechanism is inspired by the following observation: any machine translation task has a dual task, e.g., English-to-French translation (primal) versus French-to-English translation (dual); the primal and dual tasks can form a closed loop, and generate informative feedback signals to train the translation models, even if without the involvement of a human labeler. In the dual-learning mechanism, we use one agent to represent the model for the primal task and the other agent to represent the model for the dual task, then ask them to teach each other through a reinforcement learning process. Based on the feedback signals generated during this process (e.g., the languagemodel likelihood of the output of a model, and the reconstruction error of the original sentence after the primal and dual translations), we can iteratively update the two models until convergence (e.g., using the policy gradient methods). We call the corresponding approach to neural machine translation dual-NMT. Experiments show that dual-NMT works very well on English↔French translation; especially, by learning from monolingual data (with 10% bilingual data for warm start), it achieves a comparable accuracy to NMT trained from the full bilingual data for the French-to-English translation task. [E_Dominant]"""

input_path = sys.argv[2]

with open(input_path, "r") as f:
    data = json.load(f)

related_work_context = data["related_work_context"]

source = f"{related_work_context} \n\n "

for citation in data["cited_papers"]:
    citation_type, citation_mark, title, abstract = citation["citation_type"], citation["citation_mark"], citation["title"], citation["abstract"]
    source += f"[B_{citation_type}] {citation_mark} </s> {title} | {abstract} [E_{citation_type}] "

print("\n\n\n")
print("Input to model \n" + "*"*20 + "\n", source)

print("generation of length 20: ", run_length_controlled_prediction(source, length=20))
print("\n\n")
print("generation of length 40: ", run_length_controlled_prediction(source, length=40))
print("\n\n")
print("generation of length 50: ", run_length_controlled_prediction(source, length=50))

