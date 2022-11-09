import argparse

import torch
import jsonlines
import os
import pickle

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import random
import numpy as np

from tqdm import tqdm
from util import flatten
#from paragraph_model import JointParagraphTagger
from paragraph_model import JointParagraphCRFmergeLabelTagger as JointParagraphTagger 
from dataset import JointRelatedWorkAnnotationMergeLabelDataset as JointRelatedWorkAnnotationDataset

import logging

def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def batch_token_label(labels, padding_idx):
    def letter2idx(char):
        return ord(char) - 97
    max_sent_len = max([len(label) for label in labels])
    label_matrix = torch.ones(len(labels), max_sent_len) * padding_idx
    label_list = []
    for i, label in enumerate(labels):
        label_indices = [letter2idx(evid) for evid in label]
        label_matrix[i,:len(label_indices)] = torch.tensor(label_indices)
        label_list.append(label_indices)
    return label_matrix.long(), label_list

def index2label(all_indices, mapping):
    all_labels = []
    for indices in all_indices:
        all_labels.append([mapping.get(index,"pad") for index in indices])
    return all_labels

def predict(model, dataset):
    model.eval()
    discourse_predictions = []
    citation_predictions = []
    span_predictions = []
        
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                               tokenizer.sep_token_id, tokenizer.pad_token_id)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            discourse_out, span_citation_out, _, _ = \
                    model(encoded_dict, transformation_indices, batch["N_tokens"],
                          discourse_label = padded_discourse_label.to(device),
                          span_citation_label = padded_span_citation_label.to(device),
                         )

            discourse_predictions.extend(index2label(discourse_out, dataset.discourse_label_lookup))
            span_citation_seqs = index2label(span_citation_out, dataset.span_citation_label_lookup)
            span_seq, citation_seq = separate_span_citation(span_citation_seq)

            citation_predictions.extend(citation_seq)
            span_predictions.extend(span_seq)
    return discourse_predictions, citation_predictions, span_predictions

def evaluation_metric(discourse_labels, discourse_predictions, mapping):
    positive_labels = []
    for label in mapping.keys():
        if label not in {"Other","O","pad"}:
            positive_labels.append(label)
    
    flatten_labels = flatten(discourse_labels)
    flatten_predictions = flatten(discourse_predictions)
    if len(flatten_labels) != len(flatten_predictions):
        min_len = min([len(flatten_labels), len(flatten_predictions)])
        flatten_labels = flatten_labels[:min_len]
        flatten_predictions = flatten_predictions[:min_len]
    discourse_f1 = f1_score(flatten_labels,flatten_predictions, average='micro', labels=positive_labels)
    discourse_precision = precision_score(flatten_labels,flatten_predictions, average='micro', labels=positive_labels)
    discourse_recall = recall_score(flatten_labels,flatten_predictions, average='micro', labels=positive_labels)
    return (discourse_f1, discourse_recall, discourse_precision)

def separate_span_citation(seqs):
    span_seqs = []
    citation_seqs = []
    for seq in seqs:
        span_seq = []
        citation_seq = []
        for label in seq:
            span, citation = label.split("@")
            span_seq.append(span)
            citation_seq.append(citation)
        span_seqs.append(span_seq)
        citation_seqs.append(citation_seq)
    return span_seqs, citation_seqs

def evaluation(model, dataset):
    model.eval()
    discourse_predictions = []
    discourse_labels = []
    citation_predictions = []
    citation_labels = []
    span_predictions = []
    span_labels = []
        
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            try:
                encoded_dict = encode(tokenizer, batch)
                transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                                   tokenizer.sep_token_id, tokenizer.pad_token_id)
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                transformation_indices = [tensor.to(device) for tensor in transformation_indices]

                padded_discourse_label, discourse_label = batch_token_label(batch["discourse_label"], 0) #len(dev_set.discourse_label_types))
                padded_span_citation_label, span_citation_label = batch_token_label(batch["span_citation_label"], 0) #len(dev_set.span_label_types))
                discourse_out, span_citation_out, _, _ = \
                        model(encoded_dict, transformation_indices, batch["N_tokens"],
                              discourse_label = padded_discourse_label.to(device),
                              span_citation_label = padded_span_citation_label.to(device),
                             )

                discourse_predictions.extend(index2label(discourse_out, dataset.discourse_label_lookup))
                discourse_labels.extend(index2label(discourse_label, dataset.discourse_label_lookup))
                span_citation_seqs = index2label(span_citation_out, dataset.span_citation_label_lookup)
                span_seq, citation_seq = separate_span_citation(span_citation_seq)
                span_label_seq, citation_label_seq = separate_span_citation(span_citation_label)
                
                citation_predictions.extend(citation_seq)
                citation_labels.extend(citation_label_seq)
                span_predictions.extend(span_seq)
                span_labels.extend(span_label_seq)
            except:
                pass
    
    return evaluation_metric(discourse_labels, discourse_predictions, dataset.discourse_label_types), \
evaluation_metric(citation_labels, citation_predictions, dataset.citation_label_types), \
evaluation_metric(span_labels, span_predictions, dataset.span_label_types)



def encode(tokenizer, batch):
    inputs = batch["paragraph"]
    encoded_dict = tokenizer.batch_encode_plus(
        inputs,
        pad_to_max_length=True,add_special_tokens=True,
        return_tensors='pt')
    # Single pass to BERT should not exceed max_sent_len anymore, because it's handled in dataset.py
    return encoded_dict

def token_idx_by_sentence(input_ids, sep_token_id, padding_idx):
    """
    Compute the token indices matrix of the BERT output.
    input_ids: (batch_size, paragraph_len)
    batch_indices, indices_by_batch, mask: (batch_size, N_sentence, N_token)
    bert_out: (batch_size, paragraph_len,BERT_dim)
    bert_out[batch_indices,indices_by_batch,:]: (batch_size, N_sentence, N_token, BERT_dim)
    """
    sep_tokens = (input_ids == sep_token_id).bool()
    paragraph_lens = torch.sum(sep_tokens,1).numpy().tolist() # Number of sentences per paragraph + 1
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(sep_tokens.size(0),-1)
    sep_indices = torch.split(indices[sep_tokens],paragraph_lens)
    paragraph_lens = []
    all_word_indices = []
    for paragraph in sep_indices:
        word_indices = [torch.arange(paragraph[i], paragraph[i+1]) for i in range(paragraph.size(0)-1)]
        paragraph_lens.append(len(word_indices))
        all_word_indices.extend(word_indices)
    
    indices_by_sentence = nn.utils.rnn.pad_sequence(all_word_indices, batch_first=True, padding_value=padding_idx)
    indices_by_sentence_split = torch.split(indices_by_sentence,paragraph_lens)
    indices_by_batch = nn.utils.rnn.pad_sequence(indices_by_sentence_split, batch_first=True, padding_value=padding_idx)
    batch_indices = torch.arange(sep_tokens.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1,indices_by_batch.size(1),indices_by_batch.size(-1))
    mask = (indices_by_batch!=padding_idx) 
    return batch_indices.long(), indices_by_batch.long(), mask.long()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "allenai/scibert_scivocab_uncased", help="Word embedding file")
    argparser.add_argument('--train_file', type=str, default="")
    argparser.add_argument('--distant_file', type=str, default="")
    argparser.add_argument('--pre_trained_model', type=str)
    #argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--test_file', type=str, default="")
    argparser.add_argument('--bert_lr', type=float, default=1e-5, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=768, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=15, help="Training epoch")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "joint_tagger.model")
    argparser.add_argument('--log_file', type=str, default = "joint_tagger_performances.jsonl")
    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=1) # roberta-large: 2; bert: 8
    argparser.add_argument('--discourse_coef', type=float, default=1)
    argparser.add_argument('--span_citation_coef', type=float, default=3)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    reset_random_seed(12345)

    args = argparser.parse_args()
    #device = "cpu" ###############################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    additional_special_tokens = {'additional_special_tokens': ['[BOS]']} 
    tokenizer.add_special_tokens(additional_special_tokens)
    
    if args.train_file:
        train = True
        #assert args.repfile is not None, "Word embedding file required for training."
    else:
        train = False
    if args.test_file:
        test = True
    else:
        test = False

    params = vars(args)

    for k,v in params.items():
        print(k,v)
    
    if train:
        train_set = JointRelatedWorkAnnotationDataset(args.train_file, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN)
        if args.distant_file is not None:
            distant_set = JointRelatedWorkAnnotationDataset(args.distant_file, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN, dummy_discourse=True) #####
            train_set.merge(distant_set)
    dev_set = JointRelatedWorkAnnotationDataset(args.test_file, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN)
    
    model = JointParagraphTagger(args.repfile, args.bert_dim,
                                      args.dropout)#.to(device)

    if args.pre_trained_model is not None:
        model.load_state_dict(torch.load(args.pre_trained_model))
    
    model = model.to(device)
    
    if train:
        settings = [{'params': model.bert.parameters(), 'lr': args.bert_lr}]
        for module in model.extra_modules:
            settings.append({'params': module.parameters(), 'lr': args.lr})
        optimizer = torch.optim.Adam(settings)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epoch)
        model.train()

        prev_performance = 0
        for epoch in range(args.epoch):
            tq = tqdm(DataLoader(train_set, batch_size = args.batch_size, shuffle=True))
            for i, batch in enumerate(tq):
                encoded_dict = encode(tokenizer, batch)
                transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"],
                                                               tokenizer.sep_token_id, tokenizer.pad_token_id)
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                transformation_indices = [tensor.to(device) for tensor in transformation_indices]
                padded_discourse_label, discourse_label = batch_token_label(batch["discourse_label"], 0) #len(dev_set.discourse_label_types))
                padded_span_citation_label, span_citation_label = batch_token_label(batch["span_citation_label"], 0) #len(dev_set.span_label_types))
                discourse_out, span_citation_out, discourse_loss, span_citation_loss = \
                    model(encoded_dict, transformation_indices, batch["N_tokens"],
                          discourse_label = padded_discourse_label.to(device),
                          span_citation_label = padded_span_citation_label.to(device),
                         )

                loss = discourse_loss * args.discourse_coef + span_citation_loss * args.span_citation_coef
                loss.backward()

                if i % args.update_step == args.update_step - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    tq.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
            scheduler.step()

            # Evaluation
            train_discourse_score, train_citation_score, train_span_score = evaluation(model, train_set)
            print(f'Epoch {epoch}, train discourse f1 p r: %.4f, %.4f, %.4f' % train_discourse_score)
            print(f'Epoch {epoch}, train citation f1 p r: %.4f, %.4f, %.4f' % train_citation_score)
            print(f'Epoch {epoch}, train span f1 p r: %.4f, %.4f, %.4f' % train_span_score)
            
            dev_discourse_score, dev_citation_score, dev_span_score = evaluation(model, dev_set)
            print(f'Epoch {epoch}, dev discourse f1 p r: %.4f, %.4f, %.4f' % dev_discourse_score)
            print(f'Epoch {epoch}, dev citation f1 p r: %.4f, %.4f, %.4f' % dev_citation_score)
            print(f'Epoch {epoch}, dev span f1 p r: %.4f, %.4f, %.4f' % dev_span_score)
               
            dev_perf = dev_discourse_score[0] * dev_citation_score[0] *  dev_span_score[0]
            if dev_perf >= prev_performance:
                torch.save(model.state_dict(), args.checkpoint)
                best_state_dict = model.state_dict()
                prev_performance = dev_perf
                best_scores = (dev_discourse_score, dev_citation_score, dev_span_score)
                print("New model saved!")
            else:
                print("Skip saving model.")
        
        #torch.save(model.state_dict(), args.checkpoint)
        params["discourse_f1"] = best_scores[0][0]
        params["discourse_precision"] = best_scores[0][1]
        params["discourse_recall"] = best_scores[0][2]
        
        params["citation_f1"] = best_scores[1][0]
        params["citation_precision"] = best_scores[1][1]
        params["citation_recall"] = best_scores[1][2]
        
        params["span_f1"] = best_scores[2][0]
        params["span_precision"] = best_scores[2][1]
        params["span_recall"] = best_scores[2][2]

        with jsonlines.open(args.log_file, mode='a') as writer:
            writer.write(params)

"""
    if test:
        if train:
            del model
            model = JointParagraphClassifier(args.repfile, args.bert_dim, label_size,
                                              args.dropout).to(device)
            model.load_state_dict(best_state_dict)
            print("Testing on the new model.")
        else:
            model.load_state_dict(torch.load(args.checkpoint))
            print("Loaded saved model.")
        
        # Evaluation
        dev_score = evaluation(model, dev_set)
        print(f'Test f1 p r: %.4f, %.4f, %.4f' % dev_score)
        
        params["f1"] = dev_score[0]
        params["precision"] = dev_score[1]
        params["recall"] = dev_score[2]

        with jsonlines.open(args.log_file, mode='a') as writer:
            writer.write(params)
"""