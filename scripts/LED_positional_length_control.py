import os
import argparse
import numpy as np
import torch
from dataset import (
    CitationTextGenerationDatasetNoCitationType,
    CitationTextGenerationDataset,
)
import pickle
from datasets import Dataset as HuggingfaceDataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
)
# from seq_trainer import SeqToSeqNew as Seq2SeqTrainer
import modelling_led

# compute Rouge score during validation
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def process_data_to_model_inputs(batch, special_tokens=None):
    # tokenize the inputs and labels
    if special_tokens is None:
        special_tokens = ['[Mask]']
        # special_tokens = []
    
    additional_special_tokens_lookup = {token: idx for token, idx in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)}
    special_token_ids = set([additional_special_tokens_lookup[token] for token in special_tokens])
    special_token_ids.add(tokenizer.mask_token_id)
    
    inputs = tokenizer(
        batch["source"],
        padding="max_length",
        truncation=True,
        max_length=args.max_input_length,
        add_special_tokens=True 
    )
    outputs = tokenizer(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=args.max_output_length,
        add_special_tokens=True 
    )

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

    # setting [CLS] token as global attention token for classification
    if args.classification or args.predict_length:
        batch["global_attention_mask"][i_batch][0] = 1
            
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    return batch

def read_write_cached_dataset(dataset_name, suffix, process_func, process_args):
    if process_func == load_dataset:
        suffix += "_"+str(args.batch_size)
    if os.path.isdir(dataset_name):
        if dataset_name[-1] == "/":
            dataset_name = dataset_name[:-1]
        pkl_name = dataset_name + suffix
    else:
        pkl_name = dataset_name.replace(".jsonl", suffix)
    try:
        with open(pkl_name,"rb") as f:
            dataset = pickle.load(f)
        print("Read "+ pkl_name)
    except:
        print("Reading "+ dataset_name +".")
        dataset = process_func(*process_args)
        with open(pkl_name,"wb") as f:
            pickle.dump(dataset, f)
        print("Saved "+ pkl_name)
    return dataset

def load_dataset(dataset_class, tokenizer, dataset_name, suffix, train):
    original_dataset = read_write_cached_dataset(dataset_name, suffix, dataset_class, [dataset_name, tokenizer, train])
    return prepare_dataset(original_dataset)

def prepare_dataset(original_dataset, batch_size=None):
    #print("Converting dataset!",flush=True)
    dataset = HuggingfaceDataset.from_dict(original_dataset.get_dict())
    #print("Starting mapping!",flush=True)
    dataset = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size or args.batch_size,
        remove_columns=["id", "source", "target"],
    )
    #print("End mapping!",flush=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels", "length"],
    )
    return dataset
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "allenai/led-base-16384", help="Word embedding file")
    argparser.add_argument('--train_dataset', type=str)
    argparser.add_argument('--distant_dataset', type=str)
    argparser.add_argument('--dev_dataset', type=str)
    argparser.add_argument('--pre_trained_model', type=str)

    argparser.set_defaults(predict_length=False)
    argparser.add_argument('--classification', dest='classification', action='store_true')
    argparser.set_defaults(classification=False)

    argparser.add_argument('--no_citation_type', dest='no_citation_type', action='store_false')
    argparser.set_defaults(no_citation_type=True)

    argparser.add_argument('--dominant_only', dest='dominant_only', action='store_true')
    argparser.set_defaults(dominant_only=False)

    argparser.add_argument('--no_context', dest='no_context', action='store_true')
    argparser.set_defaults(no_context=False)

    argparser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    argparser.add_argument('--epoch', type=int, default=3, help="Training epoch")
    argparser.add_argument('--max_input_length', type=int, default=16384) #1024
    argparser.add_argument('--max_output_length', type=int, default=1024)
    argparser.add_argument('--sinpostype', type=str, 
        choices=['absolute', 'ratio'], default="ratio",
        help="sinusodial embedding type")
    argparser.add_argument('--checkpoint', type=str, default="./")
    argparser.add_argument('--batch_size', type=int, default=1)

    args = argparser.parse_args()
    torch.manual_seed(12345) # pytorch random seed
    np.random.seed(12345) # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    if args.no_citation_type:
        special_tokens = ['<doc>','</doc>', '[BOS]', '[Mask]', '[B_Mask]',  '[E_Mask]']
    else:
        special_tokens = ['<doc>','</doc>', '[BOS]', '[Dominant]', '[Reference]', '[B_Dominant]',  '[E_Dominant]', '[B_Reference]', '[E_Reference]']

    additional_special_tokens = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(additional_special_tokens)
        
    # Creating the Training and Validation dataset for further creation of Dataloader

    if not args.no_citation_type and args.no_context:
        raise ValueError("no_context is available for no_citation_type scenario")

    if args.no_citation_type:
        training_set = CitationTextGenerationDatasetNoCitationType(args.train_dataset, tokenizer,
                                    MAX_SENT_LEN=args.max_input_length,
                                    include_intro=False,
                                    skip_no_citations=True, 
                                    no_context=args.no_context,
                                    )
    else:
        training_set = CitationTextGenerationDataset(args.train_dataset, tokenizer,
                                    MAX_SENT_LEN=args.max_input_length,
                                    include_intro=False,
                                    skip_no_citations=True, 
                                    )


    if args.distant_dataset is not None:
        if args.no_citation_type:
            distant_dataset = CitationTextGenerationDatasetNoCitationType(args.distant_dataset, tokenizer,
                                    MAX_SENT_LEN=args.max_input_length,
                                    skip_no_citations=True, 
                                    include_intro=False,
                                    no_context=args.no_context,
                                    )
        else:
            distant_dataset = CitationTextGenerationDataset(args.distant_dataset, tokenizer,
                                    MAX_SENT_LEN=args.max_input_length,
                                    skip_no_citations=True, 
                                    include_intro=False,
                                    )

        training_set.merge(distant_dataset)

    if args.dominant_only:
        training_set.filter_citation_type(citation_type="Dominant")   

    print("Sample datapoint: ", training_set[0])

    training_set = prepare_dataset(training_set)
    if args.no_citation_type:
        val_set = CitationTextGenerationDatasetNoCitationType(
            args.dev_dataset, tokenizer, 
            MAX_SENT_LEN = args.max_input_length,
            include_intro=False, 
            skip_no_citations=True, 
            no_context=args.no_context,
        )
    else:
        val_set = CitationTextGenerationDataset(
            args.dev_dataset, tokenizer, 
            MAX_SENT_LEN = args.max_input_length,
            include_intro=False, 
            skip_no_citations=True, 
        )

    if args.dominant_only:
        val_set.filter_citation_type(citation_type="Dominant")

    val_set = prepare_dataset(val_set)
    
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs = args.epoch,
        fp16=True,
        fp16_backend="auto",
        output_dir=args.checkpoint,
        eval_steps = 1000,
        logging_steps=250,
        save_steps=500,
        warmup_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        prediction_loss_only = True,
        overwrite_output_dir = True ,
        learning_rate=args.lr,
        lr_scheduler_type="polynomial",
    )
    
    def set_model_arguments(led_model, args):
        led_model.config.sinpostype = args.sinpostype
        print(led_model.config)
        return led_model

    # load model + enable gradient checkpointing & disable cache for checkpointing
    led = modelling_led.LEDForConditionalGeneration.from_pretrained(
        args.repfile, 
        gradient_checkpointing=True, 
        use_cache=False
        )
    led = set_model_arguments(led, args)

    # set generate hyperparameters
    led.config.num_beams = 4
    led.config.max_length = args.max_output_length
    led.config.min_length = 1
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3
    
    led.resize_token_embeddings(len(tokenizer))
    
    trainer = Seq2SeqTrainer(
        model=led,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=training_set,
        eval_dataset=val_set
    )

    trainer.train()
