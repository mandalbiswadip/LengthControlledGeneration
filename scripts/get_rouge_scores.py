import argparse
import json
import sys

from datasets import load_dataset, load_metric

sys.path.append("../notebooks")

import analysis_utils as utils


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--output_file', type=str,
                           help="output json file path")
    argparser.add_argument('--generation_type', type=str, help="span or paragraph generation", 
                            choices=["span", "paragraph"])

    args = argparser.parse_args()


    with open(args.output_file, "r") as f:
        raw_data = json.loads(f.read())


    predicted, reference = [], []
    dominant_predicted, dominant_reference = [], []
    reference_predicted, reference_reference = [], []
    target_cite_count, gen_cite_count = 0, 0

    if args.generation_type == "paragraph":
        print("Total raw datapoints: ", len(raw_data))

        accumulated_data = utils.get_valid_paragraph_datapoints(raw_data)
        

    elif args.generation_type == "span":
        accumulated_data = raw_data

    for data in accumulated_data:
            
        target = data["target"]
        gen = data["generated"]

        # remove citations         
        gen, t, g  = utils.remove_citation_get_citation_count(
            gen, source=data["source"])
        target = utils.remove_citation_from_sentence(target, source=data["source"])
        target_cite_count += t
        gen_cite_count += g 


        if "[Dominant]" in data["source"]:
            dominant_predicted.append(gen)
            dominant_reference.append(target)
            
            
        if "[Reference]" in data["source"]:
            reference_predicted.append(gen)
            reference_reference.append(target)

        predicted.append(gen)
        reference.append(target)

    print("Total number of citation marks in target data ", target_cite_count)
    print("Total number of citation marks in generated data ", gen_cite_count)

    rouge = load_metric("rouge")

    print("Overall result")
    print( rouge.compute(predictions=predicted, references=reference, rouge_types=["rouge1","rouge2","rougeL"]))

    print("\n")
    print("Dominant results")
    print(rouge.compute(predictions=dominant_predicted, references=dominant_reference, rouge_types=["rouge1","rouge2","rougeL"]))

    print("\n")
    print("Reference results")
    print(rouge.compute(predictions=reference_predicted, references=reference_reference, rouge_types=["rouge1","rouge2","rougeL"]))