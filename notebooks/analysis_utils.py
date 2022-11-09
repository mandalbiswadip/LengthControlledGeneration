from thefuzz import fuzz
from copy import deepcopy
import os
import six
import json
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer

rouge = load_metric("rouge")


def fuzzy_replace(pattern, repl, string):
    l = len(pattern.split())  # Length to read orig_str chunk by chunk
    splitted = string.split()
    for i in range(len(splitted)-l+1):
        test = " ".join(splitted[i:i+l])
        if fuzz.ratio(pattern, test) > 75:  # Using fuzzwuzzy library to test ratio
            before = " ".join(splitted[:i])
            after = " ".join(splitted[i+1:])
            # Output will be sandwich of these three strings
            return before+" "+repl+" "+after, True
    return string, False


def load_json(filename):
    with open(filename, 'r') as f:
        return json.loads(f.read())


def load_accumulated_from_model_folder(file_path):
    with open(os.path.abspath(os.path.join(file_path, "../sample_output.json")), 'r') as f:
        accumulated_data = json.loads(f.read())
    return accumulated_data


def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def write_text(text, filename):
    with open(filename, "w") as f:
        f.write(text)


def remove_sep_tokens(accumulated_data, sep_tokens=None):
    """remove sep tokens from the generation.
    the method doesn't return anything just changes the list

    Args:
        accumulated_data (_type_): _description_
        sep_tokens (_type_, optional): _description_. Defaults to None.
    """

    if sep_tokens is None:
        sep_tokens = ["<s>", "</s>"]
    for index, data in enumerate(accumulated_data):
        for token in sep_tokens:
            data["generated"] = data["generated"].replace(token, "").strip()


class GenerationStandardData(object):

    """A helper class to get a unfied object for generation results"""

    def __init__(self, data, SEP_TOKEN="</s>") -> None:
        for key in ["source", "target", "generated", "part_id"]:
            if key not in data:
                raise ValueError("{} is not found in input data".format(key))
        self.citations = self.get_citations(data["source"])
        def remove_citations(x): return self.remove_citations(
            x, self.citations)

        self.part_id = data["part_id"]

        self.source = data["source"]
        self._current_abstract = self.source.split(SEP_TOKEN)[0]

        self._related_work, self._cited_papers = self.source.split(SEP_TOKEN)[1].split("\n\n")[0], \
            self.source.split("\n\n")[1]

        self._related_work = remove_citations(self._related_work)
        self._cited_papers = remove_citations(self._cited_papers)

        self._target = data["target"]
        self._target = remove_citations(self._target)
        self._generated_text = data["generated"]
        self._generated_text = remove_citations(self._generated_text)

        self._citation_type = None
        if "[Dominant]" in data["source"]:
            self._citation_type = "Dominant"
        else:
            self._citation_type = "Reference"

    @property
    def current_abstract(self):
        return self._current_abstract

    @property
    def related_work(self):
        return self._related_work

    @property
    def cited_papers(self):
        return self._cited_papers

    @property
    def target(self):
        return self._target

    @property
    def generated_text(self):
        return self._generated_text

    @property
    def citation_type(self):
        return self._citation_type

    @staticmethod
    def get_citations(src):
        """Get citations given source content"""
        all_citations = []
        for cite_data in src.split("[B_Reference]")[1:]:

            all_citations.append(cite_data.split("</s>")[0].strip())

        for cite_data in src.split("[B_Dominant]")[1:]:

            all_citations.append(cite_data.split("</s>")[0].strip())

        return all_citations

    def remove_citations(self, text, citations):
        for citation in citations:
            text = text.replace(citation, " ")
        return text

    def __str__(self) -> str:
        return "target:\n{}\n\ngenerated:\n{}".format(self.target.strip(),
                                                      self.generated_text.strip())

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def rouge(self):
        if hasattr(self, "_rouge"):
            return self._rouge
        rr = rouge.compute(
            predictions=[self._generated_text],
            references=[self._target],
            rouge_types=["rouge1"]
        )

        self._rouge = rr["rouge1"].mid.fmeasure
        return self._rouge


def strip_context(paragraph):
    """strip context from paragraph and only return span"""
    return paragraph.split("<context>")[1].split("</context>")[0].strip()


def get_context(paragraph):
    """strip context from paragraph and only return span"""
    return paragraph.split("<context>")[0] + " " + paragraph.split("</context>")[1]


def get_citations(src):
    """Get citations given source content"""
    all_citations = []
    for cite_data in src.split("[B_Reference]")[1:]:

        all_citations.append(cite_data.split("</s>")[0].strip())

    for cite_data in src.split("[B_Dominant]")[1:]:

        all_citations.append(cite_data.split("</s>")[0].strip())

    for cite_data in src.split("[B_Mask]")[1:]:

        all_citations.append(cite_data.split("</s>")[0].strip())

    return all_citations


def strip_context_from_generation(para_accumulated_data):
    """strip context from generation data.
    remove datapoints with improper context separation"""
    good_datapoints = []
    for data in para_accumulated_data:
        # remove the reference to avoid editing passed list
        data = dict(data)
        gen = data["generated"]
        target = data["target"]

        if "<context>" in gen and "</context>" in gen:
            gen = strip_context(gen)
            target = strip_context(target)
            data["target"] = target
            data["generated"] = gen
            good_datapoints.append(data)
    return good_datapoints


get_valid_paragraph_datapoints = strip_context_from_generation



def if_source_domiant(source):
    if "[Dominant]" in source or "[B_Dominant]" in source:
        return True

    if "[Reference]" in source:
        return False

    return None

def split_data_by_citation_type(accumulated_data):
    dominant_data, reference_data = [], []

    for data in accumulated_data:
        res = if_source_domiant(data["source"])
        if res is None:
            print("Warning: datapoint couldn't be categorized into any citation type")
        elif res :
            dominant_data.append(data)
        else:
            reference_data.append(data)
    return dominant_data, reference_data


def remove_citation_from_sentence(sentence, source):
    """remove citation marks from sentence"""
    sentence = sentence.replace(",", "").replace(".", "")

    for c in get_citations(source):
        c = c.replace(",", "").replace(".", "")
        sentence = sentence.replace(c, "")
    return sentence


def remove_citation_get_citation_count(sentence, source):
    """remove citation marks from sentence"""
    sentence = sentence.replace(",", "").replace(".", "")
    target_cite_count = 0
    gen_cite_count = 0

    for c in get_citations(source):
        target_cite_count += 1
        c = c.replace(",", "").replace(".", "")
        if c in sentence:
            gen_cite_count += 1

        sentence = sentence.replace(c, "")
    return sentence, target_cite_count, gen_cite_count


class ParagraphGeneration(object):
    """util operations for paragraph generation text"""

    def __init__(self, text, source=None) -> None:

        if not isinstance(text, six.string_types):
            raise ValueError(
                "String input is expected for text! received {}".format(type(text)))
        self.text = text
        self.valid = True
        self.source = source

        if "<context>" in text and "</context>" in text:
            self.span_text = strip_context(self.text)
        else:
            self.valid = False

        if self.source:
            self._span_clean_text = remove_citation_from_sentence(
                self.span_text, self.source
            )

        self.context = get_context(self.text)

    @property
    def span_clean_text(self):
        if self.source is None:
            raise ValueError("source is needed to remove citation marks!!")

        return self._span_clean_text

    def __str__(self) -> str:
        return self.span_text

    def __repr__(self) -> str:
        return self.__str__()


def remove_eos_tokens(old_accumulated_data):
    accumulated_data = deepcopy(old_accumulated_data)
    sep_tokens = ["<s>", "</s>"]
    for index, data in enumerate(accumulated_data):
        for token in sep_tokens:
            data["generated"] = data["generated"].replace(token, "").strip()
    return accumulated_data


def get_rouge1_f1(predicted, references):
    return rouge.compute(
        predictions=predicted,
        references=references,
        rouge_types=["rouge1"],
        use_stemmer=True
    )["rouge1"].mid.fmeasure


def get_rougeL_f1(predicted, references):
    return rouge.compute(
        predictions=predicted,
        references=references,
        rouge_types=["rougeL"],
        use_stemmer=True
    )["rougeL"].mid.fmeasure


def generation_length_difference(s1, s2, tokenizer):
    return tokenizer.tokenize(s1).__len__() - tokenizer.tokenize(s2).__len__()


def remove_punctuations_from_citations(citation):
    return citation.replace(",", "").replace(":", "").replace(".", "")


def get_length_variance_from_data(all_data, tokenizer):
    """get length variance between target and genetation"""
    return .001* 1 / len(all_data) * sum([ (tokenizer.tokenize(x["target"]).__len__() - tokenizer.tokenize(x["generated"]).__len__())**2 for x in all_data])


def get_length_variance_from_data_list(l1_list, l2_list):
    """get length variance between two length lists"""
    assert len(l1_list) == len(l2_list), "both list should have same lengths"
    return .001* 1 / len(l1_list) * sum([ ( l1 - l2 )**2 for l1, l2 in zip(l1_list, l2_list)])



def get_mae_from_data(l1_list, l2_list):
    """get length variance between two length lists"""
    assert len(l1_list) == len(l2_list), "both list should have same lengths"
    return  1 / len(l1_list) * sum([ abs( l1 - l2 ) for l1, l2 in zip(l1_list, l2_list)])



def get_utility_tokenizer():
    path = "/home/bxm200000/models/dominant_only/led_generations/par_v1_cdlm/checkpoint-60500/"
    tokenizer = AutoTokenizer.from_pretrained(path)
    special_tokens = ['<doc>','</doc>', '[BOS]', '[Dominant]', '[Reference]', '[B_Dominant]',  '[E_Dominant]', '[B_Reference]', '[E_Reference]', '<context>', '</context>']
    additional_special_tokens = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(additional_special_tokens)
    return tokenizer


