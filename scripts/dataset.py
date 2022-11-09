from glob import glob
from collections import defaultdict
from lib2to3.pgen2 import token

from torch.utils.data import Dataset
from tqdm import tqdm

from config import CITATION_TYPES
from util import *
import traceback

class DiscourseTaggingDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, dataset: str, sep_token="</s>", train=True,
                 label_ind=None):
        paper_ids, str_seqs, label_seqs = read_passages_json(dataset, train)
        str_seqs = clean_words(str_seqs)

        self.paper_ids = paper_ids
        self.str_seqs = str_seqs
        self.label_seqs = label_seqs

        if not label_ind:
            self.label_ind = {}
        else:
            self.label_ind = label_ind

        if len(self.label_ind) == 0:
            if train:
                for str_seq, label_seq in zip(str_seqs, label_seqs):
                    for label in label_seq:
                        if label not in self.label_ind:
                            # Add new labels with values 0,1,2,....
                            self.label_ind[label] = len(self.label_ind)
            else:
                assert ("No label_ind provided.")
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}

        self.samples = []
        if train:
            for str_seq, label_seq in zip(str_seqs, label_seqs):
                label_string = ""
                assert (len(str_seq) == len(label_seq))
                for label in label_seq:
                    label_string += str(self.label_ind[label])
                concat_sentences = (" " + sep_token + " ").join(str_seq)

                self.samples.append({
                    'paragraph': concat_sentences,
                    'label': label_string
                })

        else:
            concat_sentences = (" " + sep_token + " ").join(str_seq)

            self.samples.append({
                'paragraph': concat_sentences
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CORWADatasetTextFile(Dataset):

    def __init__(self, folder, tokenizer, citation_types=None, context_window=2,
                 citation_replacement=None, MAX_SENT_LEN=512) -> None:
        self.max_sent_len = MAX_SENT_LEN
        if citation_types is None:
            self.citation_types = CITATION_TYPES
        else:
            self.citation_types = citation_types

        files = glob(os.path.join(folder, "*.txt"))

        self.samples = []

        for file in files:
            ann_file = file.replace(".txt", ".ann")

            with open(file, "r") as textread:
                file_text = textread.read()
            if not os.path.exists(ann_file):
                raise FileNotFoundError("file: {}".format(ann_file))

            with open(ann_file, "r") as annread:
                ann_text = annread.read()

            citations = []
            for line in ann_text.splitlines():
                tag_no, tags, text = line.split("\t")
                tag, start, end = tags.split()
                if tag in self.citation_types:
                    start, end = int(start), int(end)
                    citations.append((start, end, tag, text))

            # sort based on starting index
            citations = sorted(citations, key=lambda x: x[0])

            position, i = 0, 0
            paragraphs = file_text.split("\n\n")
            next_par_position = paragraphs[0].__len__() + 2

            for start, end, tag, text in citations:

                if start >= next_par_position:
                    while start >= next_par_position:
                        i += 1
                        position = next_par_position
                        next_par_position += paragraphs[i].__len__() + 2

                paragraph = file_text[position: next_par_position]
                par_start, par_end = start - position, end - position
                context_text = paragraph[
                               :par_start] + "_CITE_" + paragraph[
                                                        par_end:]

                context_sents = context_text.splitlines()
                for idx, ln in enumerate(context_sents):
                    if "_CITE_" in ln:
                        context_window_text = context_sents[
                                              max(idx - context_window, 0):min(
                                                  idx + context_window + 1, len(
                                                      context_sents))]
                        context = "".join(context_window_text).replace(
                            "[BOS]", "").replace("\n", " ").strip()
                        if citation_replacement is None:
                            context = context.replace("_CITE_", text)
                        else:
                            context = context.replace("_CITE_",
                                                      citation_replacement)

                        offset_mapping = \
                            tokenizer(context,
                                      return_offsets_mapping=True)[
                                "offset_mapping"]

                        if len(offset_mapping) > self.max_sent_len:
                            context = context[
                                      :offset_mapping[self.max_sent_len - 2][
                                          -1]]

                        self.samples.append({
                            "paper_id": os.path.splitext(
                                os.path.basename(file)
                            )[0],
                            "citation_tag": tag,
                            "start": start,
                            "end": end,
                            "citation_text": text,
                            "sentence": context
                        })
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SciCiteDatasetTextFile(CORWADatasetTextFile):

    def __init__(self, folder, tokenizer, citation_types=None,
                 MAX_SENT_LEN=512) -> None:
        self.label_types = {'background': 0, 'method': 1, 'result': 2}
        self.label_lookup = {v: k for k, v in self.label_types.items()}
        super(SciCiteDatasetTextFile, self).__init__(folder=folder,
                                                     tokenizer=tokenizer,
                                                     citation_types=citation_types,
                                                     context_window=0,
                                                     citation_replacement=None,
                                                     MAX_SENT_LEN=MAX_SENT_LEN)


class SciCiteDataset(Dataset):
    def __init__(self, path_name, tokenizer, train=True, MAX_SENT_LEN=512):
        self.max_sent_len = MAX_SENT_LEN
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.label_types = {'background': 0, 'method': 1, 'result': 2}
        self.label_lookup = {v: k for k, v in self.label_types.items()}

        self.samples = []
        with open(path_name, "r") as f:
            for line in f:
                sent_json = json.loads(line)
                offset_mapping = \
                    tokenizer(sent_json["string"], return_offsets_mapping=True)[
                        "offset_mapping"]
                if len(offset_mapping) > self.max_sent_len:
                    sentence = sent_json["string"][
                               :offset_mapping[self.max_sent_len - 2][-1]]
                else:
                    sentence = sent_json["string"]
                if train:
                    self.samples.append({
                        "id": sent_json["unique_id"],
                        "sentence": sentence,
                        "label": self.label_types[sent_json["label"]]
                    })
                else:
                    self.samples.append({
                        "id": sent_json["unique_id"],
                        "sentence": sentence
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CitationFunctionDatasetTextFile(CORWADatasetTextFile):
    """For loading CORWA dataset for citation function prediction"""

    def __init__(self, folder, tokenizer, citation_types=None,
                 MAX_SENT_LEN=512) -> None:
        self.label_types = {
            'Background': 0,
            'CompareOrContrast': 1,
            'Extends': 2,
            'Future': 3,
            'Motivation': 4,
            'Uses': 5
        }
        self.label_lookup = {v: k for k, v in self.label_types.items()}

        super(CitationFunctionDatasetTextFile, self).__init__(folder=folder,
                                                              tokenizer=tokenizer,
                                                              citation_types=citation_types,
                                                              context_window=0,
                                                              citation_replacement=None,
                                                              MAX_SENT_LEN=MAX_SENT_LEN)


class CitationFunctionDataset(Dataset):
    def __init__(self, path_name, tokenizer, train=True, MAX_SENT_LEN=512,
                 boc="<boc>", eoc="<eoc>"):
        self.max_sent_len = MAX_SENT_LEN
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.label_types = {
            'Background': 0,
            'CompareOrContrast': 1,
            'Extends': 2,
            'Future': 3,
            'Motivation': 4,
            'Uses': 5
        }

        self.label_lookup = {v: k for k, v in self.label_types.items()}
        files = glob(os.path.join(path_name, "*"))
        self.samples = []
        for file in files:
            with open(file) as f:
                json_dict = json.load(f)
                for citation in json_dict["citation_contexts"]:
                    citation_text = citation["cite_context"]
                    citation_mark = restore_citation_mark(
                        citation["citing_string"])
                    if "citation_role" in citation:
                        label = citation["citation_role"]
                    elif "citation_function" in citation:
                        label = citation["citation_function"]
                    else:
                        continue
                    ID = citation["citation_id"]
                    if citation_mark not in citation_text:
                        continue
                    start_index = citation_text.find(citation_mark)
                    end_index = start_index + len(citation_mark)
                    augmented_citation = citation_text[
                                         :start_index] + boc + " " + citation_mark + " " + eoc + citation_text[
                                                                                                 end_index:]

                    self.samples.append({
                        "id": ID,
                        "sentence": augmented_citation,
                        "label": self.label_types[label]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SciResDatasetTextFile(CORWADatasetTextFile):
    """For loading CORWA dataset for scires citation function prediction"""

    # TODO include max_len argument
    def __init__(self, folder, tokenizer, citation_types=None,
                 MAX_SENT_LEN=512) -> None:
        self.label_types = {
            'Compare': 0,
            'Extent': 1,
            'Introduce': 2,
            'Produce': 3,
            'Use': 4,
            'Other': 5
        }
        self.label_lookup = {v: k for k, v in self.label_types.items()}

        super(SciResDatasetTextFile, self).__init__(folder=folder,
                                                    tokenizer=tokenizer,
                                                    citation_types=citation_types,
                                                    context_window=2,
                                                    citation_replacement="__CITE__",
                                                    MAX_SENT_LEN=MAX_SENT_LEN)


class SciResDataset(Dataset):
    def __init__(self, path_name, tokenizer, train=True, MAX_SENT_LEN=512):
        self.max_sent_len = MAX_SENT_LEN
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.first_label_types = {'Material': 0, 'Method': 1, 'Supplement': 2}
        self.first_label_lookup = {v: k for k, v in
                                   self.first_label_types.items()}
        self.second_label_types = {
            'Algorithm': 0,
            'Code': 1,
            'Data': 2,
            'Document': 3,
            'License': 4,
            'Media': 5,
            'Paper': 6,
            'Tool': 7,
            'Website': 8
        }
        self.second_label_lookup = {v: k for k, v in
                                    self.second_label_types.items()}
        self.label_types = {
            'Compare': 0,
            'Extent': 1,
            'Introduce': 2,
            'Produce': 3,
            'Use': 4,
            'Other': 5
        }
        self.label_lookup = {v: k for k, v in self.label_types.items()}

        self.samples = []
        with open(path_name, "r") as f:
            for i, line in enumerate(f):
                text, label_string = line.strip().split("__label__")
                first, second, function = label_string.split("|")
                offset_mapping = tokenizer(text, return_offsets_mapping=True)[
                    "offset_mapping"]
                if len(offset_mapping) > self.max_sent_len:
                    sentence = text[:offset_mapping[self.max_sent_len - 2][-1]]
                else:
                    sentence = text
                if train:
                    self.samples.append({
                        "id": str(i),
                        "sentence": sentence,
                        "label": self.label_types[function],
                        "first_label": self.first_label_types[first],
                        "second_label": self.second_label_types[second]
                    })
                else:
                    self.samples.append({
                        "id": str(i),
                        "sentence": sentence
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class JointRelatedWorkAnnotationMergeLabelDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=512,
                 sentence_overlap=0, dummy_discourse=False, dummy_span=False,
                 dummy_citation=False):
        self.max_sent_len = MAX_SENT_LEN
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap
        self.discourse_label_types = {"pad": 0,
                                      "Intro": 1,
                                      "Single_summ": 2,
                                      "Multi_summ": 3,
                                      "Narrative_cite": 4,
                                      "Reflection": 5,
                                      "Transition": 1,
                                      "Other": 6
                                      }
        self.discourse_label_lookup = {0: "pad",
                                       1: "Transition",
                                       2: "Single_summ",
                                       3: "Multi_summ",
                                       4: "Narrative_cite",
                                       5: "Reflection",
                                       6: "Other"
                                       }

        self.span_label_types = {"pad": 0, "O": 1, "B_span": 2, "I_span": 3}
        self.span_label_lookup = {v: k for k, v in
                                  self.span_label_types.items()}

        self.citation_label_types = {"pad": 0, "O": 1, "B_Dominant": 2,
                                     "I_Dominant": 3, "B_Reference": 4,
                                     "I_Reference": 5}
        self.citation_label_lookup = {v: k for k, v in
                                      self.citation_label_types.items()}

        self.idx2letter = {i: chr(i + 97) for i in range(26)}

        self.span_citation_label_types = {}
        for i in range(len(self.span_label_lookup)):
            for j in range(len(self.citation_label_lookup)):
                self.span_citation_label_types[self.span_label_lookup[i] + "@" +
                                               self.citation_label_lookup[
                                                   j]] = i * len(
                    self.citation_label_lookup) + j

        self.span_citation_label_lookup = {v: k for k, v in
                                           self.span_citation_label_types.items()}

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.samples = []
        for text_file in text_files:
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len, self.sentence_overlap)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))
            try:
                if train:
                    annotation_file = text_file.replace(".txt", ".ann")
                    all_annotations = read_annotations(annotation_file, offsets)
                    for paragraph_id, paragraph, paragraph_annotation in zip(
                            paragraph_ids, paragraphs, all_annotations):
                        for annotation in paragraph_annotation:
                            assert paragraph[annotation[0]:annotation[1]] == \
                                   annotation[-1]
                        # paragraph = paragraph.lower().replace("[bos]","[BOS]") ######
                        # N_tokens = len(tokenizer.tokenize(paragraph)) + 2 # CLS and SEP
                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation, paragraph,
                            self.discourse_label_types)
                        # validate_span_annotation(paragraph_annotation)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        # span_BIO_labels = get_span_BIO_labels(span_indices, paragraph, tokenizer)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        # citation_BIO_labels = get_citation_BIO_labels(citation_mark_span_indices, paragraph, tokenizer)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)

                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        if dummy_discourse:
                            discourse_label_string = "".join([self.idx2letter[
                                                                  len(
                                                                      self.discourse_label_types)]
                                                              for label in
                                                              discourse_labels])
                        else:
                            discourse_label_string = "".join([self.idx2letter[
                                                                  self.discourse_label_types[
                                                                      label]]
                                                              for label in
                                                              discourse_labels])

                        span_citation_label_string = "".join([self.idx2letter[
                                                                  self.span_citation_label_types[
                                                                      span + "@" + citation]]
                                                              for span, citation
                                                              in zip(
                                span_BIO_labels, citation_BIO_labels)])
                        self.samples.append({
                            'id': paragraph_id,
                            'paragraph': paragraph.replace("[BOS]",
                                                           tokenizer.sep_token),
                            'discourse_label': discourse_label_string,
                            'span_citation_label': span_citation_label_string,
                            'N_tokens': N_tokens
                        })
                else:
                    for paragraph_id, paragraph in zip(paragraph_ids,
                                                       paragraphs):
                        N_tokens = len(
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"])
                        self.samples.append({
                            'id': paragraph_id,
                            'paragraph': paragraph.replace("[BOS]",
                                                           tokenizer.sep_token),
                            'N_tokens': N_tokens
                        })
            except:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)


class JointRelatedWorkAnnotationDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=512,
                 sentence_overlap=0, dummy_discourse=False, dummy_span=False,
                 dummy_citation=False):
        self.max_sent_len = MAX_SENT_LEN
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap
        self.discourse_label_types = {"pad": 0,
                                      "Intro": 1,
                                      "Single_summ": 2,
                                      "Multi_summ": 3,
                                      "Narrative_cite": 4,
                                      "Reflection": 5,
                                      "Transition": 1,
                                      "Other": 6
                                      }
        self.discourse_label_lookup = {0: "pad",
                                       1: "Transition",
                                       2: "Single_summ",
                                       3: "Multi_summ",
                                       4: "Narrative_cite",
                                       5: "Reflection",
                                       6: "Other"
                                       }

        self.span_label_types = {"pad": 0, "O": 1, "B_span": 2, "I_span": 3}
        self.span_label_lookup = {v: k for k, v in
                                  self.span_label_types.items()}

        self.citation_label_types = {"pad": 0, "O": 1, "B_Dominant": 2,
                                     "I_Dominant": 3, "B_Reference": 4,
                                     "I_Reference": 5}
        self.citation_label_lookup = {v: k for k, v in
                                      self.citation_label_types.items()}

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.samples = []
        for text_file in text_files:
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))
            if train:
                if "requirements" in text_file:
                    continue
                try:
                    annotation_file = text_file.replace(".txt", ".ann")
                    all_annotations = read_annotations(annotation_file, offsets)

                    for paragraph_id, paragraph, paragraph_annotation in zip(
                            paragraph_ids, paragraphs, all_annotations):
                        for annotation in paragraph_annotation:
                            assert paragraph[annotation[0]:annotation[1]] == \
                                   annotation[-1]
                        # paragraph = paragraph.lower().replace("[bos]","[BOS]") ######
                        # N_tokens = len(tokenizer.tokenize(paragraph)) + 2 # CLS and SEP
                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation, paragraph,
                            self.discourse_label_types)
                        # validate_span_annotation(paragraph_annotation)
                        try:
                            span_indices = read_span_indices(
                                paragraph_annotation,
                                paragraph)
                        except:
                            continue
                        # span_BIO_labels = get_span_BIO_labels(span_indices, paragraph, tokenizer)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        # citation_BIO_labels = get_citation_BIO_labels(citation_mark_span_indices, paragraph, tokenizer)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)

                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        if dummy_discourse:
                            discourse_label_string = "".join(
                                [str(len(self.discourse_label_types)) for label
                                 in
                                 discourse_labels])
                        else:
                            discourse_label_string = "".join(
                                [str(self.discourse_label_types[label]) for
                                 label in
                                 discourse_labels])
                        if dummy_span:
                            # Put placeholder indices
                            span_label_string = "".join(
                                [str(len(self.span_label_types)) for label in
                                 span_BIO_labels])
                        else:
                            span_label_string = "".join(
                                [str(self.span_label_types[label]) for label in
                                 span_BIO_labels])

                        if dummy_citation:
                            citation_label_string = "".join(
                                [str(len(self.citation_label_types)) for label
                                 in
                                 citation_BIO_labels])
                        else:
                            citation_label_string = "".join(
                                [str(self.citation_label_types[label]) for label
                                 in
                                 citation_BIO_labels])

                        self.samples.append({
                            'id': paragraph_id,
                            'paragraph': paragraph.replace("[BOS]",
                                                           tokenizer.sep_token),
                            'discourse_label': discourse_label_string,
                            'span_label': span_label_string,
                            'citation_label': citation_label_string,
                            'N_tokens': N_tokens
                        })
                except:
                    continue

            else:
                for paragraph_id, paragraph in zip(paragraph_ids, paragraphs):
                    N_tokens = len(
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"])
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': paragraph.replace("[BOS]",
                                                       tokenizer.sep_token),
                        'N_tokens': N_tokens
                    })

    def update_samples(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)


class CORWAanalysisDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, context_window=2,
                 MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl"
                 ):
        self.max_sent_len = MAX_SENT_LEN
        self.context_window = context_window
        self.discourse_label_types = {"pad": 0,
                                      "Intro": 1,
                                      "Single_summ": 2,
                                      "Multi_summ": 3,
                                      "Narrative_cite": 4,
                                      "Reflection": 5,
                                      "Transition": 1,
                                      "Other": 6
                                      }
        self.discourse_label_lookup = {0: "pad",
                                       1: "Transition",
                                       2: "Single_summ",
                                       3: "Multi_summ",
                                       4: "Narrative_cite",
                                       5: "Reflection",
                                       6: "Other"
                                       }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        for text_file in text_files:
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))

            annotation_file = text_file.replace(".txt", ".ann")
            all_annotations = read_annotations(annotation_file, offsets)
            for paragraph_id, paragraph, paragraph_annotation in zip(
                    paragraph_ids, paragraphs, all_annotations):
                for annotation in paragraph_annotation:
                    assert paragraph[annotation[0]:annotation[1]] == annotation[
                        -1]
                tokens = tokenizer.tokenize(paragraph, add_special_tokens=True)
                # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                sentences = [sent for sent in paragraph.split("[BOS] ")[1:]]

                offset_mapping = \
                    tokenizer(paragraph, return_offsets_mapping=True)[
                        "offset_mapping"]
                N_tokens = len(offset_mapping)
                discourse_labels = read_discourse_labels(paragraph_annotation,
                                                         paragraph,
                                                         self.discourse_label_types)
                discourse_labels = ["Transition" if disc == "Intro" else disc
                                    for disc in discourse_labels]

                span_indices = read_span_indices(paragraph_annotation,
                                                 paragraph)
                span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                         offset_mapping)
                citation_mark_span_indices = read_citation_mark(
                    paragraph_annotation, paragraph)
                citation_BIO_labels = get_aligned_BIO_labels(
                    citation_mark_span_indices, offset_mapping)

                # print(tokenizer.tokenize(paragraph))
                assert (N_tokens == len(span_BIO_labels) == len(
                    citation_BIO_labels))
                # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                #    continue

                # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                    paragraph_id, sentences, self.related_work_jsons,
                    offset_mapping, citation_BIO_labels, separator="[BOS] ")
                paragraph_citation_links_pre = new_sentence_citation_link(
                    pargraph_citation_info, len(sentences))
                # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                span_citation_mapping = map_span_citation(span_BIO_labels,
                                                          citation_BIO_labels,
                                                          pargraph_citation_info,
                                                          offset_mapping)

                span_sent_mapping, i_span = new_span_sentence_map(tokens,
                                                                  span_BIO_labels,
                                                                  bos="[BOS]")
                paragraph_citation_links = propagate_citation_cross_sentences(
                    span_sent_mapping, paragraph_citation_links_pre, i_span)
                citation_type_by_sentence = citation_by_sentence(tokens,
                                                                 citation_BIO_labels)
                self.dataset.append({
                    "paragraph_id": paragraph_id,
                    "paragraph": paragraph,
                    # "related_work": augmented_paragraph,
                    "citation_links_by_sentence": paragraph_citation_links,
                    # "augmented_sentences": augmented_sentences,
                    "discourse_labels": discourse_labels,
                    "sentences": sentences,
                    # "span_labels": span_BIO_labels,
                    # "citation_labels": citation_BIO_labels,
                    "span_sent_mapping": span_sent_mapping,
                    # "tokens": tokens
                    # "i_span": i_span,
                    "span_citation_mapping": span_citation_mapping,
                    # "offset_mapping": offset_mapping,
                    "citation_info": pargraph_citation_info,
                    "citation_type_by_sentence": citation_type_by_sentence
                })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CitationSentenceGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False
                 ):

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    paragraph_data = []
                    for i_span, span in enumerate(span_citation_mapping):
                        source_data = {}
                        start = previous_sentence_end_index_from_span(paragraph,
                                                                      span[
                                                                          "char_start"]) + 1
                        end = next_sentence_start_index_from_span(paragraph,
                                                                  span[
                                                                      "char_end"]) - 1

                        source_data["range"] = [start, end]
                        cited_context = []
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context.append("[B_Reference]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Reference]")
                        source_data["cited_context"] = " ".join(cited_context)
                        paragraph_data.append(source_data)

                    if len(paragraph_data) < 1:
                        continue
                    citation_tracker = 0
                    # Merge sentences having overlapped span
                    paragraph_data.sort(key=lambda x: x["range"][0])
                    result = []
                    current, context = paragraph_data[0]["range"], \
                                       paragraph_data[0]["cited_context"]
                    for i in range(1, len(paragraph_data)):
                        if current[1] >= paragraph_data[i]["range"][0]:
                            current[1] = max(current[1],
                                             paragraph_data[i]["range"][1])
                            context = context + paragraph_data[i][
                                "cited_context"]
                        else:
                            result.append(
                                {"range": current, "cited_context": context})
                            context = paragraph_data[i]["cited_context"]
                            current = paragraph_data[i]["range"]

                    result.append({"range": current, "cited_context": context})

                    for span_data in result:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data["range"]

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 start:end].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        source.append("[Dominant]")
                        source.append(context_after)
                        source.append(span_data["cited_context"])

                        if skip_no_citations and not span_data[
                            "cited_context"].strip():
                            continue
                        source = " ".join(source)

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except Exception as e:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class HumanEvaluationSpanGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False
                 ):

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation_sentence(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    sentence_citation_mapping = defaultdict(list)
                    for i_span, span in enumerate(span_citation_mapping):

                        # regex to find out the index of the citation mark
                        # maintain {(start, end) : {citaion: link}} --> convert to data

                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context = []
                            if citation_mark[-4:] == "_pos":
                                continue
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)

                            citation_pos_mark = citation_mark + "_pos"
                            if citation_pos_mark not in span["span_citation_mapping"]["Dominant"]:
                                continue

                            citation_start, citation_end = span["span_citation_mapping"]["Dominant"][citation_pos_mark][0], \
                                                           span["span_citation_mapping"]["Dominant"][citation_pos_mark][1]
                            start = previous_sentence_end_index_from_span(
                                paragraph,
                                citation_start) + 1
                            end = next_sentence_start_index_from_span(paragraph,
                                                                      citation_end) - 1
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                            sentence_citation_mapping[(start, end)].append(
                                {citation_mark: " ".join(cited_context)})
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context = []
                            if citation_mark[-4:] == "_pos":
                                continue
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)

                            citation_pos_mark = citation_mark + "_pos"
                            if citation_pos_mark not in \
                                    span["span_citation_mapping"]["Reference"]:
                                continue

                            citation_start, citation_end = \
                            span["span_citation_mapping"]["Reference"][
                                citation_pos_mark][0], \
                            span["span_citation_mapping"]["Reference"][
                                citation_pos_mark][1]
                            start = previous_sentence_end_index_from_span(
                                paragraph,
                                citation_start) + 1
                            end = next_sentence_start_index_from_span(paragraph,
                                                                      citation_end) - 1
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                            sentence_citation_mapping[(start, end)].append(
                                {citation_mark: " ".join(cited_context)})

                    citation_tracker = 0
                    for span_data in sentence_citation_mapping:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 start:end].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        source.append(tokenizer.mask_token)
                        source.append(context_after)
                        for cited_context in sentence_citation_mapping[span_data]:
                            source.append(list(cited_context.values())[0])

                        if skip_no_citations and len(sentence_citation_mapping[span_data]) == 0:
                            continue
                        source = " ".join(source)

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CitationSingleSentenceGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False
                 ):

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation_sentence(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    sentence_citation_mapping = defaultdict(list)
                    for i_span, span in enumerate(span_citation_mapping):

                        # regex to find out the index of the citation mark
                        # maintain {(start, end) : {citaion: link}} --> convert to data
                        cite_type = span["span_type"]
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context = []
                            if citation_mark[-4:] == "_pos":
                                continue
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)

                            citation_pos_mark = citation_mark + "_pos"
                            if citation_pos_mark not in span["span_citation_mapping"]["Dominant"]:
                                continue

                            citation_start, citation_end = span["span_citation_mapping"]["Dominant"][citation_pos_mark][0], \
                                                           span["span_citation_mapping"]["Dominant"][citation_pos_mark][1]
                            start = previous_sentence_end_index_from_span(
                                paragraph,
                                citation_start) + 1
                            end = next_sentence_start_index_from_span(paragraph,
                                                                      citation_end) - 1
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                            sentence_citation_mapping[(start, end)].append(
                                {"citation_mark": " ".join(cited_context), "cite_type": cite_type})
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context = []
                            if citation_mark[-4:] == "_pos":
                                continue
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)

                            citation_pos_mark = citation_mark + "_pos"
                            if citation_pos_mark not in \
                                    span["span_citation_mapping"]["Reference"]:
                                continue

                            citation_start, citation_end = \
                            span["span_citation_mapping"]["Reference"][
                                citation_pos_mark][0], \
                            span["span_citation_mapping"]["Reference"][
                                citation_pos_mark][1]
                            start = previous_sentence_end_index_from_span(
                                paragraph,
                                citation_start) + 1
                            end = next_sentence_start_index_from_span(paragraph,
                                                                      citation_end) - 1
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                            sentence_citation_mapping[(start, end)].append(
                                {"citation_mark": " ".join(cited_context), "cite_type": cite_type})

                    citation_tracker = 0
                    for span_data in sentence_citation_mapping:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 start:end].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        source.append(tokenizer.mask_token)
                        source.append(context_after)

                        citations = []
                        for cited_context in sentence_citation_mapping[span_data]:
                            source.append(cited_context["citation_mark"])
                            citations.append(cited_context["cite_type"])
                        fin_citation = ""
                        if "Dominant" in citations:
                            fin_citation = "Dominant"
                        else:
                            fin_citation = "Reference"

                        if skip_no_citations and len(sentence_citation_mapping[span_data]) == 0:
                            continue
                        source = " ".join(source)
                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                "citation_label" : fin_citation
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CitationParagraphGenerationDatasetOld(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True,
                 MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False,
                 context_sep_flag = False,
                 context_sep_tokens = None
                 ):
        if context_sep_tokens is None:
            context_sep_tokens = ["<context>", "</context>"]

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(
                                part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph,
                                      return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(
                            paragraph_annotation,
                            paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(
                            span_indices,
                            offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences,
                            self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    paragraph_data = []
                    for i_span, span in enumerate(span_citation_mapping):
                        source_data = {}
                        start = previous_sentence_end_index_from_span(
                            paragraph,
                            span[
                                "char_start"]) + 1
                        end = next_sentence_start_index_from_span(paragraph,
                                                                  span[
                                                                      "char_end"]) - 1

                        source_data["range"] = [start, end]
                        source_data["citation_range"] = [span["char_start"],
                                                         span["char_end"]]

                        cited_context = []
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(
                                        self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context.append("[B_Reference]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(
                                        self.cited_paper[link]))
                            cited_context.append("[E_Reference]")
                        source_data["cited_context"] = " ".join(
                            cited_context)
                        paragraph_data.append(source_data)

                    if len(paragraph_data) < 1:
                        continue
                    citation_tracker = 0
                    # Merge sentences having overlapped span
                    paragraph_data.sort(key=lambda x: x["range"][0])
                    result = []
                    current, context, citation_ranges = paragraph_data[0][
                                                            "range"], \
                                                        paragraph_data[0][
                                                            "cited_context"], [
                                                            paragraph_data[
                                                                0][
                                                                "citation_range"]]
                    for i in range(1, len(paragraph_data)):
                        if current[1] >= paragraph_data[i]["range"][0]:
                            current[1] = max(current[1],
                                             paragraph_data[i]["range"][1])
                            context = context + paragraph_data[i][
                                "cited_context"]
                            citation_ranges.append(
                                paragraph_data[i]["citation_range"])
                        else:
                            result.append(
                                {"range": current, "cited_context": context,
                                 "citation_ranges": citation_ranges})
                            context = paragraph_data[i]["cited_context"]
                            current = paragraph_data[i]["range"]
                            citation_ranges = [
                                paragraph_data[i]["citation_range"]]

                    result.append(
                        {"range": current, "cited_context": context,
                         "citation_ranges": citation_ranges})

                    for span_data in result:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data["range"]

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        # target = paragraph[
                        #          start:end].replace(
                        #     "[BOS] ", "")
                            
                        if context_sep_flag:
                            target = paragraph[:start] + " {} ".format(context_sep_tokens[0]) + paragraph[start:end] + \
                            " {} ".format(context_sep_tokens[1]) + paragraph[end:]
                            target = target.replace(
                                "[BOS] ", "")
                        else:
                            target = paragraph.replace(
                                "[BOS] ", "")
                                
                        source.append(context_before)
                        source.append("[Dominant]")
                        source.append(context_after)
                        source.append(span_data["cited_context"])

                        if skip_no_citations and not span_data[
                            "cited_context"].strip():
                            continue
                        source = " ".join(source)

                        if len(tokenizer.tokenize(
                                source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                "citations": [paragraph[d[0]:d[1]].replace(
                                    "[BOS] ", "") for d in
                                    span_data["citation_ranges"]]
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except Exception as e:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class GenericCitationGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True,
                 MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False
                 ):

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(
                                part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph,
                                      return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(
                            paragraph_annotation,
                            paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(
                            span_indices,
                            offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences,
                            self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    paragraph_data = []
                    for i_span, span in enumerate(span_citation_mapping):
                        source_data = {}
                        start = previous_sentence_end_index_from_span(
                            paragraph,
                            span[
                                "char_start"]) + 1
                        end = next_sentence_start_index_from_span(paragraph,
                                                                  span[
                                                                      "char_end"]) - 1

                        source_data["range"] = [start, end]
                        source_data["citation_range"] = [span["char_start"],
                                                         span["char_end"]]

                        cited_context = []
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(
                                        self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context.append("[B_Reference]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(
                                        self.cited_paper[link]))
                            cited_context.append("[E_Reference]")
                        source_data["cited_context"] = " ".join(
                            cited_context)
                        paragraph_data.append(source_data)

                    if len(paragraph_data) < 1:
                        continue
                    citation_tracker = 0
                    # Merge sentences having overlapped span
                    paragraph_data.sort(key=lambda x: x["range"][0])
                    result = []
                    current, context, citation_ranges = paragraph_data[0][
                                                            "range"], \
                                                        paragraph_data[0][
                                                            "cited_context"], [
                                                            paragraph_data[
                                                                0][
                                                                "citation_range"]]
                    for i in range(1, len(paragraph_data)):
                        if current[1] >= paragraph_data[i]["range"][0]:
                            current[1] = max(current[1],
                                             paragraph_data[i]["range"][1])
                            context = context + paragraph_data[i][
                                "cited_context"]
                            citation_ranges.append(
                                paragraph_data[i]["citation_range"])
                        else:
                            result.append(
                                {"range": current, "cited_context": context,
                                 "citation_ranges": citation_ranges})
                            context = paragraph_data[i]["cited_context"]
                            current = paragraph_data[i]["range"]
                            citation_ranges = [
                                paragraph_data[i]["citation_range"]]

                    result.append(
                        {"range": current, "cited_context": context,
                         "citation_ranges": citation_ranges})

                    for span_data in result:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data["range"]

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 start:end].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        source.append("[Dominant]")
                        source.append(context_after)
                        source.append(span_data["cited_context"])

                        if skip_no_citations and not span_data[
                            "cited_context"].strip():
                            continue
                        source = " ".join(source)

                        if len(tokenizer.tokenize(
                                source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                "citations": [paragraph[d[0]:d[1]].replace(
                                    "[BOS] ", "") for d in
                                    span_data["citation_ranges"]]
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except Exception as e:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset

class CitationContextSpanSeparateDataset(Dataset):
    """
    Dataset used for generating context and span using multiple decoders or multi-head decoders
    """
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 ):

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)

            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            _, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                        # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        pass
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        citation_marks = []
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)
                        context_before = paragraph[:span["char_start"]].replace(
                            "[BOS] ", "")
                        context_after = paragraph[span["char_end"]:].replace(
                            "[BOS] ", "")

                        target =  paragraph[span["char_start"]:span["char_end"]].replace(
                            "[BOS] ", "")                        
                        
                        source.append(context_before)
                        if len(span["span_citation_mapping"]["Dominant"]) > 0:
                            source.append("[Dominant]")
                        else:
                            source.append("[Reference]")
                        source.append(context_after)


                        citation_marks_flag = False
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Dominant"].items():
                            source.append("[B_Dominant]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link],
                                                       include_result=include_conclusion))
                            source.append("[E_Dominant]")
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Reference"].items():
                            source.append("[B_Reference]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link]))
                            source.append("[E_Reference]")
                        source = " ".join(source)
                        if skip_no_citations and not citation_marks_flag:
                            continue

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            elm = {
                                "id": paragraph_id + "_" + str(i_span) + "_span",
                                "source": source,
                                "target": target,
                                "target_prev": "dummy",
                                "target_next": "dummy",
                                "output_indicator": 0
                            }
                            self.samples.append(elm)
                            elm = {
                                "id": paragraph_id + "_" + str(i_span) + "_context_prev",
                                "source": source,
                                "target": "dummy",
                                "target_prev": context_before,
                                "target_next": "dummy",
                                "output_indicator": 1
                            }
                            self.samples.append(elm)

                            elm = {
                                "id": paragraph_id + "_" + str(i_span) + "_context_next",
                                "source": source,
                                "target": "dummy",
                                "target_prev": "dummy",
                                "target_next": context_after,
                                "output_indicator": 2
                            }
                            self.samples.append(elm)



            except:
                #print("Skip "+paper_id)
                pass

    def filter_citation_type(self, citation_type="Dominant"):
        """ Inplace filter samples based on  citatio_type


        Args:
            citation_type (str, optional): one of ["Dominant", "Reference]. Defaults to "Dominant".

        Returns:
            _type_: filtered sampels
        """
        acceptables = ["Dominant", "Reference"]
        if citation_type  not in acceptables:
            raise ValueError(f"Invalid citation type passed. accepted are {acceptables}")
        cite_type = f"[{citation_type}]"
        samples = [sample for sample in self.samples if cite_type in sample["source"]]
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CitationContextSpanSeparateDatasetSingle(Dataset):
    """
    Dataset used for generating context and span using multiple decoders or multi-head decoders
    This one generated only one datappoint per sample while CitationContextSpanSeparateDataset generates three different datappints for prev context, span and next context
    """
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 ):

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)

            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            _, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                        # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        pass
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        citation_marks = []
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)
                        context_before = paragraph[:span["char_start"]].replace(
                            "[BOS] ", "")
                        context_after = paragraph[span["char_end"]:].replace(
                            "[BOS] ", "")

                        target =  paragraph[span["char_start"]:span["char_end"]].replace(
                            "[BOS] ", "")                        
                        
                        source.append(context_before)
                        if len(span["span_citation_mapping"]["Dominant"]) > 0:
                            source.append("[Dominant]")
                        else:
                            source.append("[Reference]")
                        source.append(context_after)


                        citation_marks_flag = False
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Dominant"].items():
                            source.append("[B_Dominant]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link],
                                                       include_result=include_conclusion))
                            source.append("[E_Dominant]")
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Reference"].items():
                            source.append("[B_Reference]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link]))
                            source.append("[E_Reference]")
                        source = " ".join(source)
                        if skip_no_citations and not citation_marks_flag:
                            continue

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            elm = {
                                "id": paragraph_id + "_" + str(i_span),
                                "source": source,
                                "target": target,
                                "target_prev": context_before,
                                "target_next": context_after
                            }
                            self.samples.append(elm)


            except:
                #print("Skip "+paper_id)
                pass

    def filter_citation_type(self, citation_type="Dominant"):
        """ Inplace filter samples based on  citatio_type


        Args:
            citation_type (str, optional): one of ["Dominant", "Reference]. Defaults to "Dominant".

        Returns:
            _type_: filtered sampels
        """
        acceptables = ["Dominant", "Reference"]
        if citation_type  not in acceptables:
            raise ValueError(f"Invalid citation type passed. accepted are {acceptables}")
        cite_type = f"[{citation_type}]"
        samples = [sample for sample in self.samples if cite_type in sample["source"]]
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CitationParagraphGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 context_sep_flag = False,
                 context_sep_tokens = None,
                 add_length_signal = None,
                 add_same_ciation = False,
                 ):
        if context_sep_tokens is None:
            context_sep_tokens = ["<context>", "</context>"]

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)

            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                        # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        pass
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        citation_marks = []
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)
                        context_before = paragraph[:span["char_start"]].replace(
                            "[BOS] ", "")
                        context_after = paragraph[span["char_end"]:].replace(
                            "[BOS] ", "")
                        span_target = paragraph[
                                 span["char_start"]:span["char_end"]].replace(
                            "[BOS] ", "")

                        if context_sep_flag:
                            target = paragraph[:span["char_start"]] + " {} ".format(context_sep_tokens[0]) + paragraph[span["char_start"]:span["char_end"]] + \
                            " {} ".format(context_sep_tokens[1]) + paragraph[span["char_end"]:]
                            target = target.replace(
                                "[BOS] ", "")
                        else:
                            target = paragraph.replace(
                                "[BOS] ", "")
                        
                        source.append(context_before)
                        if len(span["span_citation_mapping"]["Dominant"]) > 0:
                            source.append("[Dominant]")
                        else:
                            source.append("[Reference]")
                        source.append(context_after)

                        
                        context_length, span_length = None, None
                        if add_length_signal:
                            all_context = f"{context_before} {context_after}"
                            context_length = tokenizer.tokenize(all_context).__len__()
                            # source.append(tokenizer.sep_token)
                            # source.append(str(context_length))
                            # source.append(tokenizer.sep_token)

                        citation_marks_flag = False
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Dominant"].items():
                            source.append("[B_Dominant]")
                            if add_same_ciation:
                                source.append("Author Et el")
                            else:
                                source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link],
                                                       include_result=include_conclusion))
                            source.append("[E_Dominant]")
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Reference"].items():
                            source.append("[B_Reference]")
                            if add_same_ciation:
                                source.append("Author Et el")
                            else:
                                source.append(citation_mark)                            
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link]))
                            source.append("[E_Reference]")
                        source = " ".join(source)
                        if skip_no_citations and not citation_marks_flag:
                            continue

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            elm = {
                                "id": paragraph_id + "_" + str(i_span),
                                "source": source,
                                "target": target,
                                # "full_target": " ".join(sentences)
                                #"citations": "#".join(citation_marks)
                            }
                            if add_length_signal:
                                elm["context_length"] = context_length
                                elm["span_target"] = span_target
                            self.samples.append(elm)
            except:
                #print("Skip "+paper_id)
                pass

    def filter_citation_type(self, citation_type="Dominant"):
        """ Inplace filter samples based on  citatio_type


        Args:
            citation_type (str, optional): one of ["Dominant", "Reference]. Defaults to "Dominant".

        Returns:
            _type_: filtered sampels
        """
        acceptables = ["Dominant", "Reference"]
        if citation_type  not in acceptables:
            raise ValueError(f"Invalid citation type passed. accepted are {acceptables}")
        cite_type = f"[{citation_type}]"
        samples = [sample for sample in self.samples if cite_type in sample["source"]]
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset

class CitationParagraphGenerationDatasetSwitchHead(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 add_length_signal = None
                 ):


        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)

            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                        # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        pass
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        citation_marks = []
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)
                        context_before = paragraph[:span["char_start"]].replace(
                            "[BOS] ", "")
                        # TODO don't take context_length_prev, span_length, context_length_next 
                        # -- the total length of generation is always the summation of these three. it's fine to take span_length as the length of span though
                        context_length_prev = tokenizer.tokenize(context_before).__len__() + 1
                        context_after = paragraph[span["char_end"]:].replace(
                            "[BOS] ", "")
                        context_length_next = tokenizer.tokenize(context_after).__len__() + 1
                        span_target = paragraph[
                                 span["char_start"]:span["char_end"]].replace(
                            "[BOS] ", "")
                        span_length = tokenizer.tokenize(span_target).__len__()

                        target = context_before + " " + span_target + \
                        " " + context_after

                        switch_tokens, length_positions = self.get_token_type_length_positions(
                            context_before_text=context_before, 
                            span_text=span_target, 
                            context_after_text=context_after,
                            tokenizer=tokenizer,
                        )
                        switch_tokens = self.normalize_token_length(switch_tokens, target, tokenizer, extra_tokens_to_add=2)
                        
                        source.append(context_before)
                        if len(span["span_citation_mapping"]["Dominant"]) > 0:
                            source.append("[Dominant]")
                        else:
                            source.append("[Reference]")
                        source.append(context_after)

                        citation_marks_flag = False
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Dominant"].items():
                            source.append("[B_Dominant]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link],
                                                       include_result=include_conclusion))
                            source.append("[E_Dominant]")
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Reference"].items():
                            source.append("[B_Reference]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link]))
                            source.append("[E_Reference]")
                        source = " ".join(source)
                        if skip_no_citations and not citation_marks_flag:
                            continue

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            elm = {
                                "id": paragraph_id + "_" + str(i_span),
                                "source": source,
                                "target": target,
                                "segment_id_string": " ".join([str(x) for x in switch_tokens]),
                                "span_length" : span_length,
                                "context_length_prev" : context_length_prev,
                                "context_length_next" : context_length_next,
                                "length_positions_string" : " ".join([str(x) for x in length_positions]),
                                # "full_target": " ".join(sentences)
                                #"citations": "#".join(citation_marks)
                            }
                            if add_length_signal:
                                elm["span_target"] = span_target
                            self.samples.append(elm)
            except Exception as e:
                #print("Skip "+paper_id)
                # print(traceback.format_exc(e))
                pass

    def get_token_type_length_positions(self, context_before_text, span_text, context_after_text, tokenizer, context_prefix_indicator = 0, span_indicator=1, context_suffix_indicator=2):
        """generate tokens types --> i.e. `prefix prefix prefix span span span suffix suffix`, where prefix and suffix are context

        Args:
            context_before_text (_type_): prfix context text
            span_text (_type_): span text
            context_after_text (_type_): suffix context text
            tokenizer (_type_): tokenizer. 
            context_prefix_indicator (int, optional): number indicating token is prefix context. Defaults to 0.
            span_indicator (int, optional): number indicating token is span. Defaults to 1.
            context_suffix_indicator (int, optional): number indicating token is suffix context. Defaults to 2.

        Returns:
            (list, list): token types, remaining length at each positions.
        """
        tokens = []
        length_positions = []        
        # adding 1 for start of sentence token
        context_before_length = tokenizer.tokenize(context_before_text).__len__() + 1
        [tokens.append(context_prefix_indicator) for _ in range(context_before_length)]
        length_positions.extend(list(range(context_before_length, 0, -1)))

        span_length = tokenizer.tokenize(span_text).__len__() 
        [tokens.append(span_indicator) for _ in range(span_length)]
        length_positions.extend(list(range(span_length, 0, -1)))

        # adding 1 for end of sentence token
        context_after_length = tokenizer.tokenize(context_after_text).__len__() + 1
        [tokens.append(context_suffix_indicator) for _ in range(context_after_length)]
        length_positions.extend(list(range(context_after_length, 0, -1)))

        return tokens, length_positions

    def normalize_token_length(self, mode_tokens, target, tokenizer, extra_tokens_to_add = 0):
        target_tokens_length = tokenizer.encode(target).__len__()

        if len(mode_tokens) > target_tokens_length:
            mode_tokens = mode_tokens[:target_tokens_length]
        elif len(mode_tokens) < target_tokens_length:
            while len(mode_tokens) != target_tokens_length:
                mode_tokens.append(extra_tokens_to_add)
        return mode_tokens

    def filter_citation_type(self, citation_type="Dominant"):
        """ Inplace filter samples based on  citatio_type


        Args:
            citation_type (str, optional): one of ["Dominant", "Reference]. Defaults to "Dominant".

        Returns:
            _type_: filtered sampels
        """
        acceptables = ["Dominant", "Reference"]
        if citation_type  not in acceptables:
            raise ValueError(f"Invalid citation type passed. accepted are {acceptables}")
        cite_type = f"[{citation_type}]"
        samples = [sample for sample in self.samples if cite_type in sample["source"]]
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset



class CitationTextGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 len_control_flag=False,
                 ):

        if conclusion_sections is None:
            conclusion_sections = ['Conclusion', 'Conclusions',
                                   'Conclusion and Future Work',
                                   'Conclusions and Future Work',
                                   'Conclusions and future work']
            # conclusion_sections = ['Experimental Results',
            #                        'Results',
            #                        'Results and Discussion',
            #                        'Results and Analysis',
            #                        'Experiments and Results',
            #                        'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)

            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                        # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        pass
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        citation_marks = []
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)
                        context_before = paragraph[:span["char_start"]].replace(
                            "[BOS] ", "")
                        context_after = paragraph[span["char_end"]:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 span["char_start"]:span["char_end"]].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        if len(span["span_citation_mapping"]["Dominant"]) > 0:
                            source.append("[Dominant]")
                        else:
                            source.append("[Reference]")
                        source.append(context_after)

                        citation_marks_flag = False
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Dominant"].items():
                            source.append("[B_Dominant]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link],
                                                       include_result=include_conclusion))
                            source.append("[E_Dominant]")
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Reference"].items():
                            source.append("[B_Reference]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link]))
                            source.append("[E_Reference]")
                        if len_control_flag:
                            len_control_token = "[SHORT]" if len(target) < 150 else "[LARGE]"
                            source.append(len_control_token)
                        source = " ".join(source)
                        if skip_no_citations and not citation_marks_flag:
                            continue

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(i_span),
                                "source": source,
                                "target": target,
                                # "full_target": " ".join(sentences)
                                #"citations": "#".join(citation_marks)
                            })
            except:
                #print("Skip "+paper_id)
                pass

    def filter_citation_type(self, citation_type="Dominant"):
        """ Inplace filter samples based on  citatio_type


        Args:
            citation_type (str, optional): one of ["Dominant", "Reference]. Defaults to "Dominant".

        Returns:
            _type_: filtered sampels
        """
        acceptables = ["Dominant", "Reference"]
        if citation_type  not in acceptables:
            raise ValueError(f"Invalid citation type passed. accepted are {acceptables}")
        cite_type = f"[{citation_type}]"
        samples = [sample for sample in self.samples if cite_type in sample["source"]]
        self.samples = samples

    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset

class CitationTextGenerationDatasetNoCitationType(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False,
                 len_control_flag=False,
                 no_context=False
                 ):

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)

            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                        # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        pass
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        citation_marks = []
                        citation_type = None
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)
                        context_before = paragraph[:span["char_start"]].replace(
                            "[BOS] ", "")
                        context_after = paragraph[span["char_end"]:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 span["char_start"]:span["char_end"]].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        if len(span["span_citation_mapping"]["Dominant"]) > 0:
                            source.append("[Mask]")
                            citation_type = "Dominant"
                        else:
                            source.append("[Mask]")
                            citation_type = "Reference"
                        source.append(context_after)

                        if no_context:
                            source = []
                        citation_marks_flag = False
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Dominant"].items():
                            source.append("[B_Mask]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link],
                                                       include_result=include_conclusion))
                            source.append("[E_Mask]")
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Reference"].items():
                            source.append("[B_Mask]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link]))
                            source.append("[E_Mask]")
                        if len_control_flag:
                            len_control_token = "[SHORT]" if len(target) < 150 else "[LARGE]"
                            source.append(len_control_token)
                        source = " ".join(source)
                        if skip_no_citations and not citation_marks_flag:
                            continue

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(i_span),
                                "source": source,
                                "target": target,
                                "citation_type": citation_type,
                                # "full_target": " ".join(sentences)
                                #"citations": "#".join(citation_marks)
                            })
            except:
                #print("Skip "+paper_id)
                pass

    def filter_citation_type(self, citation_type="Dominant"):
        """ Inplace filter samples based on  citatio_type
        Args:
            citation_type (str, optional): one of ["Dominant", "Reference]. Defaults to "Dominant".

        Returns:
            _type_: filtered sampels
        """
        acceptables = ["Dominant", "Reference"]
        if citation_type  not in acceptables:
            raise ValueError(f"Invalid citation type passed. accepted are {acceptables}")
        samples = [sample for sample in self.samples if sample["citation_type"] == citation_type]
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CitationTextGenerationDatasetOld(Dataset):
    def __init__(self, path_name: str, tokenizer, t5_tokenizer,
                 augment_tags=False, train=True, context_window=2,
                 MAX_SENT_LEN=9999,
                 related_work_path='/home/data/20200705v1/acl/selected_related_work.jsonl',
                 cited_metadata_path='/home/data/20200705v1/acl/selected_cited_metadata.jsonl'):
        self.max_sent_len = MAX_SENT_LEN
        self.context_window = context_window
        self.t5_tokenizer = t5_tokenizer
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)

        self.dataset = {}
        for text_file in text_files:
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len, 0)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))

            annotation_file = text_file.replace(".txt", ".ann")
            all_annotations = read_annotations(annotation_file, offsets)
            for paragraph_id, paragraph, paragraph_annotation in zip(
                    paragraph_ids, paragraphs, all_annotations):
                for annotation in paragraph_annotation:
                    assert paragraph[annotation[0]:annotation[1]] == annotation[
                        -1]
                sentences = [
                    tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent))
                    for sent in paragraph.split("[BOS] ")[1:]]
                tokens = tokenizer.tokenize(paragraph)
                N_tokens = len(tokens)
                discourse_labels = read_discourse_labels(paragraph_annotation,
                                                         paragraph,
                                                         self.discourse_label_types)
                # validate_span_annotation(paragraph_annotation)
                span_indices = read_span_indices(paragraph_annotation,
                                                 paragraph)
                span_BIO_labels = get_span_BIO_labels(span_indices, paragraph,
                                                      tokenizer)[1:-1]
                citation_mark_span_indices = read_citation_mark(
                    paragraph_annotation, paragraph)
                citation_BIO_labels = get_citation_BIO_labels(
                    citation_mark_span_indices, paragraph, tokenizer)[1:-1]

                # print(tokenizer.tokenize(paragraph))
                assert (N_tokens == len(span_BIO_labels) == len(
                    citation_BIO_labels))
                # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                #    continue

                augmented_paragraph, augmented_sentences = make_augmented_paragraphs(
                    tokens, tokenizer, discourse_tokens, discourse_labels,
                    span_BIO_labels, citation_BIO_labels)
                paragraph_citation_links = sentence_citation_link(paragraph_id,
                                                                  augmented_sentences,
                                                                  self.related_work_jsons,
                                                                  tokenizer)
                span_sent_mapping, i_span = span_sentence_map(
                    augmented_sentences)
                paragraph_citation_links = propagate_citation_cross_sentences(
                    span_sent_mapping, paragraph_citation_links, i_span)
                self.dataset[paragraph_id] = {
                    "paragraph_id": paragraph_id,
                    "related_work": augmented_paragraph,
                    "citation_links": paragraph_citation_links,
                    "augmented_sentences": augmented_sentences,
                    "discourse_labels": discourse_labels,
                    "sentences": sentences
                }

        self.samples = []
        for paragraph_id, paragraph in self.dataset.items():
            for si, (sent, links, discourse) in enumerate(
                    zip(paragraph["sentences"], paragraph["citation_links"],
                        paragraph["discourse_labels"])):
                if discourse == "Single_summ":
                    if augment_tags:
                        context_before = mount_discourse_sentence(
                            paragraph["discourse_labels"][
                            max(0, si - self.context_window):si],
                            paragraph["augmented_sentences"][
                            max(0, si - self.context_window):si])
                        context_after = mount_discourse_sentence(
                            paragraph["discourse_labels"][
                            si + 1: si + self.context_window + 1],
                            paragraph["augmented_sentences"][
                            si + 1: si + self.context_window + 1])
                    else:
                        context_before = " ".join(paragraph["sentences"][max(0,
                                                                             si - self.context_window):si])
                        context_after = " ".join(paragraph["sentences"][
                                                 si + 1: si + self.context_window + 1])

                    context = context_before + " [" + discourse + "]" + " [Answer] " + context_after

                    if augment_tags:
                        dominant_context = ""
                        for citation_mark, link in links["Dominant"].items():
                            if str(link) in self.cited_metadata:
                                cited_context = self.cited_metadata[str(link)][
                                    "abstract"]
                                if cited_context is None:
                                    cited_context = ""
                            else:
                                cited_context = ""
                            dominant_context += "[B_Dominant_context] " + citation_mark + " [SEP] " + cited_context + " [E_Dominant_context] "
                        reference_context = ""
                        for citation_mark, link in links["Reference"].items():
                            if str(link) in self.cited_metadata:
                                cited_context = self.cited_metadata[str(link)][
                                    "title"]
                                if cited_context is None:
                                    cited_context = ""
                            else:
                                cited_context = ""
                            reference_context += "[B_Reference_context] " + citation_mark + " [SEP] " + cited_context + " [E_Reference_context] "

                        source = "[B_Context] " + context + " [E_Context] " + dominant_context + " " + reference_context
                    else:
                        citation_context = ""
                        merged_links = {**links["Dominant"],
                                        **links["Reference"]}
                        for citation_mark, link in merged_links.items():
                            citation_context += "[B_Citation] " + citation_mark + " [SEP] [E_Citation] "
                        source = "[B_Context] " + context + " [E_Context] " + citation_context

                    self.samples.append(
                        {
                            "sample_id": paragraph_id + "_" + str(si),
                            "source": "Generate: " + source,
                            "target": sent.strip()
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = self.samples[idx]["source"]

        text = self.samples[idx]["target"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length=1000,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length=500,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class JointTaggerAgreementDataset(Dataset):
    def __init__(self, text_files: str, tokenizer, MAX_SENT_LEN=512):

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        self.all_discourse_labels, self.all_spans_BIO_labels, self.all_citations_BIO_labels = [], [], []


        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)

                        # NEEDED
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)

                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        citation_BIO_labels = [x[2:] if x.startswith("B_") or x.startswith("I_") else x for x in citation_BIO_labels ]
                        span_BIO_labels = [x[2:] if x.startswith("B_") or x.startswith("I_") else x for x in span_BIO_labels ]
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        self.all_discourse_labels.extend(discourse_labels)
                        self.all_citations_BIO_labels.extend(citation_BIO_labels)
                        self.all_spans_BIO_labels.extend(span_BIO_labels)
                    except:
                        print("Skip " + paragraph_id)
            except:
                print("Skip " + paper_id)


class DiscourseT5Dataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name: str, tokenizer, t5_tokenizer, train=True,
                 MAX_SENT_LEN=9999, sentence_overlap=0):
        self.max_sent_len = MAX_SENT_LEN
        self.t5_tokenizer = t5_tokenizer
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }
        self.discourse_label_lookup = {v: k for k, v in
                                       self.discourse_label_types.items()}

        self.span_label_types = {"O": 0, "B_span": 1, "I_span": 2}
        self.span_label_lookup = {v: k for k, v in
                                  self.span_label_types.items()}

        self.citation_label_types = {"O": 0, "B_Dominant": 1, "I_Dominant": 2,
                                     "B_Reference": 3, "I_Reference": 4}
        self.citation_label_lookup = {v: k for k, v in
                                      self.citation_label_types.items()}

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len, self.sentence_overlap)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))
            if train:
                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)
                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[-1]
                    discourse_labels = read_discourse_labels(
                        paragraph_annotation, paragraph,
                        self.discourse_label_types)
                    # label_string = " ".join([str(self.discourse_label_types[tag]) for tag in discourse_labels])
                    label_string = " ".join(["<extra_id_" + str(i) + "> " + str(
                        self.discourse_label_types[tag]) for i, tag in
                                             enumerate(
                                                 discourse_labels)]) + " <extra_id_" + str(
                        len(discourse_labels)) + ">"
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': "Discourse: " + paragraph,
                        'label': label_string
                    })

            else:
                for paragraph_id, paragraph in zip(paragraph_ids, paragraphs):
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': "Discourse: " + paragraph,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = str(self.samples[idx]["paragraph"])
        text = self.samples[idx]["label"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length=512,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length=512,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class DiscourseSeqT5Dataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name: str, tokenizer, t5_tokenizer, train=True,
                 MAX_SENT_LEN=9999, sentence_overlap=0):
        self.max_sent_len = MAX_SENT_LEN
        self.t5_tokenizer = t5_tokenizer
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6,
                                      "O": 7
                                      }
        self.discourse_label_lookup = {v: k for k, v in
                                       self.discourse_label_types.items()}

        self.span_label_types = {"O": 0, "B_span": 1, "I_span": 2}
        self.span_label_lookup = {v: k for k, v in
                                  self.span_label_types.items()}

        self.citation_label_types = {"O": 0, "B_Dominant": 1, "I_Dominant": 2,
                                     "B_Reference": 3, "I_Reference": 4}
        self.citation_label_lookup = {v: k for k, v in
                                      self.citation_label_types.items()}

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len, self.sentence_overlap)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))
            if train:
                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)
                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[-1]
                    discourse_labels = read_discourse_labels(
                        paragraph_annotation, paragraph,
                        self.discourse_label_types)
                    paragraph = "Discourse: " + paragraph
                    discourse_seq_labels = discourse_tag2seq(paragraph,
                                                             discourse_labels,
                                                             self.t5_tokenizer)
                    label_string = " ".join(
                        [str(self.discourse_label_types[tag]) for tag in
                         discourse_seq_labels])
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': paragraph,  #######
                        'label': label_string
                    })

            else:
                for paragraph_id, paragraph in zip(paragraph_ids, paragraphs):
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': "Discourse: " + paragraph,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = str(self.samples[idx]["paragraph"])
        text = self.samples[idx]["label"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length=512,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length=512,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class CitationSpanT5Dataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name: str, tokenizer, t5_tokenizer, train=True,
                 MAX_SENT_LEN=9999, sentence_overlap=0):
        self.max_sent_len = MAX_SENT_LEN
        self.t5_tokenizer = t5_tokenizer
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap
        self.span_label_types = {"O": 0, "B_span": 1, "I_span": 2}
        self.span_label_lookup = {v: k for k, v in
                                  self.span_label_types.items()}

        self.citation_label_types = {"O": 0, "B_Dominant": 1, "I_Dominant": 2,
                                     "B_Reference": 3, "I_Reference": 4}
        self.citation_label_lookup = {v: k for k, v in
                                      self.citation_label_types.items()}

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len, self.sentence_overlap)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))
            if train:
                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)
                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[-1]

                    tokens = tokenizer.tokenize(paragraph)
                    # validate_span_annotation(paragraph_annotation)
                    span_indices = read_span_indices(paragraph_annotation,
                                                     paragraph)
                    span_BIO_labels = get_span_BIO_labels(span_indices,
                                                          paragraph, tokenizer)[
                                      1:-1]

                    augmented_paragraph, _ = make_augmented_paragraphs(tokens,
                                                                       tokenizer,
                                                                       span_BIO_labels=span_BIO_labels)
                    target = find_span(augmented_paragraph, tokenizer,
                                       "[B_span]", "[E_span]", truncate=True)
                    if len(target.strip()) == 0:
                        target = "<extra_id_0>"
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': "Citation span: " + paragraph,
                        'label': custom_span_token2T5(target)
                    })

            else:
                for paragraph_id, paragraph in zip(paragraph_ids, paragraphs):
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': "Citation span: " + paragraph
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = self.samples[idx]["paragraph"]
        text = self.samples[idx]["label"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length=1000,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length=500,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class CitationTypeT5Dataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name: str, tokenizer, t5_tokenizer, train=True,
                 MAX_SENT_LEN=9999, sentence_overlap=0):
        self.max_sent_len = MAX_SENT_LEN
        self.t5_tokenizer = t5_tokenizer
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap

        self.citation_label_types = {"O": 0, "B_Dominant": 1, "I_Dominant": 2,
                                     "B_Reference": 3, "I_Reference": 4}
        self.citation_label_lookup = {v: k for k, v in
                                      self.citation_label_types.items()}

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len, self.sentence_overlap)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))
            if train:
                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)
                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[-1]

                    tokens = tokenizer.tokenize(paragraph)
                    # validate_span_annotation(paragraph_annotation)
                    citation_mark_span_indices = read_citation_mark(
                        paragraph_annotation, paragraph)
                    citation_BIO_labels = get_citation_BIO_labels(
                        citation_mark_span_indices, paragraph, tokenizer)[1:-1]

                    augmented_paragraph, _ = make_augmented_paragraphs(tokens,
                                                                       tokenizer,
                                                                       citation_BIO_labels=citation_BIO_labels)
                    dominant_target = find_span(augmented_paragraph, tokenizer,
                                                "[B_Dominant]", "[E_Dominant]")
                    reference_target = find_span(augmented_paragraph, tokenizer,
                                                 "[B_Reference]",
                                                 "[E_Reference]")
                    # target = (dominant_target + "@" + reference_target).strip()
                    target = custom_citation_token2T5(dominant_target,
                                                      reference_target)
                    if len(target.strip()) == 0:
                        target = "<extra_id_0>"
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': "Citation type: " + paragraph,
                        'label': target
                    })

            else:
                for paragraph_id, paragraph in zip(paragraph_ids, paragraphs):
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': "Citation type: " + paragraph
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = self.samples[idx]["paragraph"]
        text = self.samples[idx]["label"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length=1000,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length=500,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class JointRelatedWorkTaggingT5Dataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, datasets, tokenizer):
        self.samples = []
        self.t5_tokenizer = tokenizer
        for dataset in datasets:
            self.samples.extend(dataset.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = self.samples[idx]["paragraph"]
        text = self.samples[idx]["label"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length=1000,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length=500,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class RelatedWorkMLMDatasetOld(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name: str, tokenizer, t5_tokenizer, train=True,
                 MAX_SENT_LEN=9999, sentence_overlap=0):
        self.max_sent_len = MAX_SENT_LEN
        self.t5_tokenizer = t5_tokenizer
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap

        self.samples = []

        with open(path_name, "r") as f_pdf:
            for line in tqdm(f_pdf):
                related_work_dict = json.loads(line)
                year = related_work_dict["year"]
                if year is None:
                    year = 0
                if (train and year <= 2017) or (not train and year == 2018):
                    # test set should include papers publised in 2019 and later
                    for pi, para in enumerate(
                            related_work_dict["related_work"]):
                        source, target = makeMaskedLanguageModelSample(
                            para["text"], tokenizer)
                        self.samples.append({
                            'id': related_work_dict["paper_id"] + "_" + str(pi),
                            'source': source,
                            'target': target
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = self.samples[idx]["source"]
        text = self.samples[idx]["target"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length=1000,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length=500,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        # source = self.t5_tokenizer.batch_encode_plus([ctext], max_length= 1000, return_tensors='pt')
        # target = self.t5_tokenizer.batch_encode_plus([text], max_length= 500, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class LEDRelatedWorkMLMDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name, tokenizer, train=True, MAX_SENT_LEN=16000,
                 sentence_overlap=0):
        self.max_sent_len = MAX_SENT_LEN
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap

        self.samples = []

        with open(path_name, "r") as f_pdf:
            for line in tqdm(f_pdf):
                related_work_dict = json.loads(line)
                year = related_work_dict["year"]
                if year is None:
                    year = 0
                if (train and year <= 2017) or (not train and year == 2018):
                    # test set should include papers publised in 2019 and later
                    for pi, para in enumerate(
                            related_work_dict["related_work"]):
                        source, target = makeMLMsample(para["text"],
                                                       tokenizer.mask_token)
                        self.samples.append({
                            'id': related_work_dict["paper_id"] + "_" + str(pi),
                            'source': source,
                            'target': target
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BERTRelatedWorkMLMDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name, tokenizer, train=True, MAX_SENT_LEN=512,
                 sentence_overlap=0):
        self.max_sent_len = MAX_SENT_LEN
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap

        self.samples = []

        with open(path_name, "r") as f_pdf:
            for line in tqdm(f_pdf):
                related_work_dict = json.loads(line)
                year = related_work_dict["year"]
                if year is None:
                    year = 0
                if (train and year <= 2017) or (not train and year == 2018):
                    # test set should include papers publised in 2019 and later
                    for pi, para in enumerate(
                            related_work_dict["related_work"]):
                        source, target = makeBertMLMsample(para["text"],
                                                           tokenizer)
                        if len(tokenizer.tokenize(source)) != len(
                                tokenizer.tokenize(target)):
                            continue
                        self.samples.append({
                            'id': related_work_dict["paper_id"] + "_" + str(pi),
                            'source': source,
                            'target': target
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RelatedWorkSentenceReorderingDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name: str, tokenizer, t5_tokenizer, train=True,
                 MAX_SENT_LEN=9999, sentence_overlap=0):
        self.max_sent_len = MAX_SENT_LEN
        self.t5_tokenizer = t5_tokenizer
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap

        self.samples = []

        with open(path_name, "r") as f_pdf:
            for line in tqdm(f_pdf):
                related_work_dict = json.loads(line)
                year = related_work_dict["year"]
                if year is None:
                    year = 0
                if (train and year <= 2017) or (not train and year == 2018):
                    # test set should include papers publised in 2019 and later
                    for pi, para in enumerate(
                            related_work_dict["related_work"]):
                        source, target = makeSentenceReorderingSample(
                            para["text"])
                        self.samples.append({
                            'id': related_work_dict["paper_id"] + "_" + str(pi),
                            'source': source,
                            'target': target
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = self.samples[idx]["source"]
        text = self.samples[idx]["target"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length=1000,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length=500,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class EmbeddingCitationTextGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, t5_tokenizer,
                 augment_tags=False, train=True,
                 context_window=2, MAX_SENT_LEN=9999, importance_rank=2,
                 related_work_path='/home/data/20200705v1/acl/selected_related_work.jsonl',
                 cited_metadata_path='/home/data/20200705v1/acl/selected_cited_metadata.jsonl',
                 cached_embedding_path='/home/data/20200705v1/acl/embeddings'
                 ):
        self.max_sent_len = MAX_SENT_LEN
        self.context_window = context_window
        self.t5_tokenizer = t5_tokenizer
        self.importance_rank = importance_rank
        self.cached_embedding_path = cached_embedding_path
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)

        self.dataset = {}
        for text_file in text_files:
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len, 0)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))

            annotation_file = text_file.replace(".txt", ".ann")
            all_annotations = read_annotations(annotation_file, offsets)
            for paragraph_id, paragraph, paragraph_annotation in zip(
                    paragraph_ids, paragraphs, all_annotations):
                for annotation in paragraph_annotation:
                    assert paragraph[annotation[0]:annotation[1]] == annotation[
                        -1]
                sentences = [
                    tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent))
                    for sent in paragraph.split("[BOS] ")[1:]]
                tokens = tokenizer.tokenize(paragraph)
                N_tokens = len(tokens)
                discourse_labels = read_discourse_labels(paragraph_annotation,
                                                         paragraph,
                                                         self.discourse_label_types)
                # validate_span_annotation(paragraph_annotation)
                span_indices = read_span_indices(paragraph_annotation,
                                                 paragraph)
                span_BIO_labels = get_span_BIO_labels(span_indices, paragraph,
                                                      tokenizer)[1:-1]
                citation_mark_span_indices = read_citation_mark(
                    paragraph_annotation, paragraph)
                citation_BIO_labels = get_citation_BIO_labels(
                    citation_mark_span_indices, paragraph, tokenizer)[1:-1]

                # print(tokenizer.tokenize(paragraph))
                assert (N_tokens == len(span_BIO_labels) == len(
                    citation_BIO_labels))
                # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                #    continue

                augmented_paragraph, augmented_sentences = make_augmented_paragraphs(
                    tokens, tokenizer, discourse_tokens, discourse_labels,
                    span_BIO_labels, citation_BIO_labels)
                paragraph_citation_links = sentence_citation_link(paragraph_id,
                                                                  augmented_sentences,
                                                                  self.related_work_jsons,
                                                                  tokenizer)
                span_sent_mapping, i_span = span_sentence_map(
                    augmented_sentences)
                paragraph_citation_links = propagate_citation_cross_sentences(
                    span_sent_mapping, paragraph_citation_links, i_span)
                self.dataset[paragraph_id] = {
                    "paragraph_id": paragraph_id,
                    "related_work": augmented_paragraph,
                    "citation_links": paragraph_citation_links,
                    "augmented_sentences": augmented_sentences,
                    "discourse_labels": discourse_labels,
                    "sentences": sentences
                }

        self.samples = []
        for paragraph_id, paragraph in self.dataset.items():
            for si, (sent, links, discourse) in enumerate(
                    zip(paragraph["sentences"], paragraph["citation_links"],
                        paragraph["discourse_labels"])):
                if discourse == "Single_summ":
                    if augment_tags:
                        context_before = mount_discourse_sentence(
                            paragraph["discourse_labels"][
                            max(0, si - self.context_window):si],
                            paragraph["augmented_sentences"][
                            max(0, si - self.context_window):si])
                        context_after = mount_discourse_sentence(
                            paragraph["discourse_labels"][
                            si + 1: si + self.context_window + 1],
                            paragraph["augmented_sentences"][
                            si + 1: si + self.context_window + 1])
                    else:
                        context_before = " ".join(paragraph["sentences"][max(0,
                                                                             si - self.context_window):si])
                        context_after = " ".join(paragraph["sentences"][
                                                 si + 1: si + self.context_window + 1])

                    context = context_before + " [" + discourse + "]" + " [Answer] " + context_after

                    if augment_tags:
                        dominant_context = ""
                        citation_links = []
                        for citation_mark, link in links["Dominant"].items():
                            if str(link) in self.cited_metadata:
                                cited_context = self.cited_metadata[str(link)][
                                    "abstract"]
                                if cited_context is None:
                                    cited_context = ""
                            else:
                                cited_context = ""
                            citation_links.append(str(link))
                            dominant_context += "[B_Dominant_context] " + citation_mark + " [SEP] [Embedding] " + cited_context + " [E_Dominant_context] "
                        reference_context = ""
                        for citation_mark, link in links["Reference"].items():
                            if str(link) in self.cited_metadata:
                                cited_context = self.cited_metadata[str(link)][
                                    "title"]
                                if cited_context is None:
                                    cited_context = ""
                            else:
                                cited_context = ""
                            citation_links.append(str(link))
                            reference_context += "[B_Reference_context] " + citation_mark + " [SEP] [Embedding]" + cited_context + " [E_Reference_context] "

                        source = "[B_Context] " + context + " [E_Context] " + dominant_context + " " + reference_context
                    else:
                        citation_context = ""
                        citation_links = []
                        merged_links = {**links["Dominant"],
                                        **links["Reference"]}
                        for citation_mark, link in merged_links.items():
                            citation_context += "[B_Citation] " + citation_mark + " [SEP] [Embedding] [E_Citation] "
                            citation_links.append(str(link))
                        source = "[B_Context] " + context + " [E_Context] " + citation_context

                    self.samples.append(
                        {
                            "sample_id": paragraph_id + "_" + str(si),
                            "source": "Generate: " + source,
                            "target": sent.strip(),
                            "citation_links": " ".join(citation_links)
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = self.samples[idx]["source"]

        text = self.samples[idx]["target"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length=512,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length=512,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        citation_links = self.samples[idx]["citation_links"].split()
        paper_embedings = []
        for link in citation_links:
            embedding_path = os.path.join(cached_embedding_path,
                                          link + ".embedding")
            if os.path.exists(path):
                embedding = torch.load(embedding_path)
                importance = torch.load(
                    embedding_path.replace(".embedding", ".importance"))
                selected_embedding = embedding[
                    importance <= self.importance_rank]
                paper_embedings.append(selected_embedding)
            else:
                paper_embedings.append(None)
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'paper_embeddings': paper_embedings
        }


class LEDRelatedWorkMLMDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name, train, bos_token="<s>", eos_token="</s>"):
        self.samples = []

        with open(path_name, "r") as f_pdf:
            for line in tqdm(f_pdf):
                related_work_dict = json.loads(line)
                year = related_work_dict["year"]
                if year is None:
                    year = 0
                if (train and year <= 2017) or (not train and year == 2018):
                    # test set should include papers publised in 2019 and later
                    for pi, para in enumerate(
                            related_work_dict["related_work"]):
                        source, target = makeMLMsample(para["text"], "<mask>")
                        self.samples.append({
                            'id': related_work_dict["paper_id"] + "_" + str(pi),
                            'source': " ".join([bos_token, source, eos_token]),
                            'target': target
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CrossDocumentLMdataset(Dataset):
    def __init__(self, path_name, tokenizer, train=True, context_window=2,
                 MAX_SENT_LEN=16000, bos_token="<s>", eos_token="</s>",
                 mask_token="<mask>",
                 bod_token="<doc>", eod_token="</doc>",
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
                 cited_paper_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl'):
        self.max_sent_len = MAX_SENT_LEN
        self.context_window = context_window
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_paper_jsons = read_related_work_jsons(cited_paper_path)

        self.dataset = {}
        for text_file in text_files:
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))

            annotation_file = text_file.replace(".txt", ".ann")
            all_annotations = read_annotations(annotation_file, offsets)
            for paragraph_id, paragraph, paragraph_annotation in zip(
                    paragraph_ids, paragraphs, all_annotations):
                for annotation in paragraph_annotation:
                    assert paragraph[annotation[0]:annotation[1]] == annotation[
                        -1]
                tokens = tokenizer.tokenize(paragraph, add_special_tokens=True)
                sentences = [sent for sent in paragraph.split("[BOS] ")[1:]]
                offset_mapping = \
                    tokenizer(paragraph, return_offsets_mapping=True)[
                        "offset_mapping"]
                N_tokens = len(offset_mapping)
                discourse_labels = read_discourse_labels(paragraph_annotation,
                                                         paragraph,
                                                         self.discourse_label_types)
                span_indices = read_span_indices(paragraph_annotation,
                                                 paragraph)
                span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                         offset_mapping)
                citation_mark_span_indices = read_citation_mark(
                    paragraph_annotation, paragraph)
                citation_BIO_labels = get_aligned_BIO_labels(
                    citation_mark_span_indices, offset_mapping)
                # print(tokenizer.tokenize(paragraph))
                assert (N_tokens == len(span_BIO_labels) == len(
                    citation_BIO_labels))
                # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                #    continue

                # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                # paragraph_citation_links = sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                    paragraph_id, sentences, self.related_work_jsons,
                    offset_mapping, citation_BIO_labels, separator="[BOS] ")
                paragraph_citation_links_pre = new_sentence_citation_link(
                    pargraph_citation_info, len(sentences))
                span_citation_mapping = map_span_citation(span_BIO_labels,
                                                          citation_BIO_labels,
                                                          pargraph_citation_info,
                                                          offset_mapping)
                span_sent_mapping, i_span = new_span_sentence_map(tokens,
                                                                  span_BIO_labels,
                                                                  bos="[BOS]")
                paragraph_citation_links = propagate_citation_cross_sentences(
                    span_sent_mapping, paragraph_citation_links_pre, i_span)
                self.dataset[paragraph_id] = {
                    "paragraph_id": paragraph_id,
                    "citation_links_by_sentence": paragraph_citation_links,
                    "discourse_labels": discourse_labels,
                    "sentences": sentences,
                    "citation_info": pargraph_citation_info,
                    "span_sent_mapping": span_sent_mapping
                }

        self.samples = []
        for paragraph_id, paragraph in self.dataset.items():
            for si, (sent, links, discourse) in enumerate(
                    zip(paragraph["sentences"],
                        paragraph["citation_links_by_sentence"],
                        paragraph["discourse_labels"])):
                context_before = " ".join(
                    paragraph["sentences"][max(0, si - self.context_window):si])
                context_after = " ".join(paragraph["sentences"][
                                         si + 1: si + self.context_window + 1])
                noisy_sent, sent = makeMLMsample(sent)
                target = " ".join([bos_token, sent, eos_token])
                context = " ".join(
                    [bod_token, context_before, bos_token, noisy_sent,
                     eos_token, context_after, eod_token])

                citation_context = ""
                merged_links = {**links["Dominant"], **links["Reference"]}
                for citation_mark, link in merged_links.items():
                    if link in self.cited_paper_jsons:
                        pdf_dict = self.cited_paper_jsons[link]
                        cited_paper = []
                        cited_paper.append(bod_token)
                        for p in pdf_dict["abstract"]:
                            # cited_paper.append(bos_token)
                            cited_paper.append("<Abstract> " + p["text"])
                            # cited_paper.append(eos_token)
                        for p in pdf_dict['body_text']:
                            # cited_paper.append(bos_token)
                            cited_paper.append(
                                "<" + p["section"] + "> " + p["text"])
                            # cited_paper.append(eos_token)
                        cited_paper.append(bod_token)
                        cited_paper = " ".join(cited_paper)

                        source = context + " " + cited_paper

                        self.samples.append(
                            {
                                "id": paragraph_id + "_" + str(si) + "_" + link,
                                "source": source,
                                "target": target
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class SimpleCrossDocumentLMdataset(Dataset):
    def __init__(self, path_name, tokenizer, train=True,
                 MAX_SENT_LEN=16000, bos_token="<s>", eos_token="</s>",
                 mask_token="<mask>",
                 bod_token="<doc>", eod_token="</doc>",
                 # related_work_path = '/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl'):
        self.max_sent_len = MAX_SENT_LEN
        self.related_work_jsons = read_related_work_jsons(path_name)
        self.cited_metadata_jsons = read_related_work_jsons(cited_metadata_path)

        self.samples = []

        for i, (ID, related_work) in tqdm(
                enumerate(self.related_work_jsons.items())):
            year = related_work["year"]
            if year is None:
                year = 0
            if (train and year <= 2017) or (not train and year == 2018):
                bib_entries = related_work["bib_entries"]
                for paragraph in related_work["related_work"]:
                    inputs = []
                    noisy_text, target = makeMLMsample(paragraph["text"])
                    inputs.extend([bod_token, noisy_text, eod_token])
                    if len(tokenizer(target)["input_ids"]) > self.max_sent_len:
                        continue
                    source = " ".join(inputs)

                    for citation in paragraph["cite_spans"]:
                        if citation["ref_id"] in bib_entries:
                            reference_link = bib_entries[citation["ref_id"]][
                                "link"]
                            if reference_link in self.cited_metadata_jsons:
                                cited_metadata = self.cited_metadata_jsons[
                                    reference_link]
                                title = cited_metadata["title"]
                                if title is None:
                                    title = ""
                                abstract = cited_metadata["abstract"]
                                if abstract is None:
                                    abstract = ""
                                inputs.extend(
                                    [bod_token, title, tokenizer.sep_token,
                                     abstract, eod_token])
                                if len(tokenizer(" ".join(inputs))[
                                           "input_ids"]) > self.max_sent_len:
                                    break
                                source = " ".join(inputs)
                    self.samples.append({
                        "id": ID + "_" + str(i),
                        "source": source,
                        "target": target
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset
