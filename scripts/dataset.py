import os
from glob import glob

from torch.utils.data import Dataset
from tqdm import tqdm
from util import *


class CitationTextGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/20200705v1/acl/pdf_parses.jsonl",
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
                 related_work_path='/home/data/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/20200705v1/acl/pdf_parses.jsonl",
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
