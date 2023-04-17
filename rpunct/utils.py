# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import re
from kaldialign import align

class Item(object):
    """
    Class representing an item in a transcript.

    """

    def __init__(self, start_time, end_time, content, original_content=None, likelihood=1):
        """
        Constructor.

        Args:
            start_time: The start time of the item in seconds (string/float) e.g. "75.24"
            end_time: The end time of the item in seconds (string/float)
            content: The content of the item (string)
            original_content: Any content elements merged in a punctuation recovered Item (string) e.g. "fifty five" for the item with content "55"
            likelihood: The predicted probability of a word (only applicable in STT systems like Whisper)

        """

        self.start_time = round(float(start_time), 3)
        self.end_time = round(float(end_time), 3)
        self.content = content.strip()
        self.original_content = original_content
        self.likelihood = likelihood


def align_texts(ref_text:list, hyp_text:list, start_position:int=0, strip_punct:bool=True):
    """
    Function for aligning two lists of strings denoting words in a sentence/segment.
    Used for aligning recovered texts to the original text.

    Args:
        ref_text: The reference text we are aligning to (list of strings) e.g. an original plaintext
        hyp_text: The hypothesis text that we are aligning to the reference (list of strings) e.g. a recovered text
        start_position: Index within texts from which to start the alignment (int)

    Returns:
        mapping: A mapping of each word in hyp_text to its equivalent 1+ words in ref_text (2D list of strings: e.g. [... [['fifty', 'five'], ['55']], ...])
    """
    if strip_punct:
        hyp_text = [re.sub(r"[.,:;?!]", "", item.replace("-", " ")).lower().strip() for item in hyp_text]
        hyp_text = " ".join(hyp_text).strip().split()
        ref_text = [re.sub(r"[.,:;?!]", "", word.replace("-", " ")).lower().strip() for word in ref_text[start_position:]]
    else:
        hyp_text = [item.lower().strip() for item in hyp_text]
        ref_text = [item.lower().strip() for item in ref_text]

    hyp_text = hyp_text[start_position:]

    # print(f"\n - REF: {ref_text[:min(5, len(ref_text)-1)]}")
    # print(f" - HYP: {hyp_text[:min(5, len(hyp_text)-1)]}")

    EPS = '*'
    alignment = align(ref_text, hyp_text, EPS)
    mapping = []
    unmatched_refs = []

    for ref, hyp in alignment:
        if ref == EPS:  # insertion (one-to-many)
            mapping[-1][1].append(hyp)
        elif hyp == EPS:  # deletion (many-to-one)
            if len(mapping) > 0:
                mapping[-1][0].append(ref)  # append new word to multi-word element
            else:
                unmatched_refs.append(ref)  # protection against mismatches in the aligned text
        else:  # single substitution (one-to-one)
            mapping.append([[ref], [hyp]])

    mapping[0][0] = unmatched_refs + mapping[0][0]
    # print(f" - ALIGNMENT: {alignment[:min(5, len(alignment)-1)]}")
    # print(f" - MAPPING: {mapping[:min(5, len(mapping)-1)]}", end='\n\n')

    return mapping
