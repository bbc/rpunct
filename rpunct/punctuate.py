# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import re
import os
import json
import warnings
from time import time
from tqdm import tqdm
from simpletransformers.ner import NERModel

from langdetect import detect

# VALID_LABELS = ["OU", "OO", ".O", "!O", ",O", ".U", "!U", ",U", ":O", ";O", ":U", "'O", "-O", "?O", "?U"]
# PUNCT_LABELS = ['O', '.', ',', ':', ';', "'", '-', '?', '!', '%']
PUNCT_LABELS = ['O', '.', ',', ':', ';', "'", '-', '?', '!']
CAPI_LABELS = ['O', 'C', 'U', 'M']
VALID_LABELS = [f"{x}{y}" for y in CAPI_LABELS for x in PUNCT_LABELS]
TERMINALS = ['.', '!', '?']


class RestorePuncts:
    """
    An RPunct punctuation restoration model object.
    Initiate an instance from a pre-trained transformer model and use to infer punctuated text from plaintext.

    Args:
        - model_source (Path): the path (from `rpunct/`) to the directory containing the pre-trained transformer model files to construct an RPunct model around.
        - use_cuda (bool): run inference on GPU (True) or CPU (False).
    """
    def __init__(self, model_source, use_cuda=True):
        self.mc_database = self.load_mixed_case_database()  # Lookup database to restore the capitalisation of mixed-case words (e.g. "iPlayer")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = NERModel(
                "bert",
                model_source,
                labels=VALID_LABELS,
                use_cuda=use_cuda,
                args={
                    "silent": True,
                    "max_seq_length": 512
                }
            )

        self._memory_buffer = ""

    def punctuate(self, input_text:str, strict_sentence_boundaries:bool=True):
        """
        Performs punctuation restoration on plaintext (in English only).

        Args:
            - text (str): text to punctuate, can be few words to as large as you want.

        Returns:
            - punct_text (str): fully punctuated output text.
        """
        # Restoration pipeline
        model_segments = self.segment_text_blocks(input_text)  # Format input text such that it can be easily passed to the transformer model
        preds_lst = self.predict(model_segments)  # Generate word-level punctuation predictions

        if preds_lst is None:
            print(" * Failed to produce punctuation predictions on a given input text")
            return input_text

        combined_preds = self.combine_results(preds_lst)  # Combine a list of text segments and their predictions into a single sequence
        punct_text = self.punctuate_texts(combined_preds, strict_sentence_boundaries)  # Apply the punctuation predictions to the text

        return punct_text

    def punctuate_segments(self, input_segments:list, strict_sentence_boundaries:bool=True):
        # Define segment boundaries s.t. they can be restored after being fed through the model
        segment_lengths = [len(s.split()) for s in input_segments]
        segment_boundaries = [(0, 0)]

        for length in segment_lengths:
            start_idx = segment_boundaries[-1][1]
            end_idx = start_idx + length
            segment_boundaries.append((start_idx, end_idx))

        segment_boundaries = segment_boundaries[1:]

        # Make predictions over entire concatenated text
        text_block = " ".join(input_segments)
        model_segments = self.segment_text_blocks(text_block)

        predictions = self.predict(model_segments)

        if predictions is None:
            print(" * Failed to produce punctuation predictions on the given input text")
            return input_segments

        combined_predictions = self.combine_results(predictions)

        # Break predictions back down into segments and apply pnctuation predictions
        segmented_predictions = [combined_predictions[start: end] for start, end in segment_boundaries]

        punct_text = []
        punct_text = [self.punctuate_texts(pred, strict_sentence_boundaries) for pred in segmented_predictions]

        # Always enforce strict sentence boundaries on first and last segment
        punct_text[0] = punct_text[0][0].capitalize() + punct_text[0][1:]

        if punct_text[-1][-1] not in TERMINALS:
            if punct_text[-1][-1].isalnum() or punct_text[-1][-1] in ['%', "'"]:
                punct_text[-1] += '.'
            else:
                punct_text[-1] = punct_text[-1][:-1] + '.'

        return punct_text

    def predict(self, input_segments):
        """
        Passes the unpunctuated text to the model for punctuation.
        """
        text_segments = [i['text'] for i in input_segments]

        predictions = self.model.predict(text_segments)
        predicted_segments = predictions[0]

        for orig, pred in zip(text_segments, predicted_segments):
            if len(orig.split()) != len(pred):
                return None

        return predicted_segments

    @staticmethod
    def segment_text_blocks(text:str, body_len:int=250, overlap_len:int=30):
        """
        Splits a string of text into predefined slices of overlapping text with indexes linked back to the original text.
        This is done to bypass 512 token limit on transformer models by sequentially feeding segments of <512 tokens.

        Args:
            - text (str): input string of text to be split.
            - body_len (int): number of words in the (non-overlapping portion of) the output text segment.
            - overlap_len (int): number of words to overlap between text segements.

        Returns:
            - resp (lst): list of dicts specifying each text segment (containing the text and its start/end indices).
            E.g. [{...}, {"text": "...", 'start_idx': 31354, 'end_idx': 32648}, {...}]
        """
        text = text.replace('\n', ' ')
        wrds = text.split()

        resp = []
        lst_chunk_idx = 0
        i = 0

        while True:
            # Words in the chunk and the overlapping portion
            body_start = body_len * i
            body_end = body_len * (i + 1)
            wrds_in_body = wrds[body_start : body_end]

            overlap_end = (body_len * (i + 1)) + overlap_len
            wrds_in_overlap = wrds[body_end : overlap_end]

            wrds_split = wrds_in_body + wrds_in_overlap

            # Break loop if no more words
            if not wrds_split:
                break

            wrds_str = " ".join(wrds_split)
            nxt_chunk_start_idx = len(" ".join(wrds_in_body))
            lst_char_idx = len(" ".join(wrds_split))

            # Text segment object
            resp_obj = {
                "text": wrds_str,
                "start_idx": lst_chunk_idx,
                "end_idx": lst_char_idx + lst_chunk_idx,
            }

            resp.append(resp_obj)
            lst_chunk_idx += nxt_chunk_start_idx + 1
            i += 1

        return resp

    @staticmethod
    def combine_results(predicted_text_blocks:list, body_len:int=250):
        """
        Given a full text and predictions of each slice combines the segmented predictions into a single text again
        (i.e. inverse function of `segment_text_blocks`).
        """
        output_text = []

        for block in predicted_text_blocks:
            pairs_in_body = block[:body_len]
            flattened_body = [list(wrd.items())[0] for wrd in pairs_in_body]
            output_text.extend(flattened_body)

        return output_text

    def punctuate_texts(self, full_pred:list, strict_sentence_boundaries:bool=True):
        """
        Given a list of predictions from the model, applies the predictions to the plaintext, restoring full punctuation and capitalisation.
        """
        valid_punctuation = [p for p in PUNCT_LABELS if p not in ["O", "'", "%"]]
        punct_resp = ""

        # Cycle through the list containing each word and its predicted label
        for i in full_pred:
            word, label = i

            # Implement capitalisation (lowercase/capitalised/uppercase/mixed-case)
            if re.sub(r'[^£$€]', '', word) or label[-1] == "O":  # `xO` => lowercase & don't capitalise if part of a currency
                punct_wrd = word

            elif label[-1] == "U":  # `xU` => uppercase
                punct_wrd = word.upper()

                if len(word) > 2 and word[-2:] == "'S":
                    punct_wrd = punct_wrd[:-2] + "'s"  # possessive

            elif label[-1] == "C":  # `xC` => capitalised
                punct_wrd = word.capitalize()

            elif label[-1] == "M":  # `xM` => mixed-case
                # Search the database for correct mixed-casing. If acronym is plural/possessive, set the trailing `s` as lowercase.
                if len(word) > 2 and word[-2:] == "'s":
                    punct_wrd = self.fetch_mixed_casing(word[:-2]) + "'s"  # possessive
                elif len(word) > 1 and word[-1:] == "s":
                    punct_wrd = self.fetch_mixed_casing(word[:-1]) + "s"  # plural
                else:
                    punct_wrd = self.fetch_mixed_casing(word)  # general mixed-case

            else:
                raise ValueError(f"Invalid capitalisation label: '{label[-1]}'")

            # Ensure terminals are followed by capitals
            if (len(punct_resp) > 1 and punct_resp[-2] in TERMINALS) or (punct_resp == "" and self._memory_buffer != "" and self._memory_buffer[-1] in TERMINALS):
                punct_wrd = punct_wrd.capitalize()

            # Add classified punctuation mark (and space) after word
            if label[0] != "O" and punct_wrd[-1] not in valid_punctuation and label[0] != punct_wrd[-1]:
                punct_wrd += label[0]

            punct_resp += punct_wrd + " "

        # Remove unnecessary whitespace and ensure the
        punct_resp = punct_resp.strip()
        punct_resp = punct_resp.replace("- ", "-")
        punct_resp = re.sub(r'[-]{1}([£$€¥]{1})', r' \1', punct_resp)

        # remove unwanted segmenting of numbers
        punct_resp = re.sub(r"([0-9]+)[\-:; ]([0-9]+)", r'\1\2', punct_resp)

        # Ensure the text starts with a capital and ends with a terminal
        if strict_sentence_boundaries:
            if len(punct_resp) > 1:
                punct_resp = punct_resp[0].capitalize() + punct_resp[1:]
            else:
                punct_resp = punct_resp.capitalize()

            if len(punct_resp) > 0:
                if punct_resp[-1].isalnum() or punct_resp[-1] in ['%', "'"]:
                    punct_resp += "."
                elif punct_resp[-1] not in TERMINALS:
                    punct_resp = punct_resp[:-1] + "."

        self._memory_buffer = punct_resp  # Save response for reference in next pass (to check if ends in a terminal and enforce capitalisation)

        return punct_resp

    @staticmethod
    def load_mixed_case_database(file='rpunct/mixed-casing.json'):
        """
        Loads the mixed-case database from a file into a python variable so it can be accessed during restoration.
        """
        # Load database of mixed-case instances
        try:
            with open(file) as f:
                database = json.load(f)
        except FileNotFoundError:
            database = None

        return database

    def fetch_mixed_casing(self, plaintext:str):
        """
        Retrieve the correct mixed-case capitalisation of a plaintext word.
        """
        # In case of no database, return as uppercase acronym
        if self.mc_database is None:
            return plaintext.upper()

        correct_capitalisation = self.mc_database.get(plaintext)  # fetch from database

        # For words not in the database, return as uppercase acronym
        if correct_capitalisation is None:
            correct_capitalisation = plaintext.upper()

        return correct_capitalisation
