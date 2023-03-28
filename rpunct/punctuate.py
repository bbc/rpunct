# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import re
import os
import json
from tqdm import tqdm
from simpletransformers.ner import NERModel

# PUNCT_LABELS = ['O', '.', ',', ':', ';', "'", '-', '?', '!', '%']
PUNCT_LABELS = ['O', '.', ',', ':', ';', "'", '-', '?', '!']
CAPI_LABELS = ['O', 'C', 'U', 'M']
VALID_LABELS = [f"{x}{y}" for y in CAPI_LABELS for x in PUNCT_LABELS]
TERMINALS = ['.', '!', '?']
# VALID_LABELS = ["OU", "OO", ".O", "!O", ",O", ".U", "!U", ",U", ":O", ";O", ":U", "'O", "-O", "?O", "?U"]


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

    def punctuate(self, text:str):
        """
        Performs punctuation restoration on plaintext (in English only).

        Args:
            - text (str): text to punctuate, can be few words to as large as you want.

        Returns:
            - punct_text (str): fully punctuated output text.
        """
        # Restoration pipeline
        segments = self.segment_text_blocks(text)  # Format input text such that it can be easily passed to the transformer model
        preds_lst = self.predict(segments)  # Generate word-level punctuation predictions
        combined_preds = self.combine_results(preds_lst, text)  # Combine a list of text segments and their predictions into a single sequence
        punct_text = self.punctuate_texts(combined_preds)  # Apply the punctuation predictions to the text

        return punct_text

    def predict(self, input_segments:str, silent:bool=True):
        """
        Passes the unpunctuated text to the model for punctuation.
        """
        if silent:
            predictions = [self.model.predict([i['text']])[0][0] for i in input_segments]
        else:
            predictions = []
            with tqdm(input_segments) as I:
                I.set_description("Punctuating text segments")
                for i in I:
                    predictions.append(self.model.predict([i['text']])[0][0])

        return predictions

    @staticmethod
    def segment_text_blocks(text:str, length:int=250, overlap:int=30):
        """
        Splits a string of text into predefined slices of overlapping text with indexes linked back to the original text.
        This is done to bypass 512 token limit on transformer models by sequentially feeding segments of <512 tokens.

        Args:
            - text (str): input string of text to be split.
            - length (int): number of words in the (non-overlapping portion of) the otuput text segment.
            - overlap (int): number of words to overlap between text segements.

        Returns:
            - resp (lst): list of dicts specifying each text segment (containing the text and its start/end indices).
            E.g. [{...}, {"text": "...", 'start_idx': 31354, 'end_idx': 32648}, {...}]
        """
        wrds = text.replace('\n', ' ').split(" ")
        resp = []
        lst_chunk_idx = 0
        i = 0

        while True:
            # Words in the chunk and the overlapping portion
            wrds_len = wrds[(length * i):(length * (i + 1))]
            wrds_ovlp = wrds[(length * (i + 1)):((length * (i + 1)) + overlap)]
            wrds_split = wrds_len + wrds_ovlp

            # Break loop if no more words
            if not wrds_split:
                break

            wrds_str = " ".join(wrds_split)
            nxt_chunk_start_idx = len(" ".join(wrds_len))
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
    def combine_results(text_slices:list, original_text:str):
        """
        Given a full text and predictions of each slice combines the segmented predictions into a single text again.
        """
        original_text_lst = original_text.replace('\n', ' ').split(" ")
        original_text_lst = [i for i in original_text_lst if i]  # remove any empty strings
        original_text_len = len(original_text_lst)
        output_text = []
        index = 0

        # Remove final element of prediction list for formatting
        if len(text_slices[-1]) <= 3 and len(text_slices) > 1:
            text_slices = text_slices[:-1]

        # Cycle thrugh slices in the full prediction
        for _slice in text_slices:
            slice_wrds = len(_slice)

            # Cycle through words in each slice
            for ix, wrd in enumerate(_slice):
                if index == original_text_len:
                    break

                # Add each (non-overlapping) word and its associated prediction to output text
                if (original_text_lst[index] == str(list(wrd.keys())[0])) and (ix <= slice_wrds - 3) and (text_slices[-1] != _slice):
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)

                elif (original_text_lst[index] == str(list(wrd.keys())[0])) and (text_slices[-1] == _slice):
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)

        # Validate that the output text content (without predictions) is the same as the full plain text
        assert [i[0] for i in output_text] == original_text_lst
        return output_text

    def punctuate_texts(self, full_pred:list):
        """
        Given a list of predictions from the model, applies the predictions to the plaintext, restoring full punctuation and capitalisation.
        """
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
            if len(punct_resp) > 1 and punct_resp[-2] in TERMINALS:
                punct_wrd = punct_wrd.capitalize()

            # Add classified punctuation mark (and space) after word
            if label[0] != "O" and label[0] in PUNCT_LABELS and punct_wrd[-1] not in PUNCT_LABELS:
                punct_wrd += label[0]

            punct_resp += punct_wrd + " "

        # Remove unnecessary whitespace and ensure the first word is capitalised
        punct_resp = punct_resp.strip()
        punct_resp = punct_resp.replace("- ", "-")
        punct_resp = punct_resp[0].capitalize() + punct_resp[1:]

        # remove unwanted segmenting of numbers
        punct_resp = re.sub(r"([0-9]+)[\-:; ]([0-9]+)", r'\1\2', punct_resp)

        # Ensure text ends with a terminal
        if punct_resp[-1].isalnum():
            punct_resp += "."
        elif punct_resp[-1] not in TERMINALS:
            punct_resp = punct_resp[:-1] + "."

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


def run_rpunct(model_location, input_txt, output_path=None, use_cuda:bool=False):
    """
    Pipeline that forms an RestorePuncts object to conduct punctuation restoration over an input file of plaintext using the specified RPunct model.
    """
    # Generate an RPunct model instance
    punct_model = RestorePuncts(model_source=model_location, use_cuda=use_cuda)

    # Read input text
    print(f"\nReading plaintext from file: {input_txt}")
    with open(input_txt, 'r') as fp:
        unpunct_text = fp.read()

    # Restore punctuation to plaintext using RPunct
    punctuated = punct_model.punctuate(unpunct_text)

    # Output restored text
    if output_path is not None:
        # print output to command line
        print("\nPrinting punctuated text", end='\n\n')
        print(punctuated)
    else:
        # Check if output directory exists
        output_dir, _ = os.path.split(output_path)
        output_path_exists = os.path.isdir(output_dir)

        # print punctuated text to output file
        if output_path_exists:
            print(f"Writing punctuated text to file: {output_path}")
            with open(output_path, 'w') as fp:
                fp.write(punctuated)
        else:
            raise FileNotFoundError(f"Directory specified to ouptut text file to does not exist: {output_dir}")


if __name__ == "__main__":
    model = 'outputs/comp-perc-1e/'
    cuda = False
    input = 'tests/inferences/full-ep/test.txt'
    output = 'output.txt'

    run_rpunct(
        model_location=model,
        input_txt=input,
        output_txt=output,
        use_cuda=cuda
    )
