# Copyright (c) 2022 British Broadcasting Corporation

"""
Module supporting punctuation recovery and post-processing of raw STT output.
"""
import re
import os
import sys

import torch
import string
import decimal
import traceback
from tqdm import tqdm
from jiwer import wer
from kaldialign import align
from num2words import num2words

try:
    from rpunct.punctuate import RestorePuncts
    from rpunct.number_recoverer import NumberRecoverer
    from rpunct.utils import Item
except ModuleNotFoundError:
    from punctuate import RestorePuncts
    from number_recoverer import NumberRecoverer
    from utils import Item


class RPunctRecoverer:
    """
    A class for loading the RPunct object and exposing it to linguine code.
    """
    def __init__(self, model_location, use_cuda=True):
            self.recoverer = RestorePuncts(
                model_source=model_location,
                use_cuda=(use_cuda and torch.cuda.is_available())
            )
            self.number_recoverer = NumberRecoverer()

    def process(self, input_transcript, input_type='items', conduct_number_recovery=True):
        """
        Format input transcripts depending on their structure (segmented or pure plaintext)
        and pass them to RPunct to have punctuation/capitalisation/numbers recovered
        Then reconstructs the punctuated output in the same format as the input.

        Args:
            input_transcript: a list of Item objects (of STT plaintext) or a single string of plaintext.
            input_type: specify which one of these above inputs is the case (segmented Items or a text string)
            conduct_number_recovery: toggle the number recoverer.

        Returns:
            a list of of lists containing Item objects (where each Item.content is the recovered text)
            or a string of punctated text (depends on input type).
        """
        if input_type.startswith('str') and type(input_transcript) == str:
            output = self.process_strings(input_transcript, conduct_number_recovery)

        elif input_type.startswith('item') and type(input_transcript) == list:
            output = self.process_items(input_transcript, conduct_number_recovery)

        else:
            raise TypeError("Input transcript to recoverer does not match the specified input type ('str' or 'items')")

        return output

    def process_strings(self, transcript, num_rec=False):
        # print('\nPlaintext: \n', transcript)
        # Conduct number recovery process on segment transcript via NumberRecoverer
        if num_rec:
            recovered = self.number_recoverer.process(transcript)
            # print('\nNumber recovered: \n', transcript)
        else:
            recovered = transcript

        # Process entire transcript, then retroactively apply punctuation to words in segments
        recovered = self.recoverer.punctuate(recovered)
        # print('\nPunctuation recovered: \n', transcript)

        return recovered

    def process_items(self, input_segments, num_rec=False):
        # Extracts list of words from flattened list, converts to a single sting per speaker segment
        transcript_segments = [[re.sub(r"[^0-9a-zA-Z']", "", item.content).lower() for item in sublist] for sublist in input_segments]
        recovered_segment_words = []

        with tqdm(range(len(transcript_segments))) as T:
            T.set_description("Restoring transcript punctuation")
            for segment in T:
                # Conduct punctuation recovery process on segment transcript via RPunct
                transcript = ' '.join(transcript_segments[segment]).strip(' ')

                # Conduct number recovery process on segment transcript via NumberRecoverer
                if num_rec:
                    recovered = self.number_recoverer.process(transcript)
                else:
                    recovered = transcript

                # Conduct punctuation recovery process on segment transcript via RPunct
                recovered = self.recoverer.punctuate(recovered)

                # TEMPORARY FIX
                recovered = recovered.replace("%%", "%")

                # Format recovered transcript back into list of segments
                recovered_words = recovered.split(' ')
                recovered_segment_words.append(recovered_words)

        # Convert recovered words to Item objects
        output_segments = self.itemise_segments(input_segments, recovered_segment_words)

        return output_segments

    def strip_punctuation(self, truth_text):
        """
        Converts a string of truth text to plaintext of the same format as STT transcripts.
        """
        # set lowercase and replace certain characters
        text = truth_text.lower()

        text = text.replace("[music]", "").replace("[music playing]", "").replace("(crowd cheering)", "").replace("(crowd shouting)", "").replace("(crowd chattering)", "").replace("(people chattering)", "").replace("(gentle music)", "").replace("(dramatic music)", "").replace("(upbeat music)", "").replace("(soft music)", "").replace("[clock ticking]", "").replace("(engine rumbling)", "").replace("(somber music)", "").replace("[speaking russian]", "").replace("(speaking russian)", "").replace("(speaking in foreign language)", "").replace("(phone ringing)", "").replace("(thunder rumbling)", "").replace("[sirens]", "")

        text = text.replace("\n", " ")
        text = text.replace(" - ", " ")
        text = text.replace("-", " ")
        text = text.replace("%", " percent")
        text = re.sub(r"[^0-9a-zA-Z' ]", "", text)
        text = text.strip()

        # convert to list
        text = text.split(" ")
        plaintext = []

        for word in text:
            # if numerical, convert to a word
            try:
                word = num2words(word)
            except decimal.InvalidOperation:
                pass

            word = re.sub(r"[^0-9a-zA-Z' ]", "", word.replace("-", " "))
            plaintext.append(word)

        plaintext = " ".join(plaintext)

        return plaintext

    def output_to_file(self, data, file_path=None):
        if not file_path:
            # Output to command line
            print("\nPrinting punctuated text:", end='\n\n')
            print(data, end='\n\n')
        else:
            # Check if output directory exists
            output_dir, _ = os.path.split(file_path)
            output_path_exists = os.path.isdir(output_dir)

            # Output to file if the directory exists
            if output_path_exists:
                print(f"Writing punctuated text to file: {file_path}")
                with open(file_path, 'w') as fp:
                    fp.write(data)
            else:
                raise FileNotFoundError(f"Directory specified to ouptut text file to does not exist: {output_dir}")

    def word_error_rate(self, truth, stripped, predicted):
        # Uses `jiwer` to compute word error rates between punctuated and unpunctuated text
        wer_plaintext = wer(truth, stripped) * 100
        word_error_rate = wer(truth, predicted) * 100

        print("Word error rate:")
        print(f"\tNo recovery     : {wer_plaintext:.2f}%")
        print(f"\tRPunct recovery : {word_error_rate:.2f}%", end='\n\n')

    def itemise_segments(self, all_original_segments, all_recovered_segments):
        """
        Convert recovered words to segments of items (not just strings) including time code data
        Need to recompute timing information on a per-word level for each segment as some words may have been concatenated by hyphenation,
        changing their end time.
        """

        for index_segment in range(len(all_original_segments)):
            # Get original (plaintext) and punctuation-restored segments
            original_segment = all_original_segments[index_segment]
            recovered_segment = all_recovered_segments[index_segment]

            # Capitalise the first word of every segment
            recovered_segment[0] = recovered_segment[0].capitalize()

            # Make the last word of a segment a full stop if no punctuation present
            last_word = recovered_segment[-1]
            if last_word[-1] not in string.punctuation:
                recovered_segment[-1] = last_word + '.'

            # Itemise segment words (taking into account that recovered segment may be fewer words due to hyphenation)
            index_orig = 0
            total_fewer_words = 0

            for index_rec in range(len(recovered_segment)):
                # Get original and restored word from segment
                orig_item = original_segment[index_orig]
                rec_word = recovered_segment[index_rec]

                # Create item object per recovered word including correct time codes
                if rec_word.count('-') > 0:  # hyphenation case
                    # The no. of words skipped over in the orginal segment list equals the no. concatenated onto the leftmost word of the hyphenation
                    no_skip_words = rec_word.count('-')

                    # If any part of hyphenation is numerical, include the skipped words that have been converted to digits
                    if re.sub(r"[^0-9]", "", rec_word):
                        split_word = rec_word.split('-')

                        for sub_word_index in range(len(split_word)):
                            if re.sub(r"[^0-9]", "", split_word[sub_word_index]):
                                no_skip_words += self.calc_end_item_index(original_segment, index_orig, recovered_segment, index_rec, position=sub_word_index)
                                # print(f" * [seg {index_segment} word {index_rec}] Original: {orig_item.content}...{original_segment[index_orig + no_skip_words].content}; Recovered: {rec_word}; Removed: {no_skip_words} / {total_fewer_words};")

                    # Find the final word of the hyphenation in the orginal segments list
                    end_item = original_segment[index_orig + no_skip_words]

                    # Itemise with word & start/end times from 1st/last word of the hyphenation
                    new_item = Item(orig_item.start_time, end_item.end_time, rec_word)

                    # Increment original segments list to jump to the end of the hyphenated word
                    index_orig += no_skip_words
                    total_fewer_words += no_skip_words

                # number recovery case
                elif re.sub(r"[^0-9]", "", rec_word):
                    no_skip_words = self.calc_end_item_index(original_segment, index_orig, recovered_segment, index_rec)
                    end_item = original_segment[index_orig + no_skip_words]

                    new_item = Item(orig_item.start_time, end_item.end_time, rec_word)

                    index_orig += no_skip_words
                    total_fewer_words += no_skip_words

                else:
                    # Itemise with word & start/end times from associated original item
                    new_item = Item(orig_item.start_time, orig_item.end_time, rec_word)

                # Return new itemised word to the segment
                recovered_segment[index_rec] = new_item
                index_orig += 1

            # Verify that the reconstructed segment is the same length as original (excluding words removed by hyphenation)
            assert len(recovered_segment) == (len(original_segment) - total_fewer_words), \
                f"While reconstructing segment structure, a mistake has occured. \
                    \n Original text: {[item.content for item in original_segment]} \
                    \n Recovered text: {[item.content for item in recovered_segment]}"

            # Return new itemised segment to the list of segments
            all_recovered_segments[index_segment] = recovered_segment

        return all_recovered_segments

    @staticmethod
    def align_original_recovered(original_lst, recovered_lst):
        stripped_recovered_lst = [re.sub(r"[^0-9a-zA-Z'%£$€ ]", "", item.replace("-", " ")).lower() for item in recovered_lst]
        stripped_recovered_lst = " ".join(stripped_recovered_lst).split(" ")

        EPS = '*'
        alignment = align(original_lst, recovered_lst, EPS)
        mapping = []

        for ref, hyp in alignment:
            if ref == EPS:
                # insertion (one-to-many)
                raise ValueError("Insertion found and not handled.")
            elif hyp == EPS:
                # deletion (many-to-one)
                mapping[-1][0].append(ref)  # append new word to multi-word element
            else:
                # single substitution (one-to-one mapping)
                mapping.append([[ref], [hyp]])

        return mapping


    def calc_end_item_index(self, plaintext_items_lst, current_plaintext_index, recovered_words_lst, current_recovered_index, position=0):
        # Generate clean list of original words
        original_segment_words = [item.content.lower() for item in plaintext_items_lst[current_plaintext_index:]]

        # If the recovered word has a percent sign the index of the word 'percent' in the original text gives number of removals
        # Similar technique if a currency symbol is present
        recovered_word = recovered_words_lst[current_recovered_index]
        numerical_removals = 0

        if recovered_word.endswith('%'):
            if original_segment_words.count('percent') > 0:
                try:
                    numerical_removals = original_segment_words.index('percent')
                except ValueError:
                    raise ValueError(f"Can't find 'percent' in list. Recovered word: {recovered_word}, original segment: {original_segment_words}.")

        elif recovered_word.startswith('£'):
            if original_segment_words.count('pounds') > 0:
                numerical_removals = original_segment_words.index('pounds')
            elif original_segment_words.count('pound') > 0:
                numerical_removals = original_segment_words.index('pound')
        elif recovered_word.startswith('$'):
            if original_segment_words.count('dollars') > 0:
                numerical_removals = original_segment_words.index('dollars')
            elif original_segment_words.count('dollar') > 0:
                numerical_removals = original_segment_words.index('dollar')
        elif recovered_word.startswith('€'):
            if original_segment_words.count('euros') > 0:
                numerical_removals = original_segment_words.index('euros')
            elif original_segment_words.count('euro') > 0:
                numerical_removals = original_segment_words.index('euro')

        else:
            # Align original natural language numbers to recovered digits
            mapping = self.align_original_recovered(original_segment_words, recovered_words_lst[current_recovered_index:])

            grouped_orig_words = mapping[position][0]

            # failsafe if mapping for element in question contents spill over onto the next element
            if len(mapping) > position + 1 and len(mapping[position + 1][0]) > 1 and not re.sub(r"[^0-9]", "", mapping[position + 1][1][0]):
                grouped_orig_words.extend(mapping[position + 1][0][:-1])

            numerical_removals = len(grouped_orig_words) - 1

        return numerical_removals

    @classmethod
    def load(cls, model_path=None, bbc_data_loc=False):
        # if bbc_data_loc:
        #     try:
        #         logger.info('Searching for existing model files at default bbc-data location')
        #         model_loc = bbc.data.path(model_path)
        #         model_loc = model_loc + '/bert-restore-punctuation'
        #         logger.info('Successfully located model files')
        #     except bbc.data.exceptions.DatasetNotInstalledError:
        #         logger.error('Dataset not found at default bbc-data path.')
        #         logger.error('Original error message:')
        #         logger.error(traceback.format_exc())
        #         logger.error('Please ensure wormhole credentials are set, and then run\
        #             `bbc-data pull %s` to download model files' %model_path)
        #         logger.error('Exiting')
        #         sys.exit(1)
        #     except Exception:
        #         logger.error('Unable to use bbc-data, please ensure wormhole credentials are set')
        #         logger.error('Exiting')
        #         sys.exit(1)

        model_loc = model_path
        rpunct_recoverer = RPunctRecoverer(model_location=model_loc)

        return rpunct_recoverer

    def run(self, input_path, output_file_path=None, compute_wer=False):
        # Read input text
        print(f"\nReading input text from file: {input_path}")
        with open(input_path, 'r') as fp:
            input_text = fp.read()

        plaintext = self.strip_punctuation(input_text)  # Convert input transcript to plaintext (no punctuation)

        punctuated = self.process_strings(plaintext, num_rec=True)  # Restore punctuation to plaintext using RPunct

        self.output_to_file(punctuated, output_file_path)  # Output restored text (to a specified TXT file or the command line)

        if compute_wer:
            self.word_error_rate(input_text, plaintext, punctuated)

        return punctuated


def rpunct_main(model_location, input_txt, output_txt=None, use_cuda=False, wer=False):
    # Generate an RPunct model instance
    punct_model = RPunctRecoverer(model_location=model_location, use_cuda=use_cuda)

    # Run e2e inference pipeline
    output = punct_model.run(input_txt, output_txt, compute_wer=wer)

    return output


if __name__ == "__main__":
    model_default = 'outputs/best_model'
    input_default = 'tests/inferences/full-ep/test.txt'
    rpunct_main(model_default, input_default)
