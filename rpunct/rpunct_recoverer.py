# Copyright (c) 2022 British Broadcasting Corporation

"""
Module supporting punctuation recovery and post-processing of raw STT output.
"""
import re
import os
import torch
import string
import decimal
from tqdm import tqdm
from jiwer import wer
from num2words import num2words

try:
    from rpunct.punctuate import *
    from rpunct.number_recoverer import *
    from rpunct.utils import *
except ModuleNotFoundError:
    from punctuate import *
    from number_recoverer import *
    from utils import *


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

    def process(self, input_transcript, conduct_number_recovery:bool=True, strip_existing_punct:bool=True):
        """
        Format input transcripts depending on their structure (segmented or pure plaintext)
        and pass them to RPunct to have punctuation/capitalisation/numbers recovered
        Then reconstructs the punctuated output in the same format as the input.

        Args:
            input_transcript: a piece of plaintext to be punctuated - e.g a string, list of strings, or list of Item objects.
            conduct_number_recovery: toggle the number recoverer.

        Returns:
            a list of of lists containing Item objects (where each Item.content is the recovered text)
            or a string of punctated text (depends on input type).
        """
        if type(input_transcript) == str:
            output = self.process_string(input_transcript, num_rec=conduct_number_recovery, strip_existing_punct=strip_existing_punct)

        elif type(input_transcript) == list and len(input_transcript) > 0:
            if all(isinstance(element, str) for element in input_transcript):
                output = self.process_string_segments(input_transcript, num_rec=conduct_number_recovery, strip_existing_punct=strip_existing_punct)

            elif all(isinstance(element, Item) for element in input_transcript):
                output = self.process_items(input_transcript, num_rec=conduct_number_recovery, strip_existing_punct=strip_existing_punct)

            elif all(isinstance(element, list) for element in input_transcript) and all(isinstance(element, Item) for segment in input_transcript for element in segment):
                output = self.process_item_segments(input_transcript, num_rec=conduct_number_recovery, strip_existing_punct=strip_existing_punct)

            else:
                raise TypeError("Input transcript to recoverer is not in a supported format/type (must be 'str' or 'Items')")
        else:
            raise TypeError("Input transcript to recoverer is not in a supported format/type (must be 'str' or 'Items')")

        return output

    def process_string(self, transcript:str, num_rec:bool=True, strip_existing_punct:bool=True) -> str:
        """
        Punctuation/number recovery pipeline a single string input transcript.

        Args:
            transcript: Single string transcript to conduct punctuation/number recovery on.
        """
        # Conduct number recovery process on segment transcript via NumberRecoverer
        if strip_existing_punct:
            transcript = self.strip_punctuation(transcript)
            # print('\nPlaintext: \n', transcript)
        else:
            transcript = transcript.replace("-", " ")

        if num_rec:
            transcript = self.number_recoverer.process(transcript)
            # print('\nNumber recovered: \n', transcript)

        recovered = self.recoverer.punctuate(transcript)
        # print('\nPunctuation recovered: \n', recovered)

        return recovered

    def process_string_segments(self, input_segments:list, num_rec:bool=True, strip_existing_punct:bool=True) -> list:
        """
        Punctuation/number recovery pipeline for list of segmented string inputs.

        Args:
            input_segments: List of string transcripts to conduct punctuation/number recovery on.
        """
        recovered_segments = []

        with tqdm(input_segments) as T:
            T.set_description("Restoring transcript punctuation")
            for transcript in T:
                # Conduct punctuation recovery process on segment transcript via RPunct
                punctuated = self.process_string(transcript.strip(), num_rec=num_rec, strip_existing_punct=strip_existing_punct)  # Restore punctuation to plaintext segment using RPunct

                # Format recovered transcript back into list of segments
                recovered_segments.append(punctuated)

        return recovered_segments

    def process_file(self, input_path, output_file_path=None, compute_wer=False, num_rec:bool=True, strip_existing_punct:bool=True):
        # Read input text
        print(f"\nReading input text from file: {input_path}")
        with open(input_path, 'r') as fp:
            input_text = fp.read()

        # Restore punctuation to plaintext using RPunct
        punctuated = self.process_string(input_text, num_rec=num_rec, strip_existing_punct=strip_existing_punct)

        self.output_to_file(punctuated, output_file_path)  # Output restored text (to a specified TXT file or the command line)

        if compute_wer:
            plaintext = self.strip_punctuation(input_text)
            self.word_error_rate(input_text, plaintext, punctuated)

        return punctuated

    def process_items(self, input_segment:list, num_rec:bool=True, strip_existing_punct:bool=True) -> list:
        """
        Punctuation/number recovery pipeline for 1D list of Item inputs.
        """
        # Extract list of words from list and convert into string sentence
        transcript = ' '.join([item.content.strip() for item in input_segment]).strip()

        if strip_existing_punct:
            transcript = self.strip_punctuation(transcript)
        else:
            transcript = transcript.replace("-", " ")

        # Conduct number recovery process on segment transcript via NumberRecoverer
        if num_rec:
            recovered = self.number_recoverer.process(transcript)
        else:
            recovered = transcript

        # Conduct punctuation recovery process on segment transcript via RPunct
        recovered = self.recoverer.punctuate(recovered)

        # Format recovered transcript back into list of segments
        recovered_words = recovered.split(' ')
        recovered_segment = self.itemise_segment(input_segment, recovered_words)

        return recovered_segment

    def process_item_segments(self, input_segments:list, num_rec:bool=True, strip_existing_punct:bool=True) -> list:
        """
        Punctuation/number recovery pipeline for (2D list of) segmented Item inputs.
        """
        output_segments = []

        with tqdm(input_segments) as T:
            T.set_description("Restoring transcript punctuation")
            # Recover punctuation of each segment of items individually
            for segment in T:
                recovered_segment_items = self.process_items(segment, num_rec=num_rec, strip_existing_punct=strip_existing_punct)
                output_segments.append(recovered_segment_items)

        return output_segments

    def strip_punctuation(self, punctuated_text:str) -> str:
        """
        Converts a string of punctuated text to plaintext with no punctuation or capitalisation and only natural language numbers.
        """
        # set lowercase and replace certain characters
        text = punctuated_text.lower()
        text = text.replace("\n", " ")
        text = text.replace(" - ", " ")
        text = text.replace("-", " ")
        text = text.replace("%", " percent")
        text = text.strip()

        text = text.split(" ")
        plaintext = []

        for word in text:
            if re.sub(r"[^0-9]", "", word):
                # Remove currency symbols
                if word[0] in CURRENCIES.values():
                    currency_word = list(CURRENCIES.keys())[list(CURRENCIES.values()).index(word[0])]
                    word = word[1:]

                    if word[-1] == 'm':
                        out_word = " " + word[-1] + "illion " + currency_word
                        word = word[:-1]
                    elif word[-1] == 'n':
                        out_word = " " + word[-2:] + "illion " + currency_word
                        word = word[:-2]
                    else:
                        out_word = " " + currency_word
                elif word[-1] == 'p':
                    out_word = " pence "
                    word = word[:-1]
                else:
                    out_word = ""

                # If fully numeric, convert to natural language
                if word.isnumeric():
                    try:
                        word = num2words(word) + out_word
                    except decimal.InvalidOperation:
                        word = word + out_word
                else:
                    word = word + out_word

            word = word.replace("-", " ")
            word = re.sub(r"[^0-9a-zA-Z' ]", "", word)
            plaintext.append(word)

        plaintext = " ".join(plaintext)

        return plaintext

    @staticmethod
    def _is_contracted_word(word:str):
        """
        Texts whether a recovered word has been comnstructured from multiple original words e.g. "fifty five" -> "55"
        """
        return word.count('-') > 0 or re.sub(r"[^0-9]", "", word)

    def itemise_segment(self, original_segment, recovered_segment) -> list:
        """
        Convert recovered words of a single segment into Items with time code data
        Need to recompute timing information on a per-word level for each segment as some words may have been concatenated by hyphenation,
        changing their end time.
        """
        # Capitalise the first word of every segment
        recovered_segment[0] = recovered_segment[0].capitalize()

        # Make the last word of a segment a full stop if no punctuation present
        last_word = recovered_segment[-1]
        if last_word[-1] not in string.punctuation:
            recovered_segment[-1] = last_word + '.'

        # Itemise segment words (taking into account that recovered segment may be fewer words due to hyphenation)
        index_orig = 0
        index_rec = 0
        total_fewer_words = 0

        while index_orig < len(original_segment) and index_rec < len(recovered_segment):
            # Get original and restored word from segment
            orig_item = original_segment[index_orig]
            rec_word = recovered_segment[index_rec]

            print(f"* Original: {orig_item.content}; Recovered: {rec_word};")

            # Skip over null words in original plaintext as these are removed from the punctuated text
            if not re.sub(r"[^0-9a-zA-Z]", "", orig_item.content):
                index_orig += 1
                total_fewer_words += 1
                continue

            # Create item object per recovered word including correct time codes
            if self._is_contracted_word(rec_word) or (re.sub(r"[^0-9a-zA-Z]", "", rec_word.lower()) != re.sub(r"[^0-9a-zA-Z]", "", orig_item.content.lower())):
                if rec_word.count('-') > 0:  # hyphenation case
                    # The no. of words skipped over in the orginal segment list equals the no. concatenated onto the leftmost word of the hyphenation
                    no_skip_words = rec_word.count('-')

                    # If any part of hyphenation is numerical, must also account for the words skipped due to digitisation
                    if re.sub(r"[^0-9]", "", rec_word):
                        # if in the case of a hyphenation + digitiation, must find where the digitisation happens within the hyphenation and start from here
                        start_position = [bool(re.sub(r"[^0-9]", "", subword)) for subword in rec_word.split('-')].index(True)
                        no_skip_words += self.calc_end_item_index(original_segment[index_orig:], recovered_segment[index_rec:], position=start_position)

                else:  # number recovery case
                    no_skip_words = self.calc_end_item_index(original_segment[index_orig:], recovered_segment[index_rec:])

                # Find the final word of the contraction in the orginal segments list
                end_item = original_segment[index_orig + no_skip_words]
                original_contents = original_segment[index_orig : index_orig + no_skip_words + 1]

                # Increment original segments list to jump to the end of the contracted word
                index_orig += no_skip_words
                total_fewer_words += no_skip_words

            else:  # one-to-one mapping case
                # Itemise with word & start/end times from associated original item
                end_item = orig_item
                original_contents = None

            # Itemise with word & start/end times from 1st/last word of the hyphenation
            new_item = Item(
                orig_item.start_time, end_item.end_time,
                rec_word, original_contents,
                orig_item.likelihood
            )

            # Return new itemised word to the segment
            recovered_segment[index_rec] = new_item
            index_orig += 1
            index_rec += 1

        # Verify all recovered words have been itemised
        try:
            assert index_rec == len(recovered_segment), \
                f"While reconstructing segment structure, one or more recovered words have been missed. \
                    \n Original text: {[item.content for item in original_segment]} \
                    \n Recovered text: {[item.content for item in recovered_segment]}"
        except AttributeError:
            assert index_rec == len(recovered_segment), \
                f"While reconstructing segment structure, one or more recovered words have been missed. \
                    \n Original text: {[item.content for item in original_segment]} \
                    \n Recovered text: {[item for item in recovered_segment]}"

        # Verify that the reconstructed segment is the same length as original (excluding words removed by hyphenation)
        assert len(recovered_segment) == (len(original_segment) - total_fewer_words), \
            f"While reconstructing segment structure, a mistake has occured. \
                \n Original text: {[item.content for item in original_segment]} \
                \n Recovered text: {[item.content for item in recovered_segment]}"

        # Return new itemised segment to the list of segments
        return recovered_segment

    def calc_end_item_index(self, plaintext_items_lst, recovered_words_lst, position=0) -> int:
        # Generate clean list of original words
        original_segment_words = [item.content.strip().lower() for item in plaintext_items_lst]

        # If the recovered word has a percent sign the index of the word 'percent' in the original text gives number of removals
        # Similar technique if a currency symbol is present
        recovered_word = re.sub(r'[.,:;!?]', '', recovered_words_lst[0])
        numerical_removals = 0

        if recovered_word.endswith('%') and original_segment_words.count('percent') > 0:
            numerical_removals = original_segment_words.index('percent')

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
        elif recovered_word.endswith('p') and original_segment_words.count('pence') > 0:
            numerical_removals = original_segment_words.index('pence')

        else:
            # Align original natural language numbers to recovered digits
            mapping = align_texts(original_segment_words, recovered_words_lst, position)
            grouped_orig_words = mapping[0][0]

            # failsafe if mapping for element in question contents spill over onto the next element
            if len(mapping) > 1 and len(mapping[1][0]) > 1 and not re.sub(r"[^0-9-]", "", mapping[1][1][0]):
                grouped_orig_words.extend(mapping[1][0][:-1])

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
