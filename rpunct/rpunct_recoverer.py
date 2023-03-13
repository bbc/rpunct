# Copyright (c) 2022 British Broadcasting Corporation

"""
Module supporting punctuation recovery and post-processing of raw STT output.
"""
import os
import re
import tqdm
import torch
import string
import decimal
from jiwer import wer
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
    def __init__(self, model_source, use_cuda=True):
            self.recoverer = RestorePuncts(
                model_source=model_source,
                use_cuda=(use_cuda and torch.cuda.is_available())
            )
            self.number_recoverer = NumberRecoverer()

    def process(self, input_transcript, input_type='segmented', conduct_number_recovery=False):
        """
        Format input transcripts depending on their structure (segmented or pure plaintext)
        and pass them to RPunct to have punctuation/capitalisation/numbers recovered
        Then reconstructs the punctuated output in the same format as the input.

        Args:
            input_transcript: a list of lists containing Item objects (of STT plaintext) or a single string of plaintext.
            input_type: specify which one of these above inputs is the case (segmented Items or a text string)
            conduct_number_recovery: toggle the number recoverer.

        Returns:
            a list of of lists containing Item objects (where each Item.content is the recovered text)
            or a string of punctated text (depends on input type).
        """
        if input_type == 'str' and type(input_transcript) == str:
            output = self.process_strings(input_transcript, conduct_number_recovery)

        elif input_type.startswith('seg') and type(input_transcript) == list:
            output = self.process_segments(input_transcript, conduct_number_recovery)

        else:
            raise TypeError("Input transcript to recoverer does not match the specified input type ('str' or 'segments')")

        return output

    def process_strings(self, transcript, num_rec=False):
        # Conduct number recovery process on segment transcript via NumberRecoverer
        if num_rec:
            recovered = self.number_recoverer.process(transcript)
        else:
            recovered = transcript

        # Process entire transcript, then retroactively apply punctuation to words in segments
        recovered = self.recoverer.punctuate(recovered)

        return recovered

    def process_segments(self, input_segments, num_rec=False):
        # Extracts list of words from flattened list, converts to a single sting per speaker segment
        transcript_segments = [[re.sub(r"[^0-9a-zA-Z']", "", item.content).lower() for item in sublist] for sublist in input_segments]
        recovered_segment_words = []

        with tqdm(range(len(transcript_segments))) as T:
            T.set_description("Restoring transcript punctuation")
            for segment in T:
                # Conduct punctuation recovery process on segment transcript via RPunct
                transcript = ' '.join(transcript_segments[segment]).strip(' ')
                recovered = self.recoverer.punctuate(transcript, lang='en')

                # Conduct number recovery process on segment transcript via NumberRecoverer
                if num_rec:
                    recovered = self.number_recoverer.process(recovered)

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
        text = text.replace("\n", " ")
        text = text.replace(" - ", " ")
        text = text.replace("' ", " ")
        text = text.replace("%", " percent")

        # convert to list
        text = text.split(" ")
        plaintext = []

        for word in text:
            # if numerical, convert to a word
            try:
                word = num2words(word)
            except decimal.InvalidOperation:
                pass

            plaintext.append(word)

        plaintext = " ".join(plaintext)

        plaintext = plaintext.replace("-", "- ")
        plaintext = re.sub(r"[^0-9a-zA-Z' ]", "", plaintext)

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

    @staticmethod
    def itemise_segments(all_original_segments, all_recovered_segments):
        """
        Convert recovered words to segments of items (not just strings) including time code data
        Need to recompute timing information on a per-word level for each segment as some words may have been concatenated by hyphenation,
        changing their end time.
        """

        for segment in range(len(all_original_segments)):
            # Get original (plaintext) and punctuation-restored segments
            original_segment = all_original_segments[segment]
            recovered_segment = all_recovered_segments[segment]

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

                    # Find the final word of the hyphenation in the orginal segments list
                    end_item = original_segment[index_orig + no_skip_words]

                    # Itemise with word & start/end times from 1st/last word of the hyphenation
                    new_item = Item(orig_item.start_time, end_item.end_time, rec_word)

                    # Increment original segments list to jump to the end of the hyphenated word
                    index_orig += no_skip_words
                    total_fewer_words += no_skip_words

                elif rec_word.isnumeric() or re.sub(r"[^0-9a-zA-Z]", "", rec_word).isnumeric():  # number recovery case
                    # Itemise with word & start/end times from associated original item
                    new_item = Item(orig_item.start_time, orig_item.end_time, rec_word)

                else:
                    # Itemise with word & start/end times from associated original item
                    new_item = Item(orig_item.start_time, orig_item.end_time, rec_word)

                # Return new itemised word to the segment
                recovered_segment[index_rec] = new_item
                index_orig += 1

            # Verify that the reconstructed segment is the same length as original (excluding words removed by hyphenation)
            assert len(recovered_segment) == (len(original_segment) - total_fewer_words), "While reconstructing segment structure, a mistake has occured"

            # Return new itemised segment to the list of segments
            all_recovered_segments[segment] = recovered_segment

        return all_recovered_segments

    def run(self, input_path, output_file_path=None, compute_wer=False):
        # Read input text
        print(f"\nReading input text from file: {input_path}")
        with open(input_path, 'r') as fp:
            input_text = fp.read()

        plaintext = self.strip_punctuation(input_text)  # Convert input transcript to plaintext (no punctuation)
        punctuated = self.process_strings(plaintext)  # Restore punctuation to plaintext using RPunct

        self.output_to_file(self, punctuated, output_file_path)  # Output restored text (to a specified TXT file or the command line)

        if compute_wer:
            self.word_error_rate(input_text, plaintext, punctuated)

        return punctuated


def rpunct_main(model_location, input_txt, output_txt=None, use_cuda=False):
    # Generate an RPunct model instance
    punct_model = RPunctRecoverer(model_source=model_location, use_cuda=use_cuda)

    # Run e2e inference pipeline
    output = punct_model.run(input_txt, output_txt)

    return output


if __name__ == "__main__":
    model_default = 'outputs/best_model'
    input_default = 'tests/inferences/full-ep/test.txt'
    rpunct_main(model_default, input_default)
