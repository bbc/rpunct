# Copyright (c) 2022 British Broadcasting Corporation

"""
Module supporting punctuation recovery and post-processing of raw STT output.
"""
import re
import string
from kaldialign import align
from num2words import num2words
from number_parser import parse as number_parser, parse_number as individual_number_parser


TERMINALS = ['.', '!', '?']

STRING_NUMBERS = {
    'million': 'm',
    'billion': 'bn',
    'trillion': 'tn'
}

CURRENCIES = {
    'pound': '£',
    'euro': '€',
    'dollar': '$'
}

DECADES = {
    'hundreds': '00s',
    'tens': '10s',
    'twenties': '20s',
    'thirties': '30s',
    'fourties': '40s',
    'fifties': '50s',
    'sixties': '60s',
    'seventies': '70s',
    'eighties': '80s',
    'nineties': '90s'
}

ORDINALS = [
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh",
    "Eighth", "Ninth", "Tenth", "Eleventh", "Twelfth", "Thirteenth",
    "Fourteenth", "Fifteenth", "Sixteenth", "Seventeenth", "Eighteenth",
    "Nineteenth", "Twentieth", "Thirtieth", "Fortieth", "Fiftieth",
    "Sixtieth", "Seventieth", "Eightieth", "Ninetieth", "Hundredth",
    "Thousandth", "Millionth", "Billionth"
]


class NumberRecoverer:
    """
    Parent class for number recovery. Uses `number_parser` (https://pypi.org/project/number-parser/)
    to convert numbers written in the natural language to their equivalent numeric forms.
    """

    def __init__(self, correct_currencies=True, correct_bbc_style_numbers=True, comma_separators=True, restore_decimals=True, restore_percentages=True):
        self.correct_currencies = correct_currencies
        self.correct_bbc_style_numbers = correct_bbc_style_numbers
        self.comma_separators = comma_separators
        self.restore_decimals = restore_decimals
        self.restore_percentages = restore_percentages

    def process(self, input_text:str):
        """
        Pipeline for recovering formatting of numbers within a piece of text.
        """
        # Convert numerical strings to digits in text using `number_parser` package
        parsed_text = self.number_parser(input_text)

        # Convert percentages to use the symbol notation
        if self.restore_percentages:
            parsed_text = self.insert_percentage_symbols(parsed_text)

        # Restore decimal points
        parsed_list = parsed_text.split(" ")
        if self.restore_decimals:
            parsed_list = self.replace_decimal_points(parsed_list)

        # Correct currencies, BBC styling of numbers, and insert currency separators into numbers >= 10,000
        output_text = ""
        for i in range(len(parsed_list)):
            word = parsed_list[i]
            stripped_word = re.sub(r"[^0-9a-zA-Z]", "", word).lower()

            if stripped_word:
                # Restore currency words to their symbols
                if self.correct_currencies and self.is_currency(stripped_word):
                        output_text = self.insert_currency_symbols(output_text, word)

                # BBC Style Guide asserts that single digit numbers should be written as words, so revert those numbers
                elif self.correct_bbc_style_numbers and self.is_stylable_number(word):
                        output_text = self.bbc_style_numbers(output_text, word)

                # Format numbers with many digits to include comma separators
                elif self.comma_separators and stripped_word.isnumeric() and int(stripped_word) >= 10000:
                    output_text += self.insert_comma_seperators(word)

                # Replace colloquial decades terms with digits
                elif self.is_date_decade(stripped_word) and parsed_list[i - 1].isnumeric():
                    output_text = self.decades_to_digits(output_text, stripped_word)

                else:
                    output_text += word + " "

        # Restore/format ordinal numbers
        output_text = self.recover_ordinals(input_text, output_text)

        # Remove any unwanted whitespace
        output_text = output_text.strip()
        output_text = output_text.replace(" - ", "-")
        output_text = output_text.replace("- ", "-")
        output_text = re.sub(r'([0-9]*)-([0-9]*)', r'\1\2', output_text)  # remove unwanted segmenting of numbers

        return output_text

    def number_parser(self, text):
        """
        Converts numbers in text to digits instead of words.
        Optionally very large (>= million) numbers can stay as words.
        """
        # BBC Style Guide asserts that single digit numbers should be written as words, so don't convert those numbers
        # also ensure large number definitions remain as words in text
        if self.bbc_style_numbers:
            # Swap digits that we don't want to be parsed with control characters (from STRING_NUMBERS lookup table)
            control_chars = list(enumerate(STRING_NUMBERS.keys()))
            for index, number in control_chars:
                text = text.replace(number, f"\\{index}")

            # Convert all other numbers to digits
            parsed = number_parser(text)

            # Return control characters to words
            control_chars.reverse()
            for index, number in control_chars:
                parsed = parsed.replace(f"\\{index}", number)
        else:
            parsed = number_parser(text)

        # `number_parser` adds spaces around numbers, interrupting the formatting of any trailing punctuation, so re-concatenate
        for punct in string.punctuation:
            parsed = parsed.replace(f" {punct}", f"{punct}")

        return parsed

    @staticmethod
    def is_currency(word):
        """Checks if a word is a currency term."""
        return (word in CURRENCIES.keys()) or (word[-1] == 's' and word[:-1] in CURRENCIES.keys())

    @staticmethod
    def is_stylable_number(number):
        """Checks if a number is single digit and should be converted to a word according to the BBC Style Guide."""
        # (Includes failsafe if number is immediately followed by a punctuation character)
        return (number.isnumeric() and int(number) < 10) or (not number[-1].isnumeric() and number[:-1].isnumeric() and int(number[:-1]) < 10)

    @staticmethod
    def is_date_decade(word):
        """Checks if a word is a natural language date term specifying a decade (e.g. nineties)"""
        return word in DECADES.keys()

    @staticmethod
    def is_ordinal(text):
        """Checks if a natural language number is an ordinal"""
        return text.capitalize() in ORDINALS

    @staticmethod
    def replace_decimal_points(text_list):
        """
        Correctly format numbers with decimal places (e.g. "1 point 5" -> "1.5").
        """
        corrected_list = []
        i = 0

        while i < len(text_list):
            # Cycle through words in the text until the word "point" appears (can't be the 1st or last word)
            word = text_list[i]

            if re.sub(r"[^0-9a-zA-Z]", "", word) == "point" and i > 0 and i < len(text_list) - 1:
                # When a case for decimal point formatting is identified, combine stripped full no. and decimal digits together around a `.` char
                pre_word = text_list[i - 1]
                pre_word_stripped = re.sub(r"[,.?!%]", "", pre_word)
                post_word = text_list[i + 1]
                post_word_stripped = re.sub(r"[^0-9a-zA-Z]", "", post_word)

                # Ensure both words around the deminal point are numerical
                # N.B. concatenate the original (not stripped) post word s.t. any trailing punctuation is preserved
                if pre_word_stripped.isnumeric() and post_word_stripped.isnumeric():
                    full_number = pre_word_stripped + '.' + post_word
                    corrected_list = corrected_list[:-1]
                    corrected_list.append(full_number)
                    i += 2

                # All other words are simply added back into the text
                else:
                    corrected_list.append(word)
                    i += 1
            else:
                corrected_list.append(word)
                i += 1

        return corrected_list

    @staticmethod
    def insert_currency_symbols(text, currency):
        """
        Converts currency terms in text to symbols before their respective numerical values.
        """
        # Get plaintext version of currency keyword
        stripped_currency = re.sub(r"[^0-9a-zA-Z]", "", currency).lower()
        if stripped_currency[-1] == 's':
            stripped_currency = stripped_currency[:-1]

        found = False
        lookback = 1

        # Scan through lookback window to find the number to which the currency symbol punctuates
        text_list = text.split(" ")
        quantifying_tag = ""

        while lookback < 5:
            prev_word = text_list[-lookback]
            prev_word_stripped = re.sub(r"[^0-9a-zA-Z]", "", prev_word)

            # When a numeric word is found, reconstruct the output text around this (i.e. previous_text + currency_symbol + number + trailing_text)
            if prev_word_stripped.isnumeric():
                new_output_text = text_list[:-lookback]  # previous text before currency number

                new_output_text.append(CURRENCIES.get(stripped_currency) + prev_word +  quantifying_tag)  # currency symbol and number (including any large denominators - e.g. million -> m)

                new_output_text.extend([word for word in text_list[-lookback + 1:] if word not in STRING_NUMBERS.keys()])  # trailing text after currency symbol/number
                text = " ".join(new_output_text)

                # Add back in any punctuation trailing the original currency keyword
                if not currency[-1].isalnum():
                    text = text[:-1] + currency[-1] + " "

                found = True
                break

            # convert big number words (e.g. million) to condensed versions (e.g. mn)
            elif prev_word_stripped in STRING_NUMBERS.keys():
                quantifying_tag = STRING_NUMBERS[prev_word_stripped] + quantifying_tag
                lookback += 1

            else:
                lookback += 1

        # Keep the currency keyword as text if no numeric words found in lookback window
        if not found:
            text += currency + " "

        return text

    @staticmethod
    def insert_percentage_symbols(text):
        """
        Converts the natural language term 'percent' in text to the symbol '%' if following a digit.
        """
        text = re.sub(r'([0-9]+) percent', r'\1%', text)

        return text

    @staticmethod
    def bbc_style_numbers(text, number):
        """
        Converts small numbers back from digits to words (according to BBC Style Guide rules).
        """
        if not number[-1].isnumeric():
            # Don't convert number if it is involved with some mathematical/currency expression
            if number[-1] in ['%', '*', '+', '<', '>', '$', '£', '€']:
                text += number + " "
                return text
            # But separate off any trailing punctuation other than this and continue
            else:
                number, end_chars = number[:-1], number[-1]
        else:
            end_chars = ""

        # Return number to word notation
        formatted_number = num2words(number)

        # Capitalise numeric word if at start of sentence
        if text == "" or (len(text) > 2 and text[-2] in TERMINALS):
            formatted_number = formatted_number.capitalize()

        # Add word to text
        text += formatted_number + end_chars + " "

        return text

    @staticmethod
    def insert_comma_seperators(number):
        """
        Inserts comma separators into numbers with many digits to break up 1000s (e.g. '100000' -> '100,000').
        """
        # Strip leading non-numeric characters and trailing digits after the decimal point in floats
        if not number[0].isalnum():
            start_char = number[0]
            number = number[1:]
        else:
            start_char = ""

        if "." in number:
            number, end_chars = number.split(".")
            end_chars = "." + end_chars
        else:
            end_chars = ""

        # Cycle through number in reverse order and insert comma separators every three digits
        for i in range(len(number) - 3, 0, -3):
            number = number[:i] + "," + number[i:]

        # Reconcatenate leading/trailing chars/digits
        number = start_char + number + end_chars + " "

        return number

    @staticmethod
    def decades_to_digits(text, decade):
        if text.endswith(" "):
            text = text[:-1]

        output_text_list = text.split(" ")
        output = " ".join(output_text_list[:-1]) + " " + output_text_list[-1][:2] + DECADES[decade] + " "

        return output

    @staticmethod
    def format_ordinal(number):
        if number.isnumeric():
            number = int(number)
        else:
            number = individual_number_parser(number)

        if number < 10:
            ordinal_word = num2words(number, to='ordinal')
        else:
            ordinal_function = lambda n: "%s%s" % ("{:,}".format(n),"tsnrhtdd"[(n//10%10 != 1) * (n%10 < 4) * n%10::4])
            ordinal_word = ordinal_function(number)

        return ordinal_word

    def recover_ordinals(self, plain, recovered):
        # Align number recovered text with original s.t. we can find where ordinals have been lost
        plain = plain.split(" ")
        recovered = recovered.split(" ")

        stripped_recovered = [re.sub(r"[^0-9a-zA-Z'%£$€ ]", "", item.replace("-", " ")).lower() for item in recovered]
        stripped_recovered = " ".join(stripped_recovered).strip().split(" ")

        EPS = '*'
        alignment = align(plain, stripped_recovered, EPS)
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

        formatted_output = ""

        for plain_words, rec_word in mapping:
            if self.is_ordinal(plain_words[-1]):
                ordinal_word = self.format_ordinal(rec_word[0])
                formatted_output += ordinal_word + " "
            else:
                formatted_output += rec_word[0] + " "

        return formatted_output
