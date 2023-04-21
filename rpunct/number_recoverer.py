# Copyright (c) 2022 British Broadcasting Corporation

"""
Module supporting punctuation recovery and post-processing of raw STT output.
"""
import re
from num2words import num2words
from number_parser import parse as number_parser, parse_number as individual_number_parser
import decimal

try:
    from rpunct.utils import *
except ModuleNotFoundError:
    from utils import *

TERMINALS = ['.', '!', '?']

STRING_NUMBERS = {
    'million': 'm',
    'billion': 'bn',
    'trillion': 'tn'
}

CURRENCIES = {
    'pound': '£',
    'euro': '€',
    'dollar': '$',
    'yen': '¥'
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

        # Initial currency restoration: 'X pence' -> 'Xp'
        if self.correct_currencies:
            parsed_text = self.convert_pence_to_p(parsed_text)

        # Convert percentages to use the symbol notation
        if self.restore_percentages:
            parsed_text = self.insert_percentage_symbols(parsed_text)

        # Restore decimal points
        parsed_list = parsed_text.split()
        if self.restore_decimals:
            parsed_list = self.replace_decimal_points(parsed_list)

        # Correct currencies, BBC styling of numbers, and insert currency separators into numbers >= 10,000
        output_text = ""
        for i, word in enumerate(parsed_list):
            stripped_word = re.sub(r"[^0-9a-zA-Z]", "", word).lower().strip()

            if stripped_word:
                # Restore currency words to their symbols
                if self.correct_currencies and self._is_currency_symbol(stripped_word):
                    output_text = self.insert_currency_symbols(output_text, word)

                # BBC Style Guide asserts that single digit numbers should be written as words, so revert those numbers
                elif self.correct_bbc_style_numbers and self._is_stylable_number(word) and parsed_list[i - 1] != 'radio':
                    output_text = self.bbc_style_numbers(output_text, word)

                # Format numbers with many digits to include comma separators
                elif self.comma_separators and re.sub(r"[^0-9]", "", word) and int(re.sub(r"[^0-9]", "", word)) >= 10000:
                    output_text += self.insert_comma_seperators(word)

                # Replace colloquial decades terms with digits
                elif self._is_date_decade(stripped_word) and parsed_list[i - 1].isnumeric():
                    output_text = self.decades_to_digits(output_text, stripped_word)

                else:
                    output_text += word.strip() + " "

        # Restore/format ordinal numbers
        output_text = self.recover_ordinals(input_text, output_text)
        output_text = output_text.strip()

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
            text = re.sub(r'([0-9]+) thousand', r'\1 \\5', text)
            control_chars = list(enumerate(STRING_NUMBERS.keys()))
            for index, number in control_chars:
                text = text.replace(number, f"\\{index}")

            # Convert all other numbers to digits
            parsed = number_parser(text)

            # Return control characters to words
            parsed = re.sub(r'([0-9]+) \\5', r'\1 thousand', parsed)
            control_chars.reverse()
            for index, number in control_chars:
                parsed = parsed.replace(f"\\{index}", number)
        else:
            parsed = number_parser(text)

        # `number_parser` adds spaces around numbers, interrupting the formatting of any trailing punctuation, so re-concatenate
        parsed = re.sub(r"([£$€¥]{1}[0-9]+)[ ]{1}([0-9]{2}[!?,-.:; ]{1})", r'\1.\2', parsed)
        parsed = re.sub(r"([0-9.]+)[ ]{1}([%!?,-.:;'$]+)", r"\1\2", parsed)
        parsed = re.sub(r'([0-9]+)[\-:]{1}([0-9]+)', r'\1\2', parsed)
        parsed = re.sub(r"([0-9]+)[ ]{1}([.]{1}[0-9a-zA-Z]{1})$", r'\1\2', parsed)
        parsed = parsed.replace(" - ", "-")
        parsed = parsed.replace("- ", "-")

        return parsed

    @staticmethod
    def _is_currency_symbol(word):
        """Checks if a word is a currency term."""
        return (word in CURRENCIES.keys()) or (word[-1] == 's' and word[:-1] in CURRENCIES.keys())

    @staticmethod
    def _is_stylable_number(number):
        """Checks if a number is single digit and should be converted to a word according to the BBC Style Guide."""
        # (Includes failsafe if number is immediately followed by a punctuation character)
        return (number.isnumeric() and int(number) < 10 and len(number) < 2) \
            or (len(number) > 0 and number[:-1].isnumeric() and not number[-1].isnumeric() and int(number[:-1]) < 10 and len(number[:-1]) < 2) \
            or (len(number) > 1 and number[:-2].isnumeric() and number.endswith("'s") and int(number[:-2]) < 10 and len(number[:-2]) < 2)

    @staticmethod
    def _is_date_decade(word):
        """Checks if a word is a natural language date term specifying a decade (e.g. nineties)"""
        return word in DECADES.keys()

    @staticmethod
    def _is_ordinal(text):
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
        text_list = text.strip().split()
        quantifying_tag = ""
        lookback_range = min(5, len(text_list) + 1)

        while lookback < lookback_range:
            prev_word = text_list[-lookback]
            prev_word_stripped = re.sub(r"[^0-9a-zA-Z]", "", prev_word)

            # When a numeric word is found, reconstruct the output text around this (i.e. previous_text + currency_symbol + number + trailing_text)
            if prev_word_stripped.isnumeric():
                new_output_text = text_list[:-lookback]  # previous text before currency number
                new_output_text.append(CURRENCIES.get(stripped_currency) + prev_word +  quantifying_tag)  # currency symbol and number (including any large denominators - e.g. million -> m)
                out_text = " ".join(new_output_text)

                # Add back in any punctuation trailing the original currency keyword
                if not currency[-1].isalnum():
                    out_text += currency[-1] + " "
                else:
                    out_text += " "

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
            out_text = text + currency + " "

        return out_text

    @staticmethod
    def convert_pence_to_p(text):
        """
        Converts the natural language term 'pence' in text to the symbol 'p' following a monetary value.
        """
        text = re.sub(r'([£$€¥]{1}[0-9]+[.]{1}[0-9]{2}) pence', r'\1', text)
        text = re.sub(r'([0-9]+) pence', r'\1p', text)

        return text

    @staticmethod
    def insert_percentage_symbols(text):
        """
        Converts the natural language term 'percent' in text to the symbol '%' if following a digit.
        """
        text = re.sub(r'([0-9]+[0-9.]*) percent', r'\1%', text)

        return text

    @staticmethod
    def bbc_style_numbers(text, number):
        """
        Converts small numbers back from digits to words (according to BBC Style Guide rules).
        """
        if not number[-1].isnumeric():
            # Don't convert number if it is involved with some mathematical/currency expression
            if number[-1] in ['%', 'p']:
                text += number + " "
                return text
            # But separate off any trailing punctuation other than this and continue
            elif number[:-1].isnumeric():
                number, end_chars = number[:-1], number[-1]
            else:
                number, end_chars = number[:-2], number[-2:]
        else:
            end_chars = ""

        # Return number to word notation
        try:
            formatted_number = num2words(number)
        except decimal.InvalidOperation:
            print(f"numb: '{number}' end chars: '{end_chars}'")
            exit(1)

        # Capitalise numeric word if at start of sentence
        if text == "" or (len(text) > 2 and text[-2] in TERMINALS):
            formatted_number = formatted_number.capitalize()

        # Add word to text
        text += formatted_number + end_chars + " "

        return text

    @staticmethod
    def insert_comma_seperators(number:str):
        """
        Inserts comma separators into numbers with many digits to break up 1000s (e.g. '100000' -> '100,000').
        """
        # Skip if contains pre-existing punctuation
        if number.count(',') > 0:
            return number + " "

        # Strip leading non-numeric characters and trailing digits after the decimal point in floats
        if not number[0].isalnum():
            start_char = number[0]
            number = number[1:]
        else:
            start_char = ""

        if number.count('.') > 0:
            dot_idx = number.index(".")
            end_chars = number[dot_idx:]
            number = number[:dot_idx]

        elif number[-1] == 'm':
            number, end_chars = number[:-1], 'm'
        elif number[-2:] == 'bn' or number[-2:] == 'bn':
            number, end_chars = number[:-2], number[-2:]
        else:
            end_chars = ""

        # Cycle through number in reverse order and insert comma separators every three digits
        for i in range(len(number) - 3, 0, -3):
            number = number[:i] + ',' + number[i:]

        # Reconcatenate leading/trailing chars/digits
        number = start_char + number + end_chars + " "

        return number

    @staticmethod
    def decades_to_digits(text, decade):
        text = text.strip()
        output_text_list = text.split()
        output = " ".join(output_text_list[:-1]) + " " + output_text_list[-1][:2] + DECADES[decade] + " "

        return output

    def recover_ordinals(self, plain, recovered):
        # Align number recovered text with original s.t. we can find where ordinals have been lost
        plain = plain.split()
        recovered = recovered.split()

        mapping = align_texts(plain, recovered, strip_punct=False)
        formatted_output = ""

        for plain_words, rec_word in mapping:
            ordinal_candidate_word = re.sub(r"[^0-9a-zA-Z]", "", plain_words[-1].strip())

            if self._is_ordinal(ordinal_candidate_word):
                # print(f" * ORDINAL FOUND: ({plain_words}, {rec_word})")
                # print(mapping)
                ordinal_word = self.format_ordinal(rec_word[0])
                formatted_output += ordinal_word + " "
            else:
                formatted_output += rec_word[0] + " "

        return formatted_output

    @staticmethod
    def format_ordinal(input_number):
        number = re.sub(r"[^0-9a-zA-Z]", "", input_number).strip()

        if number.isnumeric():
            number = int(number)
        else:
            number = individual_number_parser(number)

            if number is None:
                return input_number

        if number < 10:
            ordinal_number = num2words(number, to='ordinal')
        else:
            ordinal_function = lambda n: "%s%s" % ("{:,}".format(n),"tsnrhtdd"[(n//10%10 != 1) * (n%10 < 4) * n%10::4])
            ordinal_number = ordinal_function(number)

        return ordinal_number
