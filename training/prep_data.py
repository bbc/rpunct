# -*- coding: utf-8 -*-
# 💾⚙️🔮
__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import os
import re
import math
import json
import random
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from training.get_data import *
from rpunct.punctuate import VALID_LABELS, PUNCT_LABELS

WORDS_PER_FILE = 35000000


def e2e_data(data_type='news-transcripts', tt_split='90:10',
            start_year='2014', end_year='2022', summaries=False,
            composite_datasets_list=None, dataset_balance='o',
            validation=False, dataset_stats=False, form_mixed_case_database=False):
    """
    Full pipeline for gathering/loading data and formatting it into training and testing datasets to be used by RPunct.

    Args:
        - data_type (str): a name tag specifying source of the data to be included in the datasets.
        - start_year (str): start year of date range to gather BBC News articles from (requires `data_type = news-articles`).
        - end_year (str): end year of date range to gather BBC News articles from (requires `data_type = news-articles`).
        - summaries (str): toggle between extracting article summaries and article bodies (requires `data_type = news-articles`).
        - tt_split (str): the proportion of collected data to use for training the RPunct model (ratio of percentages in the form `TRAIN:TEST`).
        - composite_datasets_list (list): collection of name tags of multiple data sources to include when constructing a composite dataset (requires `data_type = composite`).
        - dataset_balance (str): specifier to clip dataset sizes to be included in composite dataset in a given ratio to each other (in form `X:Y:...:Z`) (requires `data_type = composite`).
    """
    # Collect raw data from input source
    if data_type == 'news-articles':
        # Gather articles data from each JSONL file enumerating all BBC News articles for each year 2014-2022
        print(f"\n> Preparing data from source: BBC News articles")
        data_source = 'news'
        data_type = 'news-' + start_year + '-' + end_year  # data name tag
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100  # training split

        if summaries:
            dataset_path = os.path.join(PATH, f'news-summaries')
            article_size = 'summary'
        else:
            dataset_path = os.path.join(PATH, f'news-{start_year}-{end_year}')
            article_size = 'body'

        collate_news_articles(int(start_year), int(end_year), summary_or_body=article_size, train_split=split, output_directory=dataset_path)

    elif data_type == 'news-transcripts':
        # Gather transcripts data from JSON files
        print(f"\n> Preparing data from source: BBC News transcripts")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100
        data_source = 'transcripts'

        dataset_path = os.path.join(PATH, f'news-transcripts')
        collate_news_transcripts(train_split=split, output_directory=dataset_path)

    elif data_type == 'subtitles':
        # Gather subtitles data from JSON files
        print(f"\n> Preparing data from source: subtitles (all genres)")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100
        data_source = data_type

        dataset_path = os.path.join(PATH, f'subtitles')
        collate_subtitles(train_split=split, output_directory=dataset_path)

    elif data_type == 'composite':
        # Create composte dataset from multiple sources
        print(f"\n> Assembling composite dataset containing: {composite_datasets_list}")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100
        data_source = data_type

        dataset_path = os.path.join(PATH, 'composite')
        create_composite_dataset(dataset_names=composite_datasets_list, train_split=split, balance=dataset_balance, output_directory=dataset_path)

    elif data_type == 'reviews':
        # Gather reviews data from tensorflow to local csv files
        print("\n> Preparing data from source: Yelp reviews")
        dataset_path = os.path.join(PATH, 'reviews')
        download_reviews(output_directory=dataset_path)
        data_source = data_type
        split = 0.9

    elif data_type == 'ted-talks':
        # Gather subtitles data from JSON files
        print(f"\n> Preparing data from source: Ted Talks transcripts")
        tt_split = tt_split.split(':')
        split = int(tt_split[0]) / 100
        data_source = data_type.replace("-", "_")

        dataset_path = os.path.join(PATH, f'ted-talks')
        collate_ted_talks(train_split=split, output_directory=dataset_path)

    else:
        raise ValueError(f"Unrecognised data source: {data_type}")

    # Don't create test dataset if using 100% of data for training
    if split == 1:
        stages = ['train']
    else:
        stages = ['train', 'test']

    # Create dataset for each stage (train/test)
    for stage in stages:
        # Constuct list of text and labels (punctuation tag per word)
        form_mixed_case_database = form_mixed_case_database and stage == 'train'
        rpunct_dataset = create_rpunct_dataset(dataset_path, data_source, train_or_test=stage, make_mc_database=form_mixed_case_database)

        create_training_samples(rpunct_dataset, data_source, file_out_path=dataset_path, train_or_test=stage)
        del rpunct_dataset

        # generate correctly formatted training data
        prepare_data(source=data_type, train_or_test=stage, validation=validation, print_stats=dataset_stats)

    # remove temporary dataset files
    remove_temp_files(dataset_path, extensions=['csv', 'npy'])

    print("\n> Data generation and preparation complete")


def create_rpunct_dataset(directory, data_type, train_or_test='train', make_mc_database=True):
    """
    Converts CSV files of punctuated text data into plaintext labelled word-level training/testing data (list of 'records')
    for supervised learning of punctuation restoration.

    Args:
        - directory (Path): location of directory holding collected data files (train/test CSVs).
        - data_type (str): a name tag specifying source type of the data to be included in the dataset.
        - train_or_test (str): specifier of whether to construct a training or testing dataset.
        - make_mc_database (bool): toggle to use newly labelled data to regrenerate the mixed-case database of plaintext-cased pairs.

    Returns:
        - all_records (lst): a 1D list of 'records' enumerating each word in the training/testing dataset and its punctuation label.
    """
    print(f"\n> Generating RPunct dataset ({train_or_test.upper()}):")

    # Load in train/test data from local file
    data_split_path = os.path.join(directory, f'{train_or_test}_{data_type}.csv')
    data_split = pd.read_csv(data_split_path)
    data_split.dropna(inplace=True)
    data_split.reset_index(drop=True, inplace=True)

    # Change inter-word dashes to commas (not supported)
    data_split['text'] = data_split['text'].str.replace(" -", ",")

    # Insert a space after intra-word hyphens (as part of pipeline for hyphenation reconstruction)
    data_split['text'] = data_split['text'].str.replace("-", "- ")

    # Toggle creation of mixed-case database (only allow if constructing a training dataset)
    make_mc_database = make_mc_database and train_or_test == 'train'
    mixed_case_database = {}

    # Constuct list of text words and labels (punctuation tag per word) over the dataset
    all_records = []
    with tqdm(data_split['text']) as T:
        T.set_description(f"{' ' * 7} * Labelling {train_or_test}ing instances {' ' if train_or_test == 'test' else ''}")
        for segment in T:
            # Create a list enumerating each word in an input segment of text and its label: [...{id, word, label}...]
            record, mixed_case_instances = create_record(segment, mixed_casing=make_mc_database)

            if make_mc_database:
                mixed_case_database.update(mixed_case_instances)

            all_records.extend(record)

    if make_mc_database:
        with open('rpunct/mixed-casing.json', 'w') as f:
            json.dump(mixed_case_database, f)

    return all_records


def create_record(text, mixed_casing=False):
    """
    Remove punctuation and create data labels for Punctuation Restoration task for each word in a given speaker segment / paragraph of text.

    Args:
        - text (str): a single segment of text from the input CSV data file (the text we want to label).
        - mixed_casing (bool): toggle to add words labelled mixed-case to a new mixed-case database of plaintext-cased pairs.

    Returns:
        - new_obs (lst): a collection of dictionaries specifying each word-label pair to be added to the dataset (dict holds the index, plaintext, and punctuation label)
        - mixed_case (dict): output collection of plaintext-cased pairs relating to identified mixed-case words to be added to the mixed-case database (empty if `mixed_casing = False`)
    """
    new_obs = []
    mixed_case = {}

    # Convert string of text (from segment) into a list of words
    observation = text.replace('\n', ' ').split()

    # Remove punctuation of each word, and label it with a tag representing what punctuation it did have
    for obs in observation:
        # Convert word to plaintext
        stripped_obs = re.sub(r"[^0-9a-zA-Z]", "", obs)

        if not stripped_obs:
            continue

        # Collect trailing punctuation for this word's label
        if not obs[-1].isalnum() and obs[-1] in PUNCT_LABELS:
            new_lab = obs[-1]
            unpunct_obs = obs[:-1]
        else:
            new_lab = "O"  # `O` => no punctuation
            unpunct_obs = obs

        # Add a capitalisation descriptor to the label
        if stripped_obs.islower() or stripped_obs.isnumeric():
            new_lab += "O"  # `xO` => lowercase (incl. numbers)
        elif stripped_obs.isupper():
            new_lab += "U"  # `xU` => uppercase
        elif stripped_obs[0].isupper() and (len(stripped_obs) == 1 or stripped_obs[1:].islower()):
            new_lab += "C"  # `xC` => capitalised
        else:
            new_lab += "M"  # `xM` => mixed-case

            # Populate database of mixed-case instances
            if mixed_casing:
                if unpunct_obs.endswith("'s"):
                    unpunct_key = unpunct_obs[:-2]
                else:
                    unpunct_key = unpunct_obs

                if unpunct_key.lower() not in mixed_case.keys():
                    mixed_case.update({unpunct_key.lower(): unpunct_key})

        # Add the word and its label to the dataset
        # N.B. some punctuation is retained in label word (any thats restored by `number_recoverer` not the model: e.g. %, £, etc.)
        new_obs.append({'sentence_id': 0, 'words': unpunct_obs, 'labels': new_lab})

    return new_obs, mixed_case


def create_training_samples(words_and_labels, data_type, file_out_path=PATH, train_or_test='train', size=WORDS_PER_FILE):
    """
    Splits up a list of words and their labels (that will form the training/testing dataset) into chunks of 500 instances (observations)
    that can be input into the BERT-based transformer model.
    Then output this segmented dataset of observations to multiple NPY files of a fixed size each holding part of the training/testing dataset.

    Args:
        - words_and_labels (lst): a 1D list of 'records' enumerating each word in the training/testing dataset and its punctuation label.
        - data_type (str): a name tag specifying source type of the data to be included in the dataset.
        - file_out_path (Path): location of directory in which to store the output NPY data files.
        - train_or_test (str): specifier of whether to sace data as a training or testing dataset.
        - size (int): number of observations to save to a single output NPY file.
    """
    # Determine number of output files dependent on size of dataset
    num_words = len(words_and_labels)
    num_splits = math.ceil(num_words / size)
    random.seed(42)
    _round = 0

    print(f"\t* Segmenting data into chunks  :")
    print(f"\t\t- No. words in {train_or_test} set {' ' if train_or_test == 'test' else ''}: {num_words}")

    # Segment dataset into `num_splits` chunks
    while _round < num_splits:
        # Locate the `_round`th chunk of word/label dicts (will be a single file)
        records = words_and_labels[size * _round: size * (_round + 1)]

        # Break down this file's chunk of dicts into smaller chunks of 500 words (`splits` = start/end indices of observation chunks)
        splits = create_tokenized_obs(records)
        records = pd.DataFrame(records)

        # Cycle through the start/end chunk index tuples forming data blocks of uniform dimensions
        observations = np.empty(shape=(len(splits), 500, 3), dtype=object)

        with tqdm(range(len((splits)))) as S:
            S.set_description(f"{' ' * 15} - Splitting data chunk {_round + 1} ")
            for j in S:
                a, b = splits[j][0], splits[j][1]
                data_slice = records.iloc[a: b, ].values.tolist()  # collect the 500 word-label dicts between the specified indices
                data_slice = np.pad(data_slice, [(0, 500 - len(data_slice)), (0, 0)], 'empty')

                observations[j] = data_slice  # add each list of 500 dicts to the dataset

        # Shuffle dataset of 500 word-label dicts
        _round += 1
        random.shuffle(observations)

        # Save this split of dataset to a distinct file
        out_file = f"{data_type}_{train_or_test}_{_round}.npy"
        out_path = os.path.join(file_out_path, out_file)

        with open(out_path, 'wb') as f:
            np.save(f, observations, allow_pickle=True)

        print(f"\t\t- Outputting data file {_round} : {out_path}")
        del records
        del observations


def create_tokenized_obs(input_list, num_toks=500, offset=250):
    """
    Generate the start/end indices of the observations (of `num_toks` instances) segmented from a given set of word-label instances,
    with an offset (sliding window) of `offset` instances.
    It is important that first token is capitalized and we use as many tokens as possible.

    Args:
        - input_list (lst): collection of word-label pairs to segment into observations.
        - num_toks (int): number of instances to include in each observation (chunk).
        - offset (int): offset from the begining of the previous chunk to start collecting data for the next chunk (gives an overlap of `num_toks -  offset` instances).

    Returns:
        - indices (lst): collection of index pairs specifying the start and end index of each observation segmented from the input data.
    """
    start = -1
    loop_end = -1
    indices = []

    # Cycle through each list of dicts (global index, {word, label}) to break into chunks with a start index `start` and end index `end`
    for ix, i in enumerate(input_list):
        # Skip over the first 250 words that have been added to a chunk already (progress to end of offset)
        if ix == loop_end:
            start = -1

        # Check if the current word is uppercase (so we can start all observations at the beginning of a sentence)
        if i['labels'][-1] == "C" and start == -1:
            start = ix  # start chunk from this uppercase word (i.e. start of a sentence)
            end = ix + num_toks  # the end of that chunk is 500 words later
            indices.append((start, end))  # enumerate the start and end of this chunk
            loop_end = start + offset  # identify end of offset s.t. loop can progress to the end of this offset and start searching again

    return indices


def prepare_data(source='news-transcripts', train_or_test='train', validation=False, print_stats=False):
    """
    Prepares data from original text into Connnl formatted datasets ready for training.
    In addition constraints label space to only labels we care about.
    """
    print("\t* Finalising output dataset    :")

    # Create output dataset paths
    output_txt = f'rpunct_{train_or_test}_set.txt'
    dataset_path = os.path.join(PATH, source, output_txt)

    val_output_txt='rpunct_val_set.txt'
    val_set_path = os.path.join(PATH, source, val_output_txt)

    # Load formatted/labelled data from file chunks into single structure
    token_data = load_datasets(source, train_or_test)

    # Remove any invalid labels
    clean_up_labels(token_data)

    # Split train/validation datasets, convert each to a text file, and print dataset stats
    if validation:
        # Training & validation sets
        split = math.ceil(0.9 * len(token_data))
        train_set = token_data[:split]
        val_set = token_data[split:]

        # Format validaton set as Connl NER txt file
        print(f"\t\t- Validation dataset shape: ({len(val_set)}, {len(val_set[0])}, {len(val_set[0][0])})")
        create_text_file(val_set, val_set_path)
    else:
        train_set = token_data

    print(f"\t\t- {train_or_test.capitalize()}ing dataset shape {' ' * 7 if train_or_test == 'train' else ' ' * 8} : ({len(train_set)}, {len(train_set[0])}, {len(train_set[0][0])})")

    # Format training set as Connl NER txt file
    create_text_file(train_set, dataset_path)

    # Remove temporary dataset files
    dataset_path = os.path.join(PATH, source)
    remove_temp_files(dataset_path, extensions=['npy', 'csv'], traintest='train')

    # Dataset statistics of label distribution
    if print_stats:
        train_stats = get_label_stats(train_set)
        print(f"\t\t- {train_or_test.capitalize()}ing data statistics:", end='\n\n')
        print(train_stats)

        if validation:
            val_stats = get_label_stats(val_set)
            print(f"\t\t- Validation data statistics:")
            print(val_stats)


def load_datasets(data_dir='composite', train_or_test='train'):
    """
    First, locate the data files that were split into chunks.
    Then, given this list of data paths return a single data object containing all data slices.
    """
    # convert from dir name to file name
    if data_dir.startswith('news-2'):
        data_type = 'news'
    elif data_dir.startswith('news-tran'):
        data_type = 'transcripts'
    elif data_dir.startswith('comp'):
        data_type = 'composite'
    elif data_dir.startswith('ted'):
        data_type = 'ted_talks'
    else:
        data_type = data_dir

    # find training data files
    path_to_data = pathlib.Path(os.path.join(PATH, data_dir))
    data_file_pattern = f'{data_type}_{train_or_test}_*.npy'
    datasets = list(path_to_data.glob(data_file_pattern))

    if len(datasets) == 0:
        raise FileNotFoundError(f"No dataset files found at location: '{os.path.join(path_to_data, data_file_pattern)}'. You may have forgotten to run the `prep_data.py` preparation process on the dataset you want to use.")

    # collate these into a single data object
    token_data = np.empty(shape=(0, 500, 3), dtype=object)

    with tqdm(datasets) as D:
        for dataset in D:
            D.set_description(f"{' ' * 15} - Combining segmented data files ")
            with open(dataset, 'rb') as f:
                data_slice = np.load(f, allow_pickle=True)

            token_data = np.append(token_data, data_slice, axis=0)
            del data_slice

    return token_data


def clean_up_labels(dataset):
    """
    Given a list of valid labels cleans up the dataset
    by limiting to only the labels available.

    In addition prepares observations for training.
    """
    for ix, i in enumerate(dataset):
        for tok in i:
            tok[0] = ix
            if tok[2] not in VALID_LABELS:
                case = tok[2][-1]
                tok[2] = f"O{case}"
                if len(tok[2]) < 2:
                    tok[2] = "OO"


def create_text_file(dataset, name):
    """
    Create Connl NER format file
    """
    with open(name, 'w') as fp:
        with tqdm(dataset) as D:
            for obs in D:
                D.set_description(f"{' ' * 15} - Outputting dataset to TXT file ")
                for tok in obs:
                    line = tok[1] + " " + tok[2] + '\n'
                    fp.write(line)
                fp.write('\n')


def get_label_stats(dataset):
    """
    Generates frequency of different labels in the dataset.
    """
    calcs = {}
    total_items = 0
    for i in dataset:
        for tok in i:
            total_items += 1
            if tok[2] not in calcs.keys():
                calcs[tok[2]] = 1
            else:
                calcs[tok[2]] += 1

    distribution = {
        'capi': 0,
        'lower': 0,
        'mixed': 0,
        'upper': 0,
        'unpunct': 0,
        '-': 0,
        '.': 0,
        ',': 0,
        ':': 0,
        ';': 0,
        '!': 0,
        '?': 0,
        "'": 0,
    }

    for label in calcs:
        label_count = calcs[label]

        if label[0] == 'O':
            distribution['unpunct'] += label_count
        else:
            distribution[label[0]] += label_count

        if label[-1] == 'C':
            distribution['capi'] += label_count
        elif label[-1] == 'U':
            distribution['upper'] += label_count
        elif label[-1] == 'M':
            distribution['mixed'] += label_count
        else:
            distribution['lower'] += label_count

    for capi_punct_status in distribution.keys():
        distribution[capi_punct_status] = (distribution[capi_punct_status] / total_items) * 100

    output = pd.DataFrame.from_dict(distribution, orient='index', columns=['Proportion (%)'])

    return output


if __name__ == "__main__":
    # specify dataset to curate
    data = 'news-transcripts'

    # run data preparation pipeline
    e2e_data(data)
