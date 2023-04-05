import os
import json
import math
import random
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm

PATH = './tests/test-other-punctuators/ted-talks-en-de/'


def collate_ted_talks(train_split=0.9, output_directory=PATH):
    """
    Loads in JSON data from BBC subtitles (stored within the source directories `source_subtitles_news` & `source_subtitles_other`) and combine into a single dataset.
    Then splits data into training and testing datasets and save these as individual CSV files.

    Args:
        - train_split (float): the proportion of subtitle transcripts to use as training data (decimal between 0-1).
        - output_directory (Path): location of output directory to store output CSV files (relative to top-level `rpunct` directory).
    """
    print(f"\n> Assembling subtitles data:")

    # Remove pre-existing data files from previous iterations
    # remove_temp_files(output_directory, extensions=['npy', 'txt'])

    # Input subtitle transcripts from JSON files
    talks_path = os.path.join(PATH)
    talks_datasets = [os.path.join(talks_path, file) for file in os.listdir(talks_path) if file.endswith('.json')]
    transcripts = np.empty(shape=(0), dtype=object)

    print(f"\t* Data split {' ' * 22} : {train_split:.2f}:{1 - train_split:.2f}")

    with tqdm(talks_datasets) as D:
        D.set_description(f"{' ' * 7} * Extracting ted talks data {' ' * 8}")
        for json_path in D:
            with open(json_path, 'r') as f:
                global_obj = json.load(f)

            for local_obj in global_obj["mteval"]["srcset"]["doc"]:
                segmented_text = local_obj["seg"]
                transcripts = np.concatenate((transcripts, segmented_text))

            del global_obj

    # Train-test split
    random.seed(42)
    random.shuffle(transcripts)
    split = math.ceil(train_split * len(transcripts))
    train = transcripts[:split]
    test = transcripts[split:]
    del transcripts

    print(f"\t* Subtitle transcripts in train set : {len(train)}")
    print(f"\t* Subtitle transcripts in test set  : {len(test)}")

    # Save train/test data to csv
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(train, columns=['text'])
    train['text'] = train['text'].str.replace('\n', ' ')
    csv_path_train = os.path.join(output_directory, 'train_ted_talks.csv')
    train.to_csv(csv_path_train, index=False)
    del train

    test = pd.DataFrame(test, columns=['text'])
    test['text'] = test['text'].str.replace('\n',' ')
    csv_path_test = os.path.join(output_directory, 'test_ted_talks.csv')
    test.to_csv(csv_path_test, index=False)
    del test


collate_ted_talks()
