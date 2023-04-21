# bbc-punctuator

This is the BBC Punctuator, developed by BBC R&D's Speech-to-Text team. BBC Punctuator is a machine learning model for restoring punctuation and capitalisation into plaintext, such as raw transcripts produced by STT systems.

The model uses HuggingFace's `bert-base-uncased` model weights that have been fine-tuned for Punctuation restoration for one epoch over a new BBC built composite punctuator training dataset, composed of BBC News articles (written between 2020-2), BBC News transcripts, and subtitles from a range of BBC TV programmes [*model=composite20v3-1e*]. The parameters learned over this training process are stored in bbc-data (*bbc/speech/transcription/bbc-punctuator:0.0.1*), which when referenced from this code make a complete system for punctuating text.

The system utilises the CPU by default but can be instructed to run on a GPU.

List of text featured available for recovery:

* Capitalisation:
  * Capitalisaton
  * Acronyms (upper-casing)
  * Mixed-casing
* Punctuation:
  * Period: **.**
  * Comma:  **,**
  * Question Mark: **?**
  * Exclamation: **!**
  * Colon:  **:**
  * Semi-colon: **;**
  * Apostrophe: **'**
  * Hypen: **-**
* Numbers:
  * Digitising natural language numbers
  * Currencies: **£**, **$**, **€**, **¥**
  * Percentages: **%**
  * Ordinals
  * Decimals
  * Dates (years/decades)

## Usage:

1. Pull the model files from bbc-data: \
   `bbc-data pull bbc/rpunct:0.0.1`
2. Install the BBC Punctuator system from Git: \
   `git clone git@github.com:bbc/rpunct.git` \
   `cd rpunct` \
   `pip install -r requirements.txt`
3. Move the model files to the reference location within bbc-punctuator: \
   `mv $(bbc-data path bbc/rpunct) model-files`
4. Run the BBC Punctuator by pointing the run file to the model files (from inside the cloned repo):

   1. To punctuate a plaintext (.txt) file from the command line: \
      `python run.py rpunct -m model-files -i input_file_path.txt -o output_file_path.txt`
   2. To use bbc-punctuator within a python script (from within the `rpunct`
      directory):

      ```
      from rpunct.rpunct_recoverer import RPunctRecoverer

      punctuator = RPunctRecoverer()
      text = "hello and welcome to the bbc punctuator \
      this is an example piece of plaintext \
      that exhibits no punctuation or capitalisation"

      punctuated_text = punctuator.process(text)
      ```

## Performance

| Metric         | Prec   | Recall | F1     |
| -------------- | ------ | ------ | ------ |
| abbreviations  | 0.9377 | 0.8605 | 0.8974 |
| capitalisation | 0.9213 | 0.9434 | 0.9322 |
| commas         | 0.6977 | 0.7029 | 0.7003 |
| hyphens        | 0.7198 | 0.8953 | 0.7980 |
| numbers        | 0.7499 | 0.7747 | 0.7621 |
| sentences      | 0.8351 | 0.8762 | 0.8552 |
| fullstops      | 0.8110 | 0.8710 | 0.8399 |
| questions      | 0.7537 | 0.4770 | 0.5843 |

## Authors

* **Tom Potter** <tom.potter@bbc.co.uk>
* **Daulet Nurmanbetov** <daulet.nurmanbetov@gmail.com> (developer of the original
  system [rpunct](https://github.com/Felflare/rpunct) upon which bbc-punctuator is
  based)
