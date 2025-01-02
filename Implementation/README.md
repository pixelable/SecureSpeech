# SecureSpeech
This folder contains 2 separate tasks:
1. Replacement of Sensitive Words (NER) from Transcripts ([NER Replacement Usage](#replacement))
2. Generation of Synthetic Audio Clips ([Generation Usage](#generation))

## Table of Contents
- [Installation](#installation)
- [NER Replacement Usage](#replacement)
- [Generation Usage](#generation)

## Installation
1. Clone the repository:
```bash
 git clone https://github.com/yourusername/SecureSpeech.git
```

2. Get Voxpopuli Data:
Install SLUE-VoxPopuli test dataset
```bash
git lfs install
git clone https://huggingface.co/datasets/asapp/slue

```
Otherwise if using own data: 
Add unique id, transcripts and NER instructions in Ori_Data, in .json as in the format below:
[{
    "id": "20171211-0900-PLENARY-4-en_20171211-17:01:30_3", 
    "Transcript": "only his own group and the liberals support the declaration but six groups have refused it.",
    "NER Instructions": "{'type': array(['QUANT', 'NORP'], dtype=object), 'start': array([64, 27], dtype=int32), 'length': array([3, 8], dtype=int32)}"
}]

3. Edit attributes lists as you fit within the attributes folder

4. Create a virtualenv then:
```bash
 pip install -r requirements.txt
```
Add your audio files in Original
Make changes to the run.sh file according to your requirements:

```bash
bash Implementation/run.sh
```
run.sh runs the following files:

## NER Replacement Usage
This package is meant for using GPT to replace NER words according to instructions to increase privacy around transcripts used. The model used is: gpt-4o-mini. This programme is able to extract from .parquet files given that the column headers are "normalized_text" and "normalized_combined_ner" as modelled after SLUE-VoxPopuli transcripts..
1. Insert OpenAI key 
```bash
    client = OpenAI(api_key="Insert OpenAI key")
```

if use own .json file, change in run.sh to: 
```bash
 python replace_ner_json.py
```

2. Enter .parquet file for analysis in run.sh:
E.g. Enter parquet file name: voxpopuli/voxpopuli/test-00000-of-00001.parquet

Output is saved in ner_output.json.

## Generation Usage
These packages generate synthetic audio files reading random transcripts from LibriSpeech ASR corpus. A random speaker created from a random speaker description will voice the replaced_ner output transcripts given.

1. Run for specific speaker generation: 
```bash
 mkdir -p Specific_Speaker_Output

python3 replace_ner.py "voxpopuli/voxpopuli/test-00000-of-00001.parquet" \
	"Specific_Speaker_Output/ner_output.json" \

python3 generate_specific_speaker.py "Specific_Speaker_Output/ner_output.json" \
	"Specific_Speaker_Output" \
	"Specific_Speaker_Output/full_generated_meta_info_random_speaker.json" \
```
Output files are saved in specific_speaker_output:
- Sound files is saved in .wav format 
- Corresponding Transcript saved in .txt format

2. Run for random speaker generation:
```bash
mkdir -p Random_Speaker_Output

python3 replace_ner.py "voxpopuli/voxpopuli/test-00000-of-00001.parquet" \
	"Random_Speaker_Output/ner_output.json" \

python3 generate_random_speaker.py "Random_Speaker_Output/ner_output.json" \
	"Random_Speaker_Output" \
	"Random_Speaker_Output/full_generated_meta_info_random_speaker.json" \
```
Output files are saved in random_speaker_output:
- Sound files is saved in .wav format 
- Corresponding Transcript saved in .txt format


## Evaluation
Instructions can be separately found in eval folder. 
