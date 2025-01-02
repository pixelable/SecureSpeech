import random
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import json
import sys
import numpy as np

#transcript and og file id
with open(sys.argv[1], 'r') as file:
    transcript_data = json.load(file)

#speaker
my_file = open("voxpopuli_speaker.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
vox_speakers = data.split("\n") 
my_file.close()

#speaker
my_file = open("attributes/speakers.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
speakers = data.split("\n") 
my_file.close()

#pitch
my_file = open("attributes/pitch.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
pitch = data.split("\n") 
my_file.close() 

#modulation
my_file = open("attributes/modulation.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
modulation = data.split("\n") 
my_file.close() 

#rate
my_file = open("attributes/rate.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
rate = data.split("\n") 
my_file.close() 

#channel
my_file = open("attributes/channel conditions.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
channel = data.split("\n") 
my_file.close() 

#distance
my_file = open("attributes/distance.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
distance = data.split("\n") 
my_file.close() 

#recording
my_file = open("attributes/recording.txt", "r") 
data = my_file.read() 
data = data.replace('â€‹', '')
recording = data.split("\n") 
my_file.close() 

"""
generate_random_env(channel, distance, recording)
Input: channel, distance, recording
Function: Due to many possible combinations of random additional attributes, this function randomly creates a combination for use on speaker description
Output: A complete sentence describing the channel, distance and recording conditions, or none if not selected.
"""

def generate_random_env(channel, distance, recording):
    if channel!="" and distance !="" and recording !="":
        random_val = random.choice([1,0])
        if random_val == 0:
            recording = ""
        else:
            distance = ""

    if channel=="" and distance =="" and recording =="":
        environment = ""
    elif channel=="":
        if distance == "":
            environment = f"The {recording}."
        else:
            environment = f"The speaker's voice is {channel}, and the {recording}."
    elif recording == "":
        if distance == "":
            environment = f"The speaker's voice is {channel}."
        else:
            environment = f"The speaker's voice is {distance}, and {channel}."
    else:
        environment = f"The speaker's voice is {distance}, and the {recording} and {channel}."

    return environment

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

meta_data = []
for transcripts in transcript_data:
    meta = {}
    random_speaker = random.choice(vox_speakers)
    speak = random.choice(speakers)
    random_pitch = random.choice(pitch)
    random_channel = random.choice(channel)
    random_distance = random.choice(distance)
    random_recording = random.choice(recording)
    random_environment = generate_random_env(random_channel, random_distance, random_recording)
    random_modulation = random.choice(modulation)
    random_rate = random.choice(rate)
    prompt = transcripts.get("Replaced Sentence")
    identity = transcripts.get("id")

    description = f"{speak} reads a book with a {random_pitch} {random_modulation} voice {random_rate}. {random_environment}"

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    
    title = f"{speak}-{random_rate}-{random_pitch}-{random_modulation}-{random_distance}-{random_channel}-{random_recording}"
    meta["id"] = identity
    meta["Description Summary"] = title
    meta["TTS Description"] = description
    meta["TTS Prompt"] = prompt
    meta["Speaker"] = random_speaker

    if audio_arr is not None and audio_arr.size > 0:
        if len(audio_arr.shape) == 1:  # Convert mono to 2D array
             audio_arr = audio_arr.reshape(-1, 1)
        sf.write(sys.argv[2] + "/" + str(identity).replace(":", "") +".wav", audio_arr, model.config.sampling_rate)
        f = open(sys.argv[2] + "/"+ str(identity).replace(":", "") + ".txt", "a")
        f.write(prompt)
        f.close()
    else:
        print("Error: Generated audio array is empty or invalid for " + identity)
        
    meta_data.append(meta)

with open(sys.argv[3], 'w') as f:
    json.dump(meta_data, f, indent=4)