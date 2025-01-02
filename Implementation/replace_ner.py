from openai import OpenAI
import pandas as pd
import json
import tiktoken
import sys
import string

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# client = OpenAI(api_key="API KEY HERE")
GPT_MODEL = "gpt-4o-mini"

data = pd.read_parquet(sys.argv[1], engine='pyarrow')
ids = data["id"].tolist()
normalized_text = data["normalized_text"].tolist()
normalized_combined_ner = data["normalized_combined_ner"].tolist()

data_2 = pd.read_parquet('voxpopuli/voxpopuli/train-00000-of-00001.parquet', engine='pyarrow')
ids_2 = data_2["id"]
normalized_text_2 = data_2["normalized_text"]
normalized_combined_ner_2 = data_2["normalized_combined_ner"]

for i in ids_2:
    ids.append(i)

for i in normalized_text_2:
    normalized_text.append(i)

for i in normalized_combined_ner_2:
    normalized_combined_ner.append(i)

text_no_punctuations = [item.translate(str.maketrans('', '', string.punctuation)) for item in normalized_text]

function_metrics_gpt = [
{
    "name": "PresentNERReplacement",
    "description": "Evaluate the following details for sentence provided by user",
    "parameters": 
    {
        "type": "object",
        "properties": 
        {
            "Sentence": 
            {
                "type": "array",
                "description": "description of sentence in array",
                "items": 
                {
                    "type": "object",
                    "properties": 
                    {
                        "Original Sentence": 
                        {
                            "type": "string",
                            "description": "sentence provided by user"
                        },
                        "NER Instructions": 
                        {
                            "type": "string",
                            "description": "insert NER instructions given by user, do not make changes from user's instructions."
                        },
                        "Replaced Sentence": 
                        {
                            "type": "string",
                            "description": "Final sentence after all ner words have been replaced"
                        },
                    }
                }
            }
        }
    }
}]

"""
replace_ner_gpt
Input: Query(str), Model(str) - preset as gpt-4o mini
Function: calls the gpt-4o mini model and provides user prompt (sentence to replace and ner instructions), system prompt (), PresentNERReplacement function to standardise JSON output. The model will then generate the sentence that has the instructed ner replaced with dissimilar words.
Output: JSON output that contains Original Sentence, NER Instructions, Replaced Sentence.
"""

def replace_ner_gpt(query: str, model: str = GPT_MODEL,) -> str:
    messages = [
        {"role": "system", "content": '''You are a Named Entity Recognition (NER) expert. If an NER instruction is given, you will replace the sensitive NER words in each of the sentences provided by user with a completely dissimilar replacement based on the ner instructions given. Output the original sentence, ner instructions and the replaced sentence in json format exactly like the following examples in delimitered by XML tags:
        <output>
         Example 1:
        {
            "Original Sentence": "secondly conclusions from the commission communication entitled towards world class clusters in the european union implementing the broad based innovation strategy",
            "NER Instructions": "Replace the 'PLACE' type word, starting from character 100 to character 113, with a dissimilar word. Replace the 'QUANT' type word, starting from character 0 to character 7, with a dissimilar word.",
            "Replaced Sentence": "Next, conclusions from the commission communication entitled towards world-class clusters across the continent implementing the comprehensive innovation strategy." ,
        }
         
        Example 2:
        {
            "Original Sentence": "along with eighty four other meps i have tabled a plenary amendment which i really hope you can all support.",
            "NER Instructions": "Replace the 'QUANT' type word, starting from character 11 to character 21, with a dissimilar word.",
            "Replaced Sentence": "Along with a substantial group of other MEPs, I have tabled a plenary amendment which I really hope you can all support.",
        }
        
         Example 3:
        {
            "Original Sentence": "in this situation we have to see reflection political responsibility and more political dialogue.",
            "NER Instructions": "",
            "Replaced Sentence": "in this situation we have to see reflection political responsibility and more political dialogue.",
        }
        
        Derive the answer step by step:
        1. Obtain the sentence from user
        2. Replace each word in the original sentence, as specified by the users NER instructions, with a dissimilar word. 
        3. Input original sentence, NER instructions and replaced sentence into the json format required. '''},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages = messages,
        response_format = {"type": "json_object"},
        functions = function_metrics_gpt,
        function_call = {"name": "PresentNERReplacement"},
        temperature=0
    )
    
    argument = response.choices[0].message.function_call.arguments
    json_obj = json.loads(argument)
    return json_obj["Sentence"]

replaced_output = []
token_budget = 10000
curr_token_len = 0
multiple_sentence = ""
for idx, each_text in enumerate(normalized_text):
    ner_command = str(normalized_combined_ner[idx])
    ner_type_list = normalized_combined_ner[idx].get("type")
    ner_start_list = normalized_combined_ner[idx].get("start")
    ner_length_list = normalized_combined_ner[idx].get("length")
    replace_command = f"Give me a replaced sentence based on the Original Sentence: '{each_text}', and NER Instructions:"
    if len(ner_type_list) != 0:
        for itr, each_replace in enumerate(ner_type_list):
            start = ner_start_list[itr]
            end = ner_start_list[itr] + ner_length_list[itr] - 1
            replace_command += f"'Replace the {each_replace} ner type word, starting from character {start} to character {end}, with a dissimilar word."
        tokens = encoding.encode(replace_command)
        token_count = len(tokens)
        if (curr_token_len + token_count) < token_budget and idx != (len(normalized_text)-1):
            multiple_sentence += replace_command
            curr_token_len += token_count
        else:
            output =replace_ner_gpt(multiple_sentence)
            output = list(output)
            for i in output:
                replaced_output.append(i)
            curr_token_len = token_count
            multiple_sentence = replace_command
            break
    else:
        no_replace = {"Original Sentence": each_text, 
                    "NER Instructions": "", 
                    "Replaced Sentence": each_text}
        replaced_output.append(no_replace)

for i in replaced_output:
    index = text_no_punctuations.index(i.get("Original Sentence", "").translate(str.maketrans('', '', string.punctuation)).lower())
    i["id"] = ids[index]

    
with open(sys.argv[2], 'w') as f:
    json.dump(replaced_output, f, indent=4)