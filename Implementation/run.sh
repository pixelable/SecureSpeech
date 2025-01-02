#Run this for Random Speaker Generation

mkdir -p Random_Speaker_Output

python3 replace_ner.py "voxpopuli/voxpopuli/test-00000-of-00001.parquet" \
	"Random_Speaker_Output/ner_output.json" \

python3 generate_random_speaker.py "Random_Speaker_Output/ner_output.json" \
	"Random_Speaker_Output" \
	"Random_Speaker_Output/full_generated_meta_info_random_speaker.json" \

# #Run this for Specific Speaker Generation
# mkdir -p Specific_Speaker_Output

# python3 replace_ner.py "voxpopuli/voxpopuli/test-00000-of-00001.parquet" \
# 	"Specific_Speaker_Output/ner_output.json" \

# python3 generate_specific_speaker.py "Specific_Speaker_Output/ner_output.json" \
# 	"Specific_Speaker_Output" \
# 	"Specific_Speaker_Output/full_generated_meta_info_random_speaker.json" \