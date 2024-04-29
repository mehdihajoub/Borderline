import pandas as pd
import json

xlsx_file = "./data/dataset_sentences_opposites.xlsx"
df = pd.read_excel(xlsx_file)

# Convert DataFrame to a list of dictionaries
data = []
for index, row in df.iterrows():
    sentence = row['Sentence']
    opposite = row['Opposite']
    data.append({'sentence': sentence, 'opposite': opposite})

# Write the data to a JSON file
json_file = "./data/opposite_data_2.json"
with open(json_file, 'w') as f:
    json.dump(data, f, indent=4)

print("JSON file generated successfully.")