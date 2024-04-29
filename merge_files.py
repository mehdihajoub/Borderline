import json

def merge_json(json_file1, json_file2, output_file):

    with open(json_file1, 'r') as f1:
        data1 = json.load(f1)

    with open(json_file2, 'r') as f2:
        data2 = json.load(f2)

    # Merge the data 
    merged_data = data1 + data2

    # new JSON file
    with open(output_file, 'w') as output_f:
        json.dump(merged_data, output_f, indent=4)

    print("JSON files merged successfully.")

json_file1 = "./data/opposite_data.json"
json_file2 = "./data/opposite_data_2.json"
output_file = "./data/opposite_data_total.json"

merge_json(json_file1, json_file2, output_file)