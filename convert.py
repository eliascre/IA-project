import csv
import json

csv_file = 'train.csv'
jsonl_file = 'train.jsonl'

with open(csv_file, mode='r', encoding='utf-8') as f_csv, open(jsonl_file, mode='w', encoding='utf-8') as f_jsonl:
    reader = csv.DictReader(f_csv)
    for row in reader:
        # Écriture de chaque ligne comme un objet JSON séparé
        json_line = json.dumps(row, ensure_ascii=False)
        f_jsonl.write(json_line + '\n')

print(f"Conversion terminée ! '{csv_file}' a été converti en '{jsonl_file}'.")
