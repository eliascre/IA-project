import csv
import json

csv_file = 'train.csv'
jsonl_file = 'train.jsonl'

# Colonnes que tu veux conserver (modifie selon ton dataset)
columns_to_keep = None  # None = toutes les colonnes

cleaned_count = 0
skipped_count = 0

with open(csv_file, mode='r', encoding='utf-8') as f_csv, open(jsonl_file, mode='w', encoding='utf-8') as f_jsonl:
    reader = csv.DictReader(f_csv)
    
    # Si tu veux, tu peux filtrer les colonnes ici
    if columns_to_keep:
        headers = [col for col in reader.fieldnames if col in columns_to_keep]
    else:
        headers = reader.fieldnames
    
    for i, row in enumerate(reader, start=1):
        clean_row = {}
        skip_line = False
        
        for key in headers:
            value = row.get(key, "").strip()  # enlever espaces
            if value == "":
                skip_line = True  # ignorer ligne si valeur vide
                break
            clean_row[key] = value
        
        if skip_line:
            skipped_count += 1
            continue
        
        # Convertir en JSON et écrire
        json_line = json.dumps(clean_row, ensure_ascii=False)
        f_jsonl.write(json_line + '\n')
        cleaned_count += 1

print(f"Conversion terminée !")
print(f"Lignes nettoyées et écrites : {cleaned_count}")
print(f"Lignes ignorées : {skipped_count}")
