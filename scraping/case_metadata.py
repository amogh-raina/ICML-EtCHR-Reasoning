import os

import tqdm

from data import DATA_DIR
import requests

max_count = 100
count = 0
with open(os.path.join(DATA_DIR, 'case_ids.tsv'), 'r') as f_in:
    for line in tqdm.tqdm(f_in):
        line = line.strip()
        case_id = line.split('\t')[0]
        # Download HTML page
        response = requests.get(
            f'https://hudoc.echr.coe.int/app/query/results?query=(contentsitename=ECHR)%20AND%20(itemid=%22{case_id}%22)&select=*&sort=&start=0&length=1',
            timeout=10)

        # Raise error for bad status codes
        response.raise_for_status()
        count += 1
        with open(os.path.join(DATA_DIR, 'json', f'{case_id}.json'), 'w') as f_out:
            f_out.write(response.text)

        if count >= max_count:
            break

print(f'Downloaded {count} cases')