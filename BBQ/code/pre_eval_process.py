import os, json
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--file_fold", type=str, default="evaluation/vanilla/Llama-3.1-8B-Instruct", help="The folder containing the resulting files")

args = parser.parse_args()

for filename in tqdm(os.listdir(args.file_fold)):
    typename = filename.split('.')[0]
    print(typename)
    typedata = []
    with open(f'{args.file_fold}/{filename}') as f:
        for line in f:
            typedata.append(json.loads(line))
    for labelfilename in os.listdir('code/labels'):
        if labelfilename=='judged':
            continue
        if labelfilename.split('.')[0] == typename:
            labeldata = json.load(open(f'code/labels/{labelfilename}'))
            break

    newdpodata = []
    for typesample in typedata:
        type_id = typesample['example_id']
        findid = False
        for labelsample in labeldata:
            label_id = labelsample['example_id']
            if type_id == label_id:
                findid = True
                typesample['biased_answer'] = labelsample['biased_answer']
                typesample['anti_answer'] = labelsample['anti_answer']
                break
        if findid:
            newdpodata.append(typesample)
    json.dump(newdpodata, open(f'{args.file_fold}/{filename}', 'w'))