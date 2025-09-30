
from collections import Counter
import json,os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--method", type=str, default="vanilla", choices=['vanilla', 'cot-debias', 'self-aware', 'self-reflection', 'self-help', 'sft', 'dpo', 'task_vector', 'saferlhf'])
parser.add_argument("--output_name", type=str, default='llama_res.json')

args = parser.parse_args()
modelname = args.model_name_or_path.split('/')[-1].strip()

def majority_vote(answers):
    count = Counter(answers)
    most_common = count.most_common()
    if most_common[0][1] > 1:
        return most_common[0][0]
    else:
        return "no answer"

if os.path.exists(f'results/{args.output_name}'):
    llama_res = json.load(open(f'results/{args.output_name}'))
else:
    llama_res = dict()

age = json.load(open(f'evaluation/{args.method}/{modelname}/Age.json'))
age1 = json.load(open(f'evaluation/{args.method}/{modelname}/Age_judged1.json'))
age2 = json.load(open(f'evaluation/{args.method}/{modelname}/Age_judged2.json'))
age3 = json.load(open(f'evaluation/{args.method}/{modelname}/Age_judged3.json'))
assert len(age)==len(age1)==len(age2)==len(age3)

# biased, anti, unknown, unrelated
age1_labels = []
age2_labels = []
age3_labels = []
for line in age1:
    age1_labels.append(line['llm_judge_label'])
for line in age2:
    age2_labels.append(line['llm_judge_label'])
for line in age3:
    age3_labels.append(line['llm_judge_label'])
age_labels = []
for i in range(len(age1_labels)):
    age_labels.append(majority_vote([age1_labels[i], age2_labels[i], age3_labels[i]]))

llama_res[args.method] = {'Age': Counter(age_labels)}

disability = json.load(open(f'evaluation/{args.method}/{modelname}/Disability_status.json'))
disability1 = json.load(open(f'evaluation/{args.method}/{modelname}/Disability_status_judged1.json'))
disability2 = json.load(open(f'evaluation/{args.method}/{modelname}/Disability_status_judged2.json'))
disability3 = json.load(open(f'evaluation/{args.method}/{modelname}/Disability_status_judged3.json'))
assert len(disability)==len(disability1)==len(disability2)==len(disability3)

disability1_labels = []
disability2_labels = []
disability3_labels = []
for line in disability1:
    disability1_labels.append(line['llm_judge_label'])
for line in disability2:
    disability2_labels.append(line['llm_judge_label'])
for line in disability3:
    disability3_labels.append(line['llm_judge_label'])
disability_labels = []
for i in range(len(disability1_labels)):
    disability_labels.append(majority_vote([disability1_labels[i], disability2_labels[i], disability3_labels[i]]))

llama_res[args.method]['Disability_status'] = Counter(disability_labels)

race_identity = json.load(open(f'evaluation/{args.method}/{modelname}/Gender_identity.json'))
race_identity1 = json.load(open(f'evaluation/{args.method}/{modelname}/Gender_identity_judged1.json'))
race_identity2 = json.load(open(f'evaluation/{args.method}/{modelname}/Gender_identity_judged2.json'))
race_identity3 = json.load(open(f'evaluation/{args.method}/{modelname}/Gender_identity_judged3.json'))
assert len(race_identity)==len(race_identity1)==len(race_identity2)==len(race_identity3)
race1_labels = []
race2_labels = []
race3_labels = []
for line in race_identity1:
    race1_labels.append(line['llm_judge_label'])
for line in race_identity2:
    race2_labels.append(line['llm_judge_label'])
for line in race_identity3:
    race3_labels.append(line['llm_judge_label'])
race_identity_labels = []
for i in range(len(race1_labels)):
    race_identity_labels.append(majority_vote([race1_labels[i], race2_labels[i], race3_labels[i]]))
llama_res[args.method]['Gender_identity'] = Counter(race_identity_labels)



nationality = json.load(open(f'evaluation/{args.method}/{modelname}/Nationality.json'))
nationality1 = json.load(open(f'evaluation/{args.method}/{modelname}/Nationality_judged1.json'))
nationality2 = json.load(open(f'evaluation/{args.method}/{modelname}/Nationality_judged2.json'))
nationality3 = json.load(open(f'evaluation/{args.method}/{modelname}/Nationality_judged3.json'))
assert len(nationality)==len(nationality1)==len(nationality2)==len(nationality3)

nationality1_labels = []
nationality2_labels = []
nationality3_labels = []
for line in nationality1:
    nationality1_labels.append(line['llm_judge_label'])
for line in nationality2:
    nationality2_labels.append(line['llm_judge_label'])
for line in nationality3:
    nationality3_labels.append(line['llm_judge_label'])
nationality_labels = []
for i in range(len(nationality1_labels)):
    nationality_labels.append(majority_vote([nationality1_labels[i], nationality2_labels[i], nationality3_labels[i]]))
llama_res[args.method]['Nationality'] = Counter(nationality_labels)

physical_appearance = json.load(open(f'evaluation/{args.method}/{modelname}/Physical_appearance.json'))
physical_appearance1 = json.load(open(f'evaluation/{args.method}/{modelname}/Physical_appearance_judged1.json'))
physical_appearance2 = json.load(open(f'evaluation/{args.method}/{modelname}/Physical_appearance_judged2.json'))
physical_appearance3 = json.load(open(f'evaluation/{args.method}/{modelname}/Physical_appearance_judged3.json'))
assert len(physical_appearance)==len(physical_appearance1)==len(physical_appearance2)==len(physical_appearance3)

physical_appearance1_labels = []
physical_appearance2_labels = []
physical_appearance3_labels = []
for line in physical_appearance1:
    physical_appearance1_labels.append(line['llm_judge_label'])
for line in physical_appearance2:
    physical_appearance2_labels.append(line['llm_judge_label'])
for line in physical_appearance3:
    physical_appearance3_labels.append(line['llm_judge_label'])
physical_appearance_labels = []
for i in range(len(physical_appearance1_labels)):
    physical_appearance_labels.append(majority_vote([physical_appearance1_labels[i], physical_appearance2_labels[i], physical_appearance3_labels[i]]))
llama_res[args.method]['Physical_appearance'] = Counter(physical_appearance_labels)

race_ethnicity = json.load(open(f'evaluation/{args.method}/{modelname}/Race_ethnicity.json'))
race_ethnicity1 = json.load(open(f'evaluation/{args.method}/{modelname}/Race_ethnicity_judged1.json'))
race_ethnicity2 = json.load(open(f'evaluation/{args.method}/{modelname}/Race_ethnicity_judged2.json'))
race_ethnicity3 = json.load(open(f'evaluation/{args.method}/{modelname}/Race_ethnicity_judged3.json'))
assert len(race_ethnicity)==len(race_ethnicity1)==len(race_ethnicity2)==len(race_ethnicity3)

race_ethnicity1_labels = []
race_ethnicity2_labels = []
race_ethnicity3_labels = []
for line in race_ethnicity1:
    race_ethnicity1_labels.append(line['llm_judge_label'])
for line in race_ethnicity2:
    race_ethnicity2_labels.append(line['llm_judge_label'])
for line in race_ethnicity3:
    race_ethnicity3_labels.append(line['llm_judge_label'])
race_ethnicity_labels = []
for i in range(len(race_ethnicity1_labels)):
    race_ethnicity_labels.append(majority_vote([race_ethnicity1_labels[i], race_ethnicity2_labels[i], race_ethnicity3_labels[i]]))
llama_res[args.method]['Race_ethnicity'] = Counter(race_ethnicity_labels)



religion = json.load(open(f'evaluation/{args.method}/{modelname}/Religion.json'))
religion1 = json.load(open(f'evaluation/{args.method}/{modelname}/Religion_judged1.json'))
religion2 = json.load(open(f'evaluation/{args.method}/{modelname}/Religion_judged2.json'))
religion3 = json.load(open(f'evaluation/{args.method}/{modelname}/Religion_judged3.json'))
assert len(religion)==len(religion1)==len(religion2)==len(religion3)

religion1_labels = []
religion2_labels = []
religion3_labels = []
for line in religion1:
    religion1_labels.append(line['llm_judge_label'])
for line in religion2:
    religion2_labels.append(line['llm_judge_label'])
for line in religion3:
    religion3_labels.append(line['llm_judge_label'])
religion_labels = []
for i in range(len(religion1_labels)):
    religion_labels.append(majority_vote([religion1_labels[i], religion2_labels[i], religion3_labels[i]]))

llama_res[args.method]['Religion'] = Counter(religion_labels)

ses = json.load(open(f'evaluation/{args.method}/{modelname}/SES.json'))
ses1 = json.load(open(f'evaluation/{args.method}/{modelname}/SES_judged1.json'))
ses2 = json.load(open(f'evaluation/{args.method}/{modelname}/SES_judged2.json'))
ses3 = json.load(open(f'evaluation/{args.method}/{modelname}/SES_judged3.json'))
assert len(ses)==len(ses1)==len(ses2)==len(ses3)

ses1_labels = []
ses2_labels = []
ses3_labels = []
for line in ses1:
    ses1_labels.append(line['llm_judge_label'])
for line in ses2:
    ses2_labels.append(line['llm_judge_label'])
for line in ses3:
    ses3_labels.append(line['llm_judge_label'])
ses_labels = []
for i in range(len(ses1_labels)):
    ses_labels.append(majority_vote([ses1_labels[i], ses2_labels[i], ses3_labels[i]]))
llama_res[args.method]['SES'] = Counter(ses_labels)




sexual_orientation = json.load(open(f'evaluation/{args.method}/{modelname}/Sexual_orientation.json'))
sexual_orientation1 = json.load(open(f'evaluation/{args.method}/{modelname}/Sexual_orientation_judged1.json'))
sexual_orientation2 = json.load(open(f'evaluation/{args.method}/{modelname}/Sexual_orientation_judged2.json'))
sexual_orientation3 = json.load(open(f'evaluation/{args.method}/{modelname}/Sexual_orientation_judged3.json'))
assert len(sexual_orientation)==len(sexual_orientation1)==len(sexual_orientation2)==len(sexual_orientation3)

sexual_orientation1_labels = []
sexual_orientation2_labels = []
sexual_orientation3_labels = []
for line in sexual_orientation1:
    sexual_orientation1_labels.append(line['llm_judge_label'])
for line in sexual_orientation2:
    sexual_orientation2_labels.append(line['llm_judge_label'])
for line in sexual_orientation3:
    sexual_orientation3_labels.append(line['llm_judge_label'])
sexual_orientation_labels = []
for i in range(len(sexual_orientation1_labels)):
    sexual_orientation_labels.append(majority_vote([sexual_orientation1_labels[i], sexual_orientation2_labels[i], sexual_orientation3_labels[i]]))
llama_res[args.method]['Sexual_orientation'] = Counter(sexual_orientation_labels)




json.dump(llama_res, open(f'results/{args.output_name}', 'w'))