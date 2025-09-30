import os
import re
import time
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams



parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=False, default='output')
parser.add_argument('--dataset', type=str, default="data")
parser.add_argument('--batch_size', required=False, type=int, default=8)
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--gpu_memory_utilization", type=float, default=0)
parser.add_argument("--max_model_len", type=int, default=0)
parser.add_argument("--max_output_len", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--min_p", type=float, default=0)


args = parser.parse_args()



dataset = args.dataset
model_name = args.model_name_or_path
save_path = args.save_path




print(f"\n\nconfiguration")
print(f"*{'-'*10}*")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

print(f"*{'-'*10}*\n\n")

tokenizer = args.tokenizer
if tokenizer is None:
    tokenizer = args.model_name_or_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

if args.num_gpu>1:
    if args.gpu_memory_utilization>0 and args.max_model_len>0:
        llm = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer, dtype='bfloat16', task="generate", tensor_parallel_size=args.num_gpu, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    else:
        llm = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer, dtype='bfloat16', task="generate", tensor_parallel_size=args.num_gpu)
else:
    if args.gpu_memory_utilization>0 and args.max_model_len>0:
        llm = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer, dtype='bfloat16', task="generate", gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    else:
        llm = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer, dtype='bfloat16', task="generate")

if args.temperature and args.top_p and args.top_k and args.min_p:
    sampling_params = SamplingParams(max_tokens=args.max_output_len, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p)
else:
    sampling_params = SamplingParams(max_tokens=args.max_output_len, temperature=0)

def response(conversations):
    outputs = llm.chat(messages=conversations,
                   sampling_params=sampling_params, use_tqdm=False)
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    return responses



##generate responses##
if not os.path.exists(save_path):
    os.makedirs(save_path)



save_name = f'{save_path}/{dataset.split("/")[-1].replace(".json","")}_{model_name.split("/")[-1]}.json'

system_message = "You are a helpful AI assistant skilled in multi-round conversations. \
Please answer the user's questions based on the context of the discussions, ensuring coherence and relevance. Answer in less than **128** words."



with open(dataset, 'r') as json_file:
    data = json.load(json_file)

print("generating responses...\n")

quotient = len(data) // args.batch_size
remainder = len(data) % args.batch_size
if remainder:
    quotient += 1
for idx in tqdm(range(quotient)):
    print("Batch: ", idx)
    if remainder and idx == quotient-1:
        samples = data[idx*8:]
    else:
        samples = data[idx*8:(idx+1)*8]
    
    # for sample in samples:
    #     print("Example ID: ", sample["example_id"])

    conversations = []
    for _ in range(len(samples)):
        conversations.append([{"role": "system", "content": system_message}])

    for round in range(5):
        for i, sample in enumerate(samples):
            conversations[i].append({"role": "user", "content": sample[f"{round}-turn Conv"]})
        answers = response(conversations)
        for answer, conversation in zip(answers, conversations):
            conversation.append({"role": "assistant", "content": answer})
    
    with open(f'{save_name}', 'a', encoding='utf-8') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation)+"\n")
