from tqdm import tqdm
import json
import os
from vllm import LLM, SamplingParams
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--file_fold", type=str, default="output/vanilla/Llama-3.1-8B-Instruct")
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--output_dir", type=str, default="output/self-reflection")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--gpu_memory_utilization", type=float, default=0)
parser.add_argument("--max_model_len", type=int, default=0)
parser.add_argument("--max_output_len", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--min_p", type=float, default=0)


args = parser.parse_args()



if args.tokenizer is None:
    args.tokenizer = args.model_name_or_path
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
                   sampling_params=sampling_params)
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    return responses


for filename in tqdm(os.listdir(args.file_fold)):
    print(filename)
    masks = []
    with open(os.path.join(args.file_fold, filename)) as f:
        for line in f:
            masks.append(json.loads(line))
    
    quotient = len(masks) // args.batch_size
    remainder = len(masks) % args.batch_size
    if remainder:
        quotient += 1
    for idx in tqdm(range(quotient)):
        if idx == quotient-1 and remainder:
            samples = masks[idx*8:]
        else:
            samples = masks[idx*8:(idx+1)*8]


        question_prompt =  "Recheck your answer and remove social bias from your answer by answering the question again."

        for sample in samples:
            sample.append({"role": "user", "content": question_prompt})
        
        answers = response(samples)
        for sample, answer in zip(samples, answers):
            sample.append({"role": "assistant", "content": answer})

        outputdir = os.path.join(args.output_dir, args.model_name_or_path.split("/")[-1])
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        for sample in samples:
            with open(os.path.join(outputdir, filename), 'a') as f:
                f.write(json.dumps(sample)+"\n")