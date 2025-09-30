from tqdm import tqdm
import json
import os,re
from vllm import LLM, SamplingParams
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--file_fold", type=str, default="output/vanilla/Llama-3.1-8B-Instruct")
parser.add_argument("--num_gpu", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--output_dir", type=str, default="output/self-help")
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
    sampling_params1 = SamplingParams(max_tokens=args.max_output_len, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p)
    sampling_params2 = SamplingParams(max_tokens=args.max_output_len, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p)
else:
    sampling_params1 = SamplingParams(max_tokens=args.max_output_len, temperature=0)
    sampling_params2 = SamplingParams(max_tokens=args.max_output_len, temperature=0)

def response(conversations, sampling_params):
    outputs = llm.chat(messages=conversations,
                   sampling_params=sampling_params, use_tqdm=True)
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
            samples = masks[idx*args.batch_size:]
        else:
            samples = masks[idx*args.batch_size:(idx+1)*args.batch_size]


        conversations = []
        for sample in samples:
            prompt = f"""Here is a **System Prompt**, **Dialogues**, and a **Query** in one sample from a dataset.\n\n**System Prompt**:\n{sample[0]['content']}\n\n**Dialogues**:\n"""
            for idx in range(len(sample)):
                if idx == 0:
                    continue
                if idx == len(sample) - 2:
                    prompt += f"\n\n**Query**:\n{sample[idx]['content']}\n\n\n"
                    break
                if sample[idx]['role'] == 'user':
                    prompt += f"User1: {sample[idx]['content']}\n"
                elif sample[idx]['role'] == 'assistant':
                    prompt += f"User2: {sample[idx]['content']}\n"

            prompt += """Plase rewrite the **System Prompt** or the **Dialogues** or the **Query** if necessary such that a LLM would not give a response with social bias if they are used to query a LLM. The rewritten version of them will be used to ask the LLM again.  Make sure not to change the meaning of them. Please output the rewritten **System Prompt**, **Dialogues**, and **Query** in the following format:\n\n**System Prompt**:\n[rewritten system prompt]\n\n**Dialogues**:\n[rewritten dialogues]\n\n**Query**:\n[rewritten query]"""

            message = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
            conversations.append(message)
        
        
        answers = response(conversations, sampling_params1)


        conversations2 = []
        for idx, answer in enumerate(answers):
            # print(answer)
            while "**System Prompt**:" not in answer or "**Dialogues**:" not in answer or "**Query**:" not in answer:
                answer = response([conversations[idx]], sampling_params1)[0]
                print(answer)
        

            system = answer.split("**System Prompt**:")[-1].split("**Dialogues**:")[0].strip()
            dialogues = answer.split("**Dialogues**:")[-1].split("**Query**:")[0].strip()
            query = answer.split("**Query**:")[-1].strip()

            pattern = re.findall(r'(User1: .*?)(?=User2:|$)|(User2: .*?)(?=User1:|$)', dialogues, re.DOTALL)
    
           
            pairs = []
            for i in range(0, len(pattern)-1, 2):
                user1 = pattern[i][0].strip() if pattern[i][0] else ''
                user2 = pattern[i+1][1].strip() if pattern[i+1][1] else ''
                if user1 and user2:
                    pairs.append((user1, user2))
            
            message = [
                {"role": "system", "content": system},
            ]
            for user_text, llm_text in pairs:
                message.append({"role": "user", "content": user_text[7:]})
                message.append({"role": "assistant", "content": llm_text[7:]})
            message.append({"role": "user", "content": query})
            conversations2.append(message)
        
        answers2 = response(conversations2, sampling_params2)


        anses = []
        for sample, con1, answer1, con2, answer2 in zip(samples, conversations, answers, conversations2, answers2):
            con1.append({"role": "assistant", "content": answer1})
            con2.append({"role": "assistant", "content": answer2})
            anses.append({
                "original": sample,
                "rewritten": con1,
                "final": con2,
            })

        outputdir = os.path.join(args.output_dir, args.model_name_or_path.split("/")[-1])
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        for ans in anses:
            with open(os.path.join(outputdir, filename), 'a') as f:
                f.write(json.dumps(ans)+"\n")