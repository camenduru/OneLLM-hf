import os
import json

json_path1 = "llama_adapter_7b.json"
json_path2 = "alpaca_lora_7b.json"
out_path = "llama_adapter_vs_alpaca_lora.json"

data1 = open(json_path1).readlines()
data2 = open(json_path2).readlines()
question = open('question.jsonl').readlines()

assert len(data1) == len(data2) == len(question)

out_data = []
for i, (d1, d2, q) in enumerate(zip(data1, data2, question)):
    d1 = json.loads(d1)
    d2 = json.loads(d2)
    q = json.loads(q)
    out_d = {
        'question_id': i,
        'instruction': q['text'],
        'input': '',
        "response1": d1['text'],
        "response2": d2['text'],
    }
    out_data.append(out_d)

# remove bias
for i, (d1, d2, q) in enumerate(zip(data2, data1, question)):
    d1 = json.loads(d1)
    d2 = json.loads(d2)
    q = json.loads(q)
    out_d = {
        'question_id': i+80,
        'instruction': q['text'],
        'input': '',
        "response1": d1['text'],
        "response2": d2['text'],
    }
    out_data.append(out_d)

with open(out_path, 'w') as f:
    # f.write("\n".join([json.dumps(x) for x in out_data]))
    json.dump(out_data, f)