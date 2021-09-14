import torch
import numpy as np
import json
import torch
from tqdm import tqdm
import string
import json
N =5
ngram_all = []
for i in range(N-1) :
  ngram_all.append({})
with open('data_process/ngram_sample/trainfile-5.count','r', encoding='UTF-8') as file:
    for line in tqdm(file):
        line = line.strip().split('\t')
        if len(line) != 2:
            continue
        words = line[0].split()
        if len(words) == 1 or len(words) > 5:
            continue
        flag = 0
        for word in words:
            if word in string.punctuation:
                flag = 1
        if flag == 1:
            continue
        ngram_all[len(words)-2][line[0]] = int(line[1])
ngram_sorted = []
for i in range(N-1) :
  sroted = sorted(ngram_all[i].items(),key=lambda d: d[1],reverse=True)
  print(len(sroted))
  ngram_sorted.append(sroted)
with open('data_process/ngram_sample/ngram_sorted','w') as f:
    json.dump(ngram_sorted,f)


import json
with open('data_process/ngram_sample/ngram_sorted') as f:
    ngram_sorted = json.load(f)
ngram_pruned_10 = []
for i in range(4) :
    index = 0
    for item in ngram_sorted[i] :
        if(item[1]<10) :
            break
        index += 1
    print(index)
    ngram_pruned_10.append(ngram_sorted[i][:index])
with open('data_process/ngram_sample/ngram_pruned_10','w') as f:
    json.dump(ngram_pruned_10,f)

import json
import pandas as pd
import numpy as np
with open('data_process/ngram_sample/ngram_pruned_10') as f:
    ngram_pruned_10 = json.load(f)
df0 = pd.DataFrame(ngram_pruned_10[0], columns=['words', 'number'])
grouped = df0.groupby(df0['number'].apply(lambda x: int(x//10)))
len(ngram_pruned_10[0])
grouped_ngram = grouped.size()
len(grouped_ngram)
sampled_grouped_ngram = grouped_ngram[:30]
ngram_number = list(sampled_grouped_ngram)
frequency = list(sampled_grouped_ngram.index)
for i,_ in enumerate(frequency):
    frequency[i]*=10
import matplotlib.pyplot as plt
import numpy as np
plt.plot(frequency,ngram_number)

df1 = pd.DataFrame(ngram_pruned_10[1], columns=['words', 'number'])
grouped = df1.groupby(df1['number'].apply(lambda x: int(x//10)))
grouped_ngram = grouped.size()
len(grouped_ngram)
sampled_grouped_ngram = grouped_ngram[:30]
ngram_number = list(sampled_grouped_ngram)
frequency = list(sampled_grouped_ngram.index)
for i,_ in enumerate(frequency):
    frequency[i]*=10
import matplotlib.pyplot as plt
import numpy as np
plt.plot(frequency,ngram_number)

df2 = pd.DataFrame(ngram_pruned_10[1], columns=['words', 'number'])
grouped = df2.groupby(df2['number'].apply(lambda x: int(x//5)))
grouped_ngram = grouped.size()
len(grouped_ngram)
sampled_grouped_ngram = grouped_ngram[:20]
ngram_number = list(sampled_grouped_ngram)
frequency = list(sampled_grouped_ngram.index)
for i,_ in enumerate(frequency):
    frequency[i]*=5
import matplotlib.pyplot as plt
import numpy as np
plt.plot(frequency,ngram_number)

df3 = pd.DataFrame(ngram_pruned_10[1], columns=['words', 'number'])
grouped = df3.groupby(df3['number'].apply(lambda x: int(x//5)))
grouped_ngram = grouped.size()
len(grouped_ngram)
sampled_grouped_ngram = grouped_ngram[:20]
ngram_number = list(sampled_grouped_ngram)
frequency = list(sampled_grouped_ngram.index)
for i,_ in enumerate(frequency):
    frequency[i]*=5
import matplotlib.pyplot as plt
import numpy as np
plt.plot(frequency,ngram_number)






import json
with open('data_process/ngram_sample/ngram_pruned_10') as f:
    ngram_pruned_10 = json.load(f)
sampled_dict_10 = {}
for ngram_list in ngram_pruned_10 :
    for ngram in ngram_list:
        sampled_dict_10[ngram[0].lower()] = sampled_dict_10.get(ngram[0].lower(),0) + ngram[1]
with open('data_process/ngram_sample/sampled_dict_10','w') as f:
    json.dump(sampled_dict_10,f)

with open('data_process/ngram_sample/sampled_dict_entropy.json') as f:
    sampled_dict_entropy = json.load(f)

import json
with open('data_process/ngram_sample/sampled_dict_entropy.json') as f:
    sampled_dict_entropy = json.load(f)
with open('data_process/ngram_sample/sampled_dict_10') as f:
    sampled_dict_10 = json.load(f)
for key in sampled_dict_entropy.keys() :
    if len(key.split()) > 5 :
        continue
    sampled_dict_10[key.lower()] = sampled_dict_10.get(key.lower(),0) + 20
with open('data_process/ngram_sample/sampled_dict_10_entropy','w') as f:
    json.dump(sampled_dict_10,f)

with open('data_process/ngram_sample/sampled_dict_10_entropy') as f:
    sampled_dict_entropy = json.load(f)

with open('data_process/ngram_sample/sampled_dict_10') as f:
    sampled_dict_entropy = json.load(f)








import json
with open('data_process/ngram_sample/sampled_dict_entropy.json') as f:
    sampled_dict_entropy = json.load(f)
with open('data_process/ngram_sample/sampled_dict_10') as f:
    sampled_dict_20 = json.load(f)
for key in sampled_dict_entropy.keys() :
    if len(key.split()) > 5 :
        continue
    sampled_dict_20[key.lower()] = sampled_dict_20.get(key.lower(),0) + 20
with open('data_process/ngram_sample/sampled_dict_20_entropy','w') as f:
    json.dump(sampled_dict_20,f)











import json
with open('data_process/ngram_sample/ngram_pruned_10') as f:
    ngram_pruned_10 = json.load(f)
sampled_dict_20 = {}
for ngram_list in ngram_pruned_10 :
    for ngram in ngram_list:
        if(ngram[1] >= 20) :
         sampled_dict_20[ngram[0].lower()] = sampled_dict_20.get(ngram[0].lower(),0) + ngram[1]
print(len(sampled_dict_20.keys()))
with open('data_process/ngram_sample/sampled_dict_20','w') as f:
    json.dump(sampled_dict_20,f)

##749408


import json
def make_n_pruned_dict(n) :
    with open('data_process/ngram_sample/ngram_pruned_10') as f:
        ngram_pruned_10 = json.load(f)
    sampled_dict = {}
    for ngram_list in ngram_pruned_10:
        for ngram in ngram_list:
            if (ngram[1] >= n):
                sampled_dict[ngram[0].lower()] = sampled_dict.get(ngram[0].lower(), 0) + ngram[1]
    print(len(sampled_dict.keys()))
    with open( 'data_process/ngram_sample/sampled_dict_'+str(n), 'w') as f:
        json.dump(sampled_dict, f)
    return len(sampled_dict.keys())

def count_n_pruned_dict(n) :
    with open('data_process/ngram_sample/ngram_pruned_10') as f:
        ngram_pruned_10 = json.load(f)
    sampled_dict = {}
    for ngram_list in ngram_pruned_10:
        for ngram in ngram_list:
            if (ngram[1] >= n):
                sampled_dict[ngram[0].lower()] = sampled_dict.get(ngram[0].lower(), 0) + ngram[1]
    print(len(sampled_dict.keys()))
    return len(sampled_dict.keys())


dict_size = []
for i in range(30) :
    dict_size.append(make_n_pruned_dict(i*10))

for i in range(100) :
    dict_size.append(count_n_pruned_dict(i))

[1649907,1649907,749408,78914,349141,272843,222990,187951,162249,142716,127234,114536,104034,95223,87661,81049,75370,70380,65947,62061,58469,55207,52150,49577,47115,44938,43001,41152,39464,37813]


count_n_pruned_dict(1000)

from data_process.data_processor import  ColaProcessor

def average_chunk_number_cola(ngram_path) :
    processor = ColaProcessor()
    cola_features  = processor.get_train_features_from_examples(data_dir='data_process/data/glue_data/CoLA',bert_type='bert',max_chunk_number=32,max_seq_len=64,ngram_path=ngram_path)
    total_sentence_number = len(cola_features)
    total_chunk_number = 0
    total_tokens = 0
    for example in cola_features :
        input_ids = example.input_ids
        chunk_lens = example.chunks_len
        token_len = 0
        for id in input_ids :
            if id ==0 :
                break
            token_len += 1
        chunk_number = 0
        for chunk_len in chunk_lens :
            if chunk_len ==0 :
                break
            chunk_number += 1
        token_len = token_len -2
        chunk_number = chunk_number - 2
        total_chunk_number += chunk_number
        total_tokens += token_len
    print(total_tokens/total_sentence_number)
    print(total_chunk_number/total_sentence_number)
    return token_len/total_sentence_number,chunk_number/total_sentence_number

average_chunk_number_cola('data_process/ngram_sample/sampled_dict_1000')

from data_process.data_processor import  MrpcProcessor

def average_chunk_number_mrpc(ngram_path) :
    processor = MrpcProcessor()
    features  = processor.get_train_features_from_examples(data_dir='data_process/data/glue_data/MRPC',bert_type='bert',max_chunk_number=64,max_seq_len=128,ngram_path=ngram_path)
    total_sentence_number = len(features)
    total_chunk_number = 0
    total_tokens = 0
    for example in features :
        input_ids = example.input_ids
        chunk_lens = example.chunks_len
        token_len = 0
        for id in input_ids :
            if id ==0 :
                break
            token_len += 1
        chunk_number = 0
        for chunk_len in chunk_lens :
            if chunk_len ==0 :
                break
            chunk_number += 1
        token_len = token_len -3
        chunk_number = chunk_number - 3
        total_chunk_number += chunk_number
        total_tokens += token_len
    print(total_tokens/total_sentence_number)
    print(total_chunk_number/total_sentence_number)
    return token_len/total_sentence_number,chunk_number/total_sentence_number

average_chunk_number_mrpc('data_process/ngram_sample/sampled_dict_10')