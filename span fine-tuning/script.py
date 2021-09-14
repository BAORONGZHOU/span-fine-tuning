
from transformers import BertConfig, BertModel,BertTokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('cached_model/bert-base-uncased')


from transformers import AlbertTokenizer, AlbertModel
import torch
tokenizer = AlbertTokenizer.from_pretrained('cached_model/albert-base-v2')
model = AlbertModel.from_pretrained('cached_model/albert-xxlarge-v2')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

from data_process.albert_tokenization import albert_tokenization
tokenizer = albert_tokenization.AlbertTokenizer.from_pretrained('cached_model/albert-base-v2')
tokenizer.tokenize('inaffable')

import json
from transformers import AlbertTokenizer, AlbertModel
tokenizer = AlbertTokenizer.from_pretrained('cached_model/albert-base-v2')
with open('data_process/sampled_dict.json') as f:
    for line in f:
        ngram_dict = json.loads(line)

from data_process import datasets_preprocess
snli_dataset = datasets_preprocess.SnliDataset('data_process/snli_1.0/snli_1.0_train.jsonl')
snli_dataset.preprocess_snli(output_dir='data_process/data/snli_uncase/train')

from data_process import data_processor
processor = data_processor.MrpcProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/MRPC',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/MRPC/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/MRPC',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/MRPC/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/MRPC',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/MRPC/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/MRPC',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/MRPC/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/MRPC',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/MRPC/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/MRPC',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/MRPC/test')


from data_process import data_processor
processor = data_processor.MnliProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='dev',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/MNLI/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='train',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/MNLI/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='test',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/MNLI/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='dev',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/MNLI/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='train',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/MNLI/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='test',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/MNLI/test')

from data_process import data_processor
processor = data_processor.MnliMismatchedProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='dev',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/MNLI/dev_mismatched')
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='test',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/MNLI/test_mismatched')
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='dev',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/MNLI/dev_mismatched')
processor.make_feature_file(data_dir='data_process/data/glue_data/MNLI',set_type='test',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/MNLI/test_mismatched')

from data_process import data_processor
processor = data_processor.ColaProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA', set_type='dev', max_seq_len=64, max_chunk_number=48, bert_type='bert',
                            output_dir='data_process/processed_data/bert/CoLA/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA', set_type='train', max_seq_len=64, max_chunk_number=48, bert_type='bert',
                            output_dir='data_process/processed_data/bert/CoLA/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA', set_type='test', max_seq_len=64, max_chunk_number=48, bert_type='bert',
                            output_dir='data_process/processed_data/bert/CoLA/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='dev',max_seq_len=64,max_chunk_number=48,bert_type='albert',
                            output_dir='data_process/processed_data/albert/CoLA/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='train',max_seq_len=64,max_chunk_number=48,bert_type='albert',
                            output_dir='data_process/processed_data/albert/CoLA/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='test',max_seq_len=64,max_chunk_number=48,bert_type='albert',
                            output_dir='data_process/processed_data/albert/CoLA/test')

from data_process import data_processor
processor = data_processor.SnliProcessor()
processor.make_feature_file(data_dir='data_process/data/snli_1.0',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/SNLI/dev')
processor.make_feature_file(data_dir='data_process/data/snli_1.0',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/SNLI/train')
processor.make_feature_file(data_dir='data_process/data/snli_1.0',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/SNLI/test')
processor.make_feature_file(data_dir='data_process/data/snli_1.0',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/SNLI/dev')
processor.make_feature_file(data_dir='data_process/data/snli_1.0',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/SNLI/train')
processor.make_feature_file(data_dir='data_process/data/snli_1.0',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/SNLI/test')
processor.get_dev_features("data_process/processed_data/bert/SNLI")


from data_process import data_processor
processor = data_processor.Sst2Processor()
processor.make_feature_file(data_dir='data_process/data/glue_data/SST-2',set_type='dev',max_seq_len=96,max_chunk_number=48,bert_type='bert',
                            output_dir='data_process/processed_data/bert/SST-2/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/SST-2',set_type='train',max_seq_len=96,max_chunk_number=48,bert_type='bert',
                            output_dir='data_process/processed_data/bert/SST-2/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/SST-2',set_type='test',max_seq_len=96,max_chunk_number=48,bert_type='bert',
                            output_dir='data_process/processed_data/bert/SST-2/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/SST-2',set_type='dev',max_seq_len=96,max_chunk_number=48,bert_type='albert',
                            output_dir='data_process/processed_data/albert/SST-2/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/SST-2',set_type='train',max_seq_len=96,max_chunk_number=48,bert_type='albert',
                            output_dir='data_process/processed_data/albert/SST-2/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/SST-2',set_type='test',max_seq_len=96,max_chunk_number=48,bert_type='albert',
                            output_dir='data_process/processed_data/albert/SST-2/test')

from data_process import data_processor
processor = data_processor.WnliProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/WNLI',set_type='dev',max_seq_len=96,max_chunk_number=48,bert_type='bert',
                            output_dir='data_process/processed_data/bert/WNLI/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/WNLI',set_type='train',max_seq_len=96,max_chunk_number=48,bert_type='bert',
                            output_dir='data_process/processed_data/bert/WNLI/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/WNLI',set_type='test',max_seq_len=96,max_chunk_number=48,bert_type='bert',
                            output_dir='data_process/processed_data/bert/WNLI/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/WNLI',set_type='dev',max_seq_len=96,max_chunk_number=48,bert_type='albert',
                            output_dir='data_process/processed_data/albert/WNLI/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/WNLI',set_type='train',max_seq_len=96,max_chunk_number=48,bert_type='albert',
                            output_dir='data_process/processed_data/albert/WNLI/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/WNLI',set_type='test',max_seq_len=96,max_chunk_number=48,bert_type='albert',
                            output_dir='data_process/processed_data/albert/WNLI/test')

from data_process import data_processor
processor = data_processor.ColaProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='dev',max_seq_len=48,max_chunk_number=24,bert_type='bert',
                            output_dir='data_process/processed_data/bert/CoLA/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='train',max_seq_len=48,max_chunk_number=24,bert_type='bert',
                            output_dir='data_process/processed_data/bert/CoLA/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='test',max_seq_len=48,max_chunk_number=24,bert_type='bert',
                            output_dir='data_process/processed_data/bert/CoLA/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='dev',max_seq_len=48,max_chunk_number=24,bert_type='albert',
                            output_dir='data_process/processed_data/albert/CoLA/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='train',max_seq_len=48,max_chunk_number=24,bert_type='albert',
                            output_dir='data_process/processed_data/albert/CoLA/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='test',max_seq_len=48,max_chunk_number=24,bert_type='albert',
                            output_dir='data_process/processed_data/albert/CoLA/test')


from data_process import data_processor
processor = data_processor.QnliProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/QNLI',set_type='dev',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/QNLI/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/QNLI',set_type='train',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/QNLI/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/QNLI',set_type='test',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/QNLI/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/QNLI',set_type='dev',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/QNLI/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/QNLI',set_type='train',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/QNLI/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/QNLI',set_type='test',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/QNLI/test')



from data_process import data_processor
processor = data_processor.RteProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/RTE',set_type='dev',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/RTE/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/RTE',set_type='train',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/RTE/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/RTE',set_type='test',max_seq_len=256,max_chunk_number=128,bert_type='bert',
                            output_dir='data_process/processed_data/bert/RTE/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/RTE',set_type='dev',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/RTE/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/RTE',set_type='train',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/RTE/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/RTE',set_type='test',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/RTE/test')

from data_process import data_processor
processor = data_processor.StsbProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/STS-B/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/STS-B/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/STS-B/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/STS-B/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/STS-B/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/STS-B/test')


from data_process import data_processor
processor = data_processor.QqpProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/QQP',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/QQP/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/QQP',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/QQP/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/QQP',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/QQP/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/QQP',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/QQP/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/QQP',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/QQP/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/QQP',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/QQP/test')


from data_process import data_processor
processor = data_processor.MnliProcessor()
len(processor.get_train_features('data_process/processed_data/bert/MNLI'))

from data_process import data_processor
processor = data_processor.RteProcessor()
len(processor.get_train_features('data_process/processed_data/bert/RTE'))

ALBERT_DATA_DIR = {'CoLA':'data_process/processed_data/albert/CoLA','MNLI':'data_process/processed_data/albert/MNLI',
                   'SNLI':'data_process/processed_data/albert/SNLI',
                   'MRPC':'data_process/processed_data/albert/MRPC'}

BERT_DATA_DIR = {'CoLA':'data_process/processed_data/bert/CoLA','MNLI':'data_process/processed_data/bert/MNLI',
                   'SNLI':'data_process/processed_data/bert/SNLI',
                   'MRPC':'data_process/processed_data/bert/MRPC'}

import torch
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
src = torch.rand(16,10, 5, 512)
out = encoder_layer(src)

from data_process import data_processor
processor = data_processor.StsbProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='dev',max_seq_len=200,max_chunk_number=100,bert_type='bert',
                            output_dir='data_process/processed_data/bert/STS-B-200/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='train',max_seq_len=200,max_chunk_number=100,bert_type='bert',
                            output_dir='data_process/processed_data/bert/STS-B-200/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='test',max_seq_len=200,max_chunk_number=100,bert_type='bert',
                            output_dir='data_process/processed_data/bert/STS-B-200/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='dev',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/STS-B/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='train',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/STS-B/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/STS-B',set_type='test',max_seq_len=256,max_chunk_number=128,bert_type='albert',
                            output_dir='data_process/processed_data/albert/STS-B/test')

from data_process import data_processor
processor = data_processor.QqpProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/QQP',set_type='dev',max_seq_len=200,max_chunk_number=100,bert_type='bert',
                            output_dir='data_process/processed_data/bert/QQP-200/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/QQP',set_type='train',max_seq_len=200,max_chunk_number=100,bert_type='bert',
                            output_dir='data_process/processed_data/bert/QQP-200/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/QQP',set_type='test',max_seq_len=200,max_chunk_number=100,bert_type='bert',
                            output_dir='data_process/processed_data/bert/QQP-200/test')

from data_process import data_processor
processor = data_processor.ColaProcessor()
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/CoLA-128/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/CoLA-128/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='bert',
                            output_dir='data_process/processed_data/bert/CoLA-128/test')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='dev',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/CoLA/dev')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='train',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/CoLA/train')
processor.make_feature_file(data_dir='data_process/data/glue_data/CoLA',set_type='test',max_seq_len=128,max_chunk_number=64,bert_type='albert',
                            output_dir='data_process/processed_data/albert/CoLA/test')

import torch
import torch.nn as nn
cnn = nn.Conv1d(200,200,kernel_size=3,padding=0)
input = torch.zeros([32,128,200])
input = input.transpose(1,2).contiguous()
out = cnn(input)
out.size()

from data_process.dense_sent_tokenization import MultiTokenizer
tokenizer = MultiTokenizer(ngram_path='data_process/ngram_sample/sampled_dict_10_entropy')

text = 'We hate them out of jealousy for being smarter than us.'
text = 'In the early 1950s, student applications declined as a result of increasing crime and poverty in the Hyde Park neighborhood.'
text = 'As a result, Tibetan Buddhism was established as the de facto state religion.'
text = "3 young man in hoods standing in the middle of a quiet street facing the camera."
text = "It is probably not the easiest time to come in and take over the shuttle program"
text = 'The New York Democrat and former first lady has said she will not run for the White House in 2004'
text = 'A little boy in a gray and white striped sweater and tan pants is playing on a piece of playground equipment'
text = 'a muddle splashed with bloody beauty as vivid as any scorsese has ever given us'
text = 'A dog jumping for a Frisbee in the snow'
text = "A boy is jumping on skateboard in the middle of a red bridge"
text = "An older man sits with his orange juice at a small table in a coffee shop while employees in bright colored shirts smile in the background."
text = "An older man is drinking orange juice at a restaurant"
text = 'An animal is outside in the cold weather, playing with a plastic toy.'
text = 'A man with blond-hair, and a brown shirt drinking out of a public water '
text = 'takes a classic story , casts attractive and talented actors and uses a magnificent landscape to create a feature film that is wickedly fun to watch '
text = 'People waiting to get on a train or just getting off.'
text = "Two women who just had lunch hugging and saying goodbye."
text ="The school is having a special event in order to show the american culture on how other cultures are dealt with in parties."
text = 'Two teenage girls conversing next to lockers.'
text = "The bird is flying over the trees"
text = "An old man in a baseball hat and an old woman in a jean jacket are standing outside"
text = "People are standing near water with a large blue boat heading their direction."
text = 'A dog is jumping for a Frisbee in the snow'
text = 'Clinton said that Monica Lewinsky made unwanted sexual advances'
text = "A basketball player with green shoes is dunking the ball in the net"
text = "Two little boys are smiling and laughing while one is standing and one is in a bouncy seat"
text = "Everyone on the street in the city seem to be busy doing their own thing."
text = "Grave authors are the funniest when they don't try to be."
text = "Controllers were working to try to get the plane to change course."
text = "Further chapters give no information about what he felt."
text = "People and a baby are crossing the street at a crosswalk to get home."
text = "A woman in a brown shirt is kneeling in the street."
text = "A blond woman speaks with a group of young dark-haired female students carrying pieces of paper."
text = "Four boys are about to be hit by an approaching wave."
text  ="An Obama Biden supporter cheers for the Presidential candidate and his running mate"
text = "Reporting Entities of the Federal Government under President Obama"
text = 'you keep talking about the president Obama'
text ="In fact, I think that the United States is still, despite Asia, more at risk from inflation than deflation."
text = "The model is intended to give managers an overview of the acquisition process and to help them decrease acquisition risks."
text = "Tesla was able to perform integral calculus in his head, which prompted his teachers to believe that he was cheating."
text = "It was a summer afternoon, and the dog was sitting in the middle of the lawn. After a while, it got up and moved to a spot under the tree"
text = 'Clinton said that Monica Lewinsky made unwanted sexual advances during her time as a journalist in the White House. '
chunk = tokenizer.make_chunk(text)
chunk