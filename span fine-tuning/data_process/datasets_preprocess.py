from data_process import dense_sent_tokenization
from tqdm import tqdm
import json

BERT_TOKENIZER_MAP = {'albert': dense_sent_tokenization.MultiTokenizerAlbert,
                      'bert': dense_sent_tokenization.MultiTokenizer}


class SnliDataset(object):
    def __init__(self,
                 data_dir,
                 bert_type='albert',
                 max_chunk_number=32,
                 ):
        self.data_dir = data_dir
        self.max_chunk_number = max_chunk_number
        self.tokenizer = BERT_TOKENIZER_MAP[bert_type](max_chunk_number=max_chunk_number)
        self.label_2_id = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

    def preprocess_snli(self, output_dir='data_process/data/dev'):
        instances = []
        with open(self.data_dir) as file:
            for line in file:
                instance = {}
                line = json.loads(line)
                if line['gold_label'] == '-':
                    # In the case of this unknown label, we will skip the whole datapoint
                    continue
                instance['sentence1'] = line['sentence1']
                instance['sentence2'] = line['sentence2']
                instance['label'] = line['gold_label']
                instances.append(instance)
        with open(output_dir, 'w') as outfile:
            for instance in tqdm(instances):
                snli_feature = self.convert_snli_instance_2_feature(instance)
                json.dump(snli_feature.to_dict(), outfile)
                outfile.write('\n')

    def convert_snli_instance_2_feature(self, instance, max_seq_len=128):
        sentence1_tokens, sentence1_chunk_lens = self.tokenizer.tokenize(instance['sentence1'])
        sentence2_tokens, sentence2_chunk_lens = self.tokenizer.tokenize(instance['sentence2'])
        concated_tokens = ['[CLS]'] + sentence1_tokens + ['[SEP]'] + sentence2_tokens
        concated_chunk_lens = [1] + sentence1_chunk_lens + [1] + sentence2_chunk_lens
        pad_tokens = ['[PAD]'] * (max_seq_len - len(concated_tokens))
        pad_chunk_lens = [0] * (self.max_chunk_number * 2 - len(concated_chunk_lens))
        padded_tokens = concated_tokens + pad_tokens
        padded_chunk_lens = concated_chunk_lens + pad_chunk_lens
        token_type_ids = [0] * (len(sentence1_tokens) + 2) + [1] * (len(sentence2_tokens)) \
                         + [0] * (max_seq_len - len(concated_tokens))
        padded_tokens_id = self.tokenizer.bert_tokenizer.convert_tokens_to_ids(padded_tokens)
        attention_mask = [1] * len(concated_tokens) + [0] * (max_seq_len - len(concated_tokens))
        label_id = self.label_2_id[instance['label']]
        return SnliFeatures(padded_tokens=padded_tokens, padded_chunk_lens=padded_chunk_lens,
                            token_type_ids=token_type_ids,
                            padded_tokens_id=padded_tokens_id, attention_mask=attention_mask, label_id=label_id)


class SnliFeatures(object):
    def __init__(self,
                 padded_tokens,
                 padded_chunk_lens,
                 token_type_ids,
                 padded_tokens_id,
                 attention_mask,
                 label_id
                 ):
        self.padded_tokens = padded_tokens
        self.padded_chunk_lens = padded_chunk_lens
        self.token_type_ids = token_type_ids
        self.padded_tokens_id = padded_tokens_id
        self.attention_mask = attention_mask
        self.label_id = label_id

    def to_dict(self):
        feature_dict = {
            'padded_chunk_lens': self.padded_chunk_lens,
            'token_type_ids': self.token_type_ids,
            'padded_tokens_id': self.padded_tokens_id,
            'attention_mask': self.attention_mask,
            'label_id': self.label_id}
        return feature_dict
