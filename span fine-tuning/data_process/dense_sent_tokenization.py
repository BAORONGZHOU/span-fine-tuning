from pytorch_pretrained_bert import tokenization
import json
from transformers import AlbertTokenizer
from transformers import BertTokenizer


TOKENIZER_MAP = {"bert": BertTokenizer, "albert": AlbertTokenizer}
TOKENIZER_PATH_MAP = {"bert":'bert-base-uncased',"albert" : 'albert-base-v2'}

class MultiTokenizer(object):

    def __init__(self,
                 ngram_path='data_process/ngram_sample/sampled_dict_10_entropy', max_n=5, max_chunk_number=32, max_seq_len=64,
                 bert_model_type='bert'):
        self.max_seq_len = max_seq_len
        self.max_chunk_number = max_chunk_number
        self.bert_tokenizer = TOKENIZER_MAP[bert_model_type].from_pretrained(TOKENIZER_PATH_MAP[bert_model_type])
        with open(ngram_path) as f:
            for line in f:
                ngram_dict = json.loads(line)
        self.ngram_dict = ngram_dict
        self.max_n = max_n

    def tokenize(self, text):
        orig_tokens = text.lower().split()
        chunked_text = []
        index = 0
        max_n = self.max_n
        while index < len(orig_tokens):
            if (index + max_n) > len(orig_tokens):
                max_n = len(orig_tokens) - 1 - index
            flag = 0
            for j in range(max_n - 1):
                if self.ngram_dict.get(' '.join(orig_tokens[index:index + max_n - j])) is not None:
                    chunked_text.append(' '.join(orig_tokens[index:index + max_n - j]))
                    index += max_n - j
                    flag = 1
                    break
            if flag == 0:
                chunked_text.append(orig_tokens[index])
                index += 1
        chunk_lens = []
        wordpiece_tokens = []
        flag = 0
        for (index, chunk) in enumerate(chunked_text):
            if index >= self.max_chunk_number - 3:
                flag = 1
                break
            chunk_piece = self.bert_tokenizer.tokenize(chunk)
            if (len(wordpiece_tokens) + len(chunk_piece)) > self.max_seq_len -3:
                flag = 1
                break
            wordpiece_tokens += chunk_piece
            chunk_lens.append(len(chunk_piece))
        if flag == 1 :
            print('too long!')
        return (wordpiece_tokens, chunk_lens)

    def convert_sentence_2_chunk_feature(self, text):
        wordpiece_tokens, chunk_lens = self.tokenize(text)
        tokens_id = self.bert_tokenizer.convert_tokens_to_ids(wordpiece_tokens)
        output_tokens_id = [0] * self.max_seq_len
        attention_mask = [0] * self.max_seq_len
        for i, token_id in enumerate(tokens_id):
            output_tokens_id[i] = token_id
            attention_mask[i] = 1
        output_chunks_len = [0] * self.max_chunk_number
        for i, chunk_len in enumerate(chunk_lens):
            output_chunks_len[i] = chunk_len
        return SentenceFeature(output_tokens_id, attention_mask, output_chunks_len)

    def chunk_text(self, piece_list, chunk_lens):
        chunked_piece_text = []
        piece_index = 0
        for chunk_len in chunk_lens:
            piece_chunk_text = []
            for i in range(chunk_len):
                piece_chunk_text.append(piece_list[piece_index + i])
            chunked_piece_text.append('_'.join(piece_chunk_text))
            piece_index += chunk_len
        return ' '.join(chunked_piece_text)

    def make_chunk(self, text):
        piece_list, chunk_lens = self.tokenize(text)
        return self.chunk_text(piece_list, chunk_lens)


class SentenceFeature(object):
    def __init__(self, tokens_id, attention_mask, chunks_len):
        self.tokens_id = tokens_id
        self.attention_mask = attention_mask
        self.chunks_len = chunks_len
