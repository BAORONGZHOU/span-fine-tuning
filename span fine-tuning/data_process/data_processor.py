# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Union
import csv
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Union

from .utils import logging
import os
from data_process import dense_sent_tokenization
from tqdm import tqdm

logger = logging.get_logger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    chunks_len: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_dict(self):
        """Serializes this instance to a JSON string."""
        return {"input_ids": self.input_ids, "chunks_len": self.chunks_len,
                "attention_mask": self.attention_mask, "token_type_ids": self.token_type_ids,
                "label": self.label}


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def convert_examples_2_features(self, examples: List[InputExample], bert_type: str, max_chunk_number: int,
                                    max_seq_len: int, set_type: str, ngram_path='data_process/sampled_dict.json'):
        tokenizer = dense_sent_tokenization.MultiTokenizer(bert_model_type=bert_type,
                                                           max_seq_len=max_seq_len // 2 - 2,
                                                           max_chunk_number=max_chunk_number // 2,
                                                           ngram_path=ngram_path)
        features = []
        if bert_type == 'albert':
            pad_token = '<pad>'
        else:
            pad_token = '[PAD]'
        label_map = {label: i for i, label in enumerate(self.get_labels())}
        for example in tqdm(examples):
            sentence1_tokens, sentence1_chunk_lens = tokenizer.tokenize(example.text_a)
            sentence2_tokens, sentence2_chunk_lens = tokenizer.tokenize(example.text_b)
            concated_tokens = ['[CLS]'] + sentence1_tokens + ['[SEP]'] + sentence2_tokens + ['[SEP]']
            concated_chunk_lens = [1] + sentence1_chunk_lens + [1] + sentence2_chunk_lens + [1]
            pad_tokens = [pad_token] * (max_seq_len - len(concated_tokens))
            pad_chunk_lens = [0] * (max_chunk_number - len(concated_chunk_lens))
            padded_tokens = concated_tokens + pad_tokens
            padded_chunk_lens = concated_chunk_lens + pad_chunk_lens
            token_type_ids = [0] * (len(sentence1_tokens) + 2) + [1] * (len(sentence2_tokens) + 1) \
                             + [0] * (max_seq_len - len(concated_tokens))
            padded_tokens_id = tokenizer.bert_tokenizer.convert_tokens_to_ids(padded_tokens)
            attention_mask = [1] * len(concated_tokens) + [0] * (max_seq_len - len(concated_tokens))
            label_id = None if set_type == "test" else label_map[example.label]
            if len(padded_tokens_id) != max_seq_len:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(len(padded_tokens_id))
            features.append(InputFeatures(input_ids=padded_tokens_id,
                                          chunks_len=padded_chunk_lens,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask, label=label_id))
        return features

    def make_feature_file(self, data_dir, set_type: str, output_dir, bert_type: str, max_chunk_number: int,
                          max_seq_len: int):
        examples = []
        if set_type == 'test':
            examples = self.get_test_examples(data_dir)
        if set_type == 'dev':
            examples = self.get_dev_examples(data_dir)
        if set_type == 'train':
            examples = self.get_train_examples(data_dir)
        features = self.convert_examples_2_features(examples, bert_type=bert_type, max_chunk_number=max_chunk_number,
                                                    max_seq_len=max_seq_len,
                                                    set_type=set_type)
        with open(output_dir, 'w') as file:
            for feature in features:
                json.dump(feature.to_dict(), file)
                file.write('\n')

    def get_train_features_from_examples(self, data_dir, bert_type, max_chunk_number, max_seq_len, ngram_path):
        examples = self.get_train_examples(data_dir)
        features = self.convert_examples_2_features(examples, bert_type=bert_type, max_chunk_number=max_chunk_number,
                                                    max_seq_len=max_seq_len,
                                                    set_type='train', ngram_path=ngram_path)
        return features

    def get_test_features_from_examples(self, data_dir, bert_type, max_chunk_number, max_seq_len, ngram_path):
        examples = self.get_test_examples(data_dir)
        features = self.convert_examples_2_features(examples, bert_type=bert_type, max_chunk_number=max_chunk_number,
                                                    max_seq_len=max_seq_len,
                                                    set_type='test', ngram_path=ngram_path)
        return features

    def get_dev_features_from_examples(self, data_dir, bert_type, max_chunk_number, max_seq_len, ngram_path):
        examples = self.get_dev_examples(data_dir)
        features = self.convert_examples_2_features(examples, bert_type=bert_type, max_chunk_number=max_chunk_number,
                                                    max_seq_len=max_seq_len,
                                                    set_type='dev', ngram_path=ngram_path)
        return features

    def get_eval_features_from_examples(self, data_dir, bert_type, max_chunk_number, max_seq_len, ngram_path):
        examples = self.get_dev_examples(data_dir)
        features = self.convert_examples_2_features(examples, bert_type=bert_type, max_chunk_number=max_chunk_number,
                                                    max_seq_len=max_seq_len,
                                                    set_type='dev', ngram_path=ngram_path)
        return features

    def get_features(self, data_dir):
        features = []
        with open(data_dir) as file:
            for line in file:
                features.append(json.loads(line))
        return features

    def get_train_features(self, data_dir):
        """See base class."""
        return self.get_features(os.path.join(data_dir, "train"))

    def get_dev_features(self, data_dir):
        """See base class."""
        return self.get_features(os.path.join(data_dir, "dev"))

    def get_test_features(self, data_dir):
        """See base class."""
        return self.get_features(os.path.join(data_dir, "test"))


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")

    def get_dev_features(self, data_dir):
        """See base class."""
        return self.get_features(os.path.join(data_dir, "dev_mismatched"))

    def get_test_features(self, data_dir):
        """See base class."""
        return self.get_features(os.path.join(data_dir, "test_mismatched"))


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        if test_mode:
            lines = lines[1:]
        text_index = 1 if test_mode else 3
        examples = []
        for (i, line) in enumerate(lines):
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(text_a=text_a, text_b=None, label=label))
        return examples

    def convert_examples_2_features(self, examples: List[InputExample], bert_type: str, max_chunk_number: int,
                                    max_seq_len: int, set_type: str, ngram_path):
        tokenizer = dense_sent_tokenization.MultiTokenizer(bert_model_type=bert_type,
                                                           max_seq_len=max_seq_len,
                                                           max_chunk_number=max_chunk_number, ngram_path=ngram_path)
        features = []
        if bert_type == 'albert':
            pad_token = '<pad>'
        else:
            pad_token = '[PAD]'
        for example in tqdm(examples):
            sentence1_tokens, sentence1_chunk_lens = tokenizer.tokenize(example.text_a)
            concated_tokens = ['[CLS]'] + sentence1_tokens + ['[SEP]']
            concated_chunk_lens = [1] + sentence1_chunk_lens + [1]
            pad_tokens = [pad_token] * (max_seq_len - len(concated_tokens))
            pad_chunk_lens = [0] * (max_chunk_number - len(concated_chunk_lens))
            padded_tokens = concated_tokens + pad_tokens
            padded_chunk_lens = concated_chunk_lens + pad_chunk_lens
            token_type_ids = [0] * max_seq_len
            padded_tokens_id = tokenizer.bert_tokenizer.convert_tokens_to_ids(padded_tokens)
            attention_mask = [1] * len(concated_tokens) + [0] * (max_seq_len - len(concated_tokens))
            label_id = None if set_type == "test" else int(example.label)
            features.append(InputFeatures(input_ids=padded_tokens_id,
                                          chunks_len=padded_chunk_lens,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask, label=label_id))
        return features


class SnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir) as file:
            for line in file:
                line = json.loads(line)
                if line['gold_label'] == '-':
                    # In the case of this unknown label, we will skip the whole datapoint
                    continue
                examples.append(
                    InputExample(text_a=line['sentence1'], text_b=line['sentence2'], label=line['gold_label']))
        return examples

    def convert_examples_2_features(self, examples: List[InputExample], bert_type: str, max_chunk_number: int,
                                    max_seq_len: int, set_type: str, ngram_path):
        if bert_type == 'albert':
            pad_token = '<pad>'
        else:
            pad_token = '[PAD]'
        tokenizer = dense_sent_tokenization.MultiTokenizer(bert_model_type=bert_type,
                                                           max_seq_len=max_seq_len,
                                                           max_chunk_number=max_chunk_number // 2,
                                                           ngram_path=ngram_path)
        features = []
        label_map = {label: i for i, label in enumerate(self.get_labels())}
        for example in tqdm(examples):
            sentence1_tokens, sentence1_chunk_lens = tokenizer.tokenize(example.text_a)
            sentence2_tokens, sentence2_chunk_lens = tokenizer.tokenize(example.text_b)
            concated_tokens = ['[CLS]'] + sentence1_tokens + ['[SEP]'] + sentence2_tokens + ['[SEP]']
            concated_chunk_lens = [1] + sentence1_chunk_lens + [1] + sentence2_chunk_lens + [1]
            pad_tokens = [pad_token] * (max_seq_len - len(concated_tokens))
            pad_chunk_lens = [0] * (max_chunk_number - len(concated_chunk_lens))
            padded_tokens = concated_tokens + pad_tokens
            padded_chunk_lens = concated_chunk_lens + pad_chunk_lens
            token_type_ids = [0] * (len(sentence1_tokens) + 2) + [1] * (len(sentence2_tokens) + 1) \
                             + [0] * (max_seq_len - len(concated_tokens))
            padded_tokens_id = tokenizer.bert_tokenizer.convert_tokens_to_ids(padded_tokens)
            attention_mask = [1] * len(concated_tokens) + [0] * (max_seq_len - len(concated_tokens))
            label_id = label_map[example.label]
            features.append(InputFeatures(input_ids=padded_tokens_id,
                                          chunks_len=padded_chunk_lens,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask, label=label_id))
        return features


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(text_a=text_a, text_b=None, label=label))
        return examples

    def convert_examples_2_features(self, examples: List[InputExample], bert_type: str, max_chunk_number: int,
                                    max_seq_len: int, set_type: str, ngram_path):
        tokenizer = dense_sent_tokenization.MultiTokenizer(bert_model_type=bert_type,
                                                           max_seq_len=max_seq_len,
                                                           max_chunk_number=max_chunk_number, ngram_path=ngram_path)
        features = []
        if bert_type == 'albert':
            pad_token = '<pad>'
        else:
            pad_token = '[PAD]'
        for example in tqdm(examples):
            sentence1_tokens, sentence1_chunk_lens = tokenizer.tokenize(example.text_a)
            concated_tokens = ['[CLS]'] + sentence1_tokens + ['[SEP]']
            concated_chunk_lens = [1] + sentence1_chunk_lens + [1]
            pad_tokens = [pad_token] * (max_seq_len - len(concated_tokens))
            pad_chunk_lens = [0] * (max_chunk_number - len(concated_chunk_lens))
            padded_tokens = concated_tokens + pad_tokens
            padded_chunk_lens = concated_chunk_lens + pad_chunk_lens
            token_type_ids = [0] * max_seq_len
            padded_tokens_id = tokenizer.bert_tokenizer.convert_tokens_to_ids(padded_tokens)
            attention_mask = [1] * len(concated_tokens) + [0] * (max_seq_len - len(concated_tokens))
            label_id = None if set_type == "test" else int(example.label)
            features.append(InputFeatures(input_ids=padded_tokens_id,
                                          chunks_len=padded_chunk_lens,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask, label=label_id))
        return features


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ['regrassion']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[7]
            text_b = line[8]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(text_a=text_a, text_b=text_b, label=label))
        return examples

    def convert_examples_2_features(self, examples: List[InputExample], bert_type: str, max_chunk_number: int,
                                    max_seq_len: int, set_type: str, ngram_path):
        tokenizer = dense_sent_tokenization.MultiTokenizer(bert_model_type=bert_type,
                                                           max_seq_len=max_seq_len // 2 - 2,
                                                           max_chunk_number=max_chunk_number // 2,
                                                           ngram_path=ngram_path)
        features = []
        if bert_type == 'albert':
            pad_token = '<pad>'
        else:
            pad_token = '[PAD]'
        for example in tqdm(examples):
            sentence1_tokens, sentence1_chunk_lens = tokenizer.tokenize(example.text_a)
            sentence2_tokens, sentence2_chunk_lens = tokenizer.tokenize(example.text_b)
            concated_tokens = ['[CLS]'] + sentence1_tokens + ['[SEP]'] + sentence2_tokens + ['[SEP]']
            concated_chunk_lens = [1] + sentence1_chunk_lens + [1] + sentence2_chunk_lens + [1]
            pad_tokens = [pad_token] * (max_seq_len - len(concated_tokens))
            pad_chunk_lens = [0] * (max_chunk_number - len(concated_chunk_lens))
            padded_tokens = concated_tokens + pad_tokens
            padded_chunk_lens = concated_chunk_lens + pad_chunk_lens
            token_type_ids = [0] * (len(sentence1_tokens) + 2) + [1] * (len(sentence2_tokens) + 1) \
                             + [0] * (max_seq_len - len(concated_tokens))
            padded_tokens_id = tokenizer.bert_tokenizer.convert_tokens_to_ids(padded_tokens)
            attention_mask = [1] * len(concated_tokens) + [0] * (max_seq_len - len(concated_tokens))
            label_id = None if set_type == "test" else float(example.label)
            if len(padded_tokens_id) != max_seq_len:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(len(padded_tokens_id))
            features.append(InputFeatures(input_ids=padded_tokens_id,
                                          chunks_len=padded_chunk_lens,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask, label=label_id))
        return features


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = None if test_mode else line[5]
            except IndexError:
                continue
            examples.append(InputExample(text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(text_a=text_a, text_b=text_b, label=label))
        return examples


def read_ner_file(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


@dataclass(frozen=True)
class NERFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    chunks_len: List[int]
    valid_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    label_mask: List[int] = None

    def to_dict(self):
        """Serializes this instance to a JSON string."""
        return {"input_ids": self.input_ids, "chunks_len": self.chunks_len,
                "attention_mask": self.attention_mask, "token_type_ids": self.token_type_ids,
                "label": self.label, "valid_ids": self.valid_ids, "label_mask": self.label_mask}


class NERProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["[PAD]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return read_ner_file(input_file)

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(text_a=text_a, text_b=text_b, label=label))
        return examples

    def convert_examples_2_features(self, examples: List[InputExample], bert_type: str, max_chunk_number: int,
                                    max_seq_len: int, set_type: str, ngram_path):
        label_map = {label: i for i, label in enumerate(self.get_labels())}
        tokenizer = dense_sent_tokenization.MultiTokenizer(bert_model_type=bert_type,
                                                           max_seq_len=max_seq_len,
                                                           max_chunk_number=max_chunk_number, ngram_path=ngram_path)
        features = []
        if bert_type == 'albert':
            pad_token = '<pad>'
        else:
            pad_token = '[PAD]'
        for example in tqdm(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            for i, word in enumerate(textlist):
                token = tokenizer.bert_tokenizer.tokenize(word)
                if len(tokens) + len(token) > max_seq_len - 2:
                    break
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                    else:
                        valid.append(0)
            valid.insert(0, 0)
            labels.insert(0, "[PAD]")
            valid.append(0)
            labels.append("[PAD]")
            valid = valid + [0] * (max_seq_len - len(valid))
            labels = labels + ["[PAD]"] * (max_seq_len - len(labels))
            label_ids = []
            label_mask = []
            for index, label in enumerate(labels):
                label_ids.append(label_map[label])
                if label == "[PAD]":
                    label_mask.append(0)
                else:
                    label_mask.append(1)
            sentence1_tokens, sentence1_chunk_lens = tokenizer.tokenize(example.text_a)
            concated_tokens = ['[CLS]'] + sentence1_tokens + ['[SEP]']
            concated_chunk_lens = [1] + sentence1_chunk_lens + [1]
            pad_tokens = [pad_token] * (max_seq_len - len(concated_tokens))
            pad_chunk_lens = [0] * (max_chunk_number - len(concated_chunk_lens))
            padded_tokens = concated_tokens + pad_tokens
            padded_chunk_lens = concated_chunk_lens + pad_chunk_lens
            token_type_ids = [0] * max_seq_len
            padded_tokens_id = tokenizer.bert_tokenizer.convert_tokens_to_ids(padded_tokens)
            attention_mask = [1] * len(concated_tokens) + [0] * (max_seq_len - len(concated_tokens))
            features.append(NERFeatures(input_ids=padded_tokens_id,
                                        chunks_len=padded_chunk_lens,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        label=label_ids,
                                        valid_ids=valid,
                                        label_mask=label_mask))
        return features


glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}
