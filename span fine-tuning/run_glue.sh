
cd /share03/zhangzs/Dialogue/LaSA/
export PROJECT_DIR=/share03/zhangzs/Dialogue/LaSA/

python download_glue_data.py --data_dir data_process/data/glue_data --tasks all

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
export CUDA_VISIBLE_DEVICES=`perl /share03/zhangzs/idle-gpus.pl -n 8`
source activate lm

python run_classifier_direct.py --task_name qnli --max_chunk_len 16 --max_chunk_number 128 --max_seq_len 256 --bert_type albert --hidden_size 4096 --do_train --do_eval --do_predict --data_dir data_process/data/glue_data/QNLI --chunk_encoder multi_cnn  --output_dir experiments_10_entropy_albertxxlarge/qnli --bert_model_path albert-xxlarge-v2 --train_batch_size 32 --eval_batch_size 8 --learning_rate 1e-5 --num_train_epochs 10 --dict_dir data_process/ngram_sample/sampled_dict_10_entropy --warmup_proportion 0.06

python run_classifier_direct.py --task_name mnli --max_chunk_len 16 --max_chunk_number 128 --max_seq_len 256 --bert_type albert --hidden_size 4096 --do_train --do_eval --do_predict --data_dir data_process/data/glue_data/MNLI --chunk_encoder multi_cnn  --output_dir experiments_10_entropy_albertxxlarge/mnli --bert_model_path albert-xxlarge-v2 --train_batch_size 64 --eval_batch_size 8 --learning_rate 3e-5 --num_train_epochs 3 --dict_dir data_process/ngram_sample/sampled_dict_10_entropy --warmup_proportion 0.1 --do_store

python run_classifier_direct.py --task_name qqp --max_chunk_len 16 --max_chunk_number 100 --max_seq_len 200 --bert_type albert --hidden_size 4096 --do_train --do_eval --do_predict --data_dir data_process/data/glue_data/QQP --chunk_encoder multi_cnn  --output_dir experiments_10_entropy_albertxxlarge/qqp --bert_model_path albert-xxlarge-v2 --train_batch_size 128 --eval_batch_size 8 --learning_rate 5e-5 --num_train_epochs 5 --dict_dir data_process/ngram_sample/sampled_dict_10_entropy --warmup_proportion 0.07

python run_classifier_direct.py --task_name sst-2 --max_chunk_len 16 --max_chunk_number 64 --max_seq_len 128 --bert_type albert --hidden_size 4096 --do_train --do_eval --do_predict --data_dir data_process/data/glue_data/SST-2 --chunk_encoder multi_cnn  --output_dir experiments_10_entropy_albertxxlarge/sst-2 --bert_model_path albert-xxlarge-v2 --train_batch_size 32 --eval_batch_size 8 --learning_rate 1e-5 --num_train_epochs 10 --dict_dir data_process/ngram_sample/sampled_dict_10_entropy --warmup_proportion 0.06

python run_classifier_direct.py --task_name snli --max_chunk_len 16 --max_chunk_number 128 --max_seq_len 256 --bert_type albert --hidden_size 4096 --do_train --do_eval --data_dir data_process/data/glue_data/SNLI --chunk_encoder multi_cnn  --output_dir experiments_10_entropy_albertxxlarge/snli --bert_model_path albert-xxlarge-v2 --train_batch_size 64 --eval_batch_size 8 --learning_rate 3e-5 --num_train_epochs 3 --dict_dir data_process/ngram_sample/sampled_dict_10_entropy --warmup_proportion 0.1
