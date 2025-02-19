
ModelPath=/mnt/danlongyuan/resource/models/huggingface/meta-llama/Meta-Llama-3.1-8B

tokenizer=/mnt/danlongyuan/resource/models/huggingface/meta-llama/Meta-Llama-3.1-8B

Name=VQVAE-Llama3.1-8B-MetaMath
# Name=test

accelerate launch --config_file ./sh/acc8.yaml  ./train.py \
--name $Name \
--logging_dir /mnt/danlongyuan/ShortR1/records/log \
--output_dir /mnt/danlongyuan/ShortR1/records/out/$Name \
--dataset_name /mnt/danlongyuan/resource/dataset/MetaMath \
--model_name_or_path $ModelPath \
--tokenizer $tokenizer \
--K 1024 \
--latent_step 16 \
--hidden_size 2048 \
--num_layers 2 \
--base_model_type llama \
--beta 1.1 \
--learning_rate 2e-5 \
--per_device_train_batch_size 32 \
--warmup_steps 100 \
--num_train_epochs 1 \
--gradient_accumulation_steps 1 \
--logging_steps 16 \
--save_steps 16 \
--print_div 10 \
--seed 42 \
--embedding_size 512 \
--attention_implement flash_attention_2
