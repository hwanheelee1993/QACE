python script/run_visual_seq2seq.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --task summarization \
    --train_file train_data.csv \
    --validation_file val_data.csv \
    --output_dir results \
    --overwrite_output_dir \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=128 \
    --predict_with_generate \
    --pad_to_max_length \
    --text_column inputs \
    --save_steps 5000 \
    --summary_column outputs
