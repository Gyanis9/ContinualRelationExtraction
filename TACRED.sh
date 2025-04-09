# shellcheck disable=SC2155
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=$1 python3 main.py \
  --memory_size 10 \
  --total_rounds 5 \
  --task_name TACRED \
  --data_file data/data_with_marker_tacred.json \
  --relation_file data/id2rel_tacred.json \
  --num_of_train_samples 420 \
  --num_of_val_samples 140 \
  --num_of_test_samples 140 \
  --batch_size 4 \
  --num_of_relations 40  \
  --cache_file data/TACRED_data.pt \
  --relations_per_task 4 \
  --additional_classifier 1