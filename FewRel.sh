

# shellcheck disable=SC2155
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$1 python3 main.py\
  --memory_size 10 \
  --total_rounds 5 \
  --task_name FewRel \
  --data_file data/data_with_marker.json \
  --relation_file data/id2rel.json \
  --num_of_train_samples 420 \
  --num_of_val_samples 140 \
  --num_of_test_samples 140 \
  --batch_size 16 \
  --num_of_relations 80  \
  --cache_file data/fewrel_data.pt \
  --relations_per_task 8 \
  --additional_classifier 1

