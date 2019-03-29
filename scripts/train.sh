cd `dirname $0`
. var

python $src_dir/train.py \
      --src $data_dir/train.problem \
      --tgt $data_dir/train.answer \
      --src-dict $data_dir/train.problem.vocab.json \
      --tgt-dict $data_dir/train.answer.vocab.json \
      --src-valid $data_dir/dev.problem \
      --tgt-valid $data_dir/dev.answer \
      --train-steps 1000
