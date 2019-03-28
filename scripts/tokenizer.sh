cd `dirname $0`
. var

SRC=ja
TGT=en

# [word, bpe, unigram, char]
TYPE=bpe

# train data
python $utils_dir/tokenizer.py --input-path $data_dir/raw \
                              --input-file train.$SRC \
                              --output-path $data_dir/tokenized/$TYPE \
                              --type $TYPE \
                              --vocab-size 15000 \
                              --lang $SRC

python $utils_dir/tokenizer.py --input-path $data_dir/raw \
                              --input-file train.$TGT \
                              --output-path $data_dir/tokenized/$TYPE \
                              --type $TYPE \
                              --vocab-size 15000 \
                              --lang $TGT

# dev data
python $utils_dir/tokenizer.py --input-path $data_dir/raw \
                              --input-file dev.$SRC \
                              --output-path $data_dir/tokenized/$TYPE \
                              --type $TYPE \
                              --model train.$SRC.$TYPE.model

python $utils_dir/tokenizer.py --input-path $data_dir/raw \
                              --input-file dev.$TGT \
                              --output-path $data_dir/tokenized/$TYPE \
                              --type $TYPE \
                              --model train.$TGT.$TYPE.model

# test data
python $utils_dir/tokenizer.py --input-path $data_dir/raw \
                              --input-file test.$SRC \
                              --output-path $data_dir/tokenized/$TYPE \
                              --type $TYPE \
                              --model train.$SRC.$TYPE.model

python $utils_dir/tokenizer.py --input-path $data_dir/raw \
                              --input-file test.$TGT \
                              --output-path $data_dir/tokenized/$TYPE \
                              --type $TYPE \
                              --model train.$TGT.$TYPE.model
