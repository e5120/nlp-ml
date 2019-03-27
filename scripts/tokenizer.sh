cd `dirname $0`
. var

# [word, bpe, unigram, char]
TYPE=unigram

python $utils_dir/tokenizer.py --input-path $data_dir/raw \
                              --input-file en \
                              --output-path $data_dir/tokenized/$TYPE \
                              --type $TYPE \
                              --vocab-size 15000 \
                              --lang en

python $utils_dir/tokenizer.py --input-path $data_dir/raw \
                              --input-file ja \
                              --output-path $data_dir/tokenized/$TYPE \
                              --type $TYPE \
                              --vocab-size 15000 \
                              --lang ja
