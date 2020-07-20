'''
https://blog.csdn.net/feilong_csdn/article/details/88655927
https://zhuanlan.zhihu.com/p/32965521
'''

'''$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
$ ./fasttext supervised
Empty input or output path.
The following arguments are mandatory:
-input training file path
-output output file path
The following arguments are optional:
-lr learning rate [0.05]
-dim size of word vectors [100]
-ws size of the context window [5]
-epoch number of epochs [5]
-minCount minimal number of word occurences [1]
-neg number of negatives sampled [5]
-wordNgrams max length of word ngram [1]
-loss loss function {ns, hs, softmax} [ns]
-bucket number of buckets [2000000]
-minn min length of char ngram [3]
-maxn max length of char ngram [6]
-thread number of threads [12]
-verbose how often to print to stdout [10000]
-t sampling threshold [0.0001]
-label labels prefix [__label__]
$ ./fasttext supervised -input training_file_path -output output_file_path
Read 3M words
Number of words:  846680
Number of labels: 311
Progress: 100.0%  words/sec/thread: 9815  lr: 0.000000  loss: 2.637867  eta: 0h0m
'''
