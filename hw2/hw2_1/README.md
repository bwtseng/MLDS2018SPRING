# hw2_1   Video caption generation

## Data
1450 videos for training  
100 videos for testing  
[MLDS_hw2_1_data.tar.gz](https://drive.google.com/file/d/1zDzkDpN0fXf1gBclH0yqLWaKat1AVfo7/view) 

## Evaluation
baseline : BLEU@1(new) >= 0.6  
[BLEU evaluation ](https://aclanthology.info/pdf/P/P02/P02-1040.pdf) 

## Usage
$1: the data directory  
$2: test data output filename (format：.txt)  
```
bash ./hw2_seq2seq.sh $1 $2
```

## package
`Python3.6` `tensorflow 1.6` `CUDA 9.0` `scipy` `pandas` `matplotlib` `json` `numpy` `pickle`
