# hw2_1   Chatbot

## Data
語音實驗室的電影字幕 500萬句對話  
[Dataset](https://drive.google.com/file/d/1JTwZ4RyEApT60v_CLHp7xAAfrirQQDEd/view) 

## Evaluation
baseline : Perplexity < 100  
[Perplexity code](https://drive.google.com/file/d/1kxWctXrq9thMh8on5-fL505JF5SPYXAT/view)  
baseline : Correlation Score > 0.45  
[Correlation code](https://drive.google.com/file/d/1OBvvTSL9PuFlr3KR0NIH0Z_okKhf_N37/view)  


## Usage
$1: input filename (format：.txt)  
$2: output filename (format：.txt)  
```
bash ./hw2_seq2seq.sh $1 $2
```

## package
`Python3.6` `tensorflow 1.6` `CUDA 9.0` `scipy` `pandas` `matplotlib` `json` `numpy` `pickle`
