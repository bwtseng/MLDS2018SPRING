#!/bin/bash
wget 'https://www.dropbox.com/s/29qusoha28hxwzh/model.ckpt-290.meta?dl=1' -O'./model.ckpt-290.meta'
wget 'https://www.dropbox.com/s/j4wnhasz9e91y88/model.ckpt-290.index?dl=1' -O'./model.ckpt-290.index'
wget 'https://www.dropbox.com/s/toygsj8al6orlqx/model.ckpt-290.data-00000-of-00001?dl=1' -O'./model.ckpt-290.data-00000-of-00001'
wget 'https://www.dropbox.com/s/qf0i50ay3ljzxtg/vac.pickle?dl=1' -O'./vac.pickle'
python3 ./hw2_1.py $1 $2