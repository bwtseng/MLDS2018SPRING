#!/bin/bash
wget 'https://www.dropbox.com/s/mfi71e4sssgph2d/model.ckpt-14.meta?dl=1' -O'./model.ckpt-14.meta'
wget 'https://www.dropbox.com/s/egt4n191mpi89d1/model.ckpt-14.index?dl=1' -O'./model.ckpt-14.index'
wget 'https://www.dropbox.com/s/o49u4vx6r4ks0q9/model.ckpt-14.data-00000-of-00001?dl=1' -O'./model.ckpt-14.data-00000-of-00001'
wget 'https://www.dropbox.com/s/hdytk2d3aiqnw9t/dic.pickle?dl=1' -O'./dic.pickle'
python3 ./hw2_2.py $1 $2