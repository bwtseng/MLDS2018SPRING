#!/bin/bash
wget 'https://www.dropbox.com/s/furilyuutbi7a3s/v3.data-00000-of-00001?dl=1' -O'v3.data-00000-of-00001'
wget 'https://www.dropbox.com/s/texoxa9mmnpecth/v3.index?dl=1' -O'v3.index'
wget 'https://www.dropbox.com/s/96r4uzyzmind1or/v3.meta?dl=1' -O'v3.meta'
wget 'https://www.dropbox.com/s/cor6rkdieeuc9fr/cwgan_no_TA.pickle?dl=1' -O'cwgan_no_TA.pickle'
python3 wgan_mod.py $1