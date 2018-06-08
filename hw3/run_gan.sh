#!/bin/bash
wget 'https://www.dropbox.com/s/x1udmeka6paoeik/dc_23.data-00000-of-00001?dl=1' -O'dc_23.data-00000-of-00001'
wget 'https://www.dropbox.com/s/oc30tdropg5wpux/dc_23.index?dl=1' -O'dc_23.index'
wget 'https://www.dropbox.com/s/qfnp6s8bhipzqxw/dc_23.meta?dl=1' -O'dc_23.meta'
wget 'https://www.dropbox.com/s/2oh7kvajp04k957/dc_no_20.pickle?dl=1' -O'dc_no_20.pickle'
python3 DC_GAN.py