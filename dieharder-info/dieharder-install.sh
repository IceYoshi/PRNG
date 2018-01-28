#!/bin/sh

mkdir ~/stow

wget https://webhome.phy.duke.edu/~rgb/General/dieharder/dieharder-3.31.1.tgz
tar xvzf dieharder-3.31.1.tgz
cd dieharder-3.31.1

module load toolchain/foss/2017a
module load numlib/GSL/2.3-foss-2017a

./configure --prefix=$HOME/stow/dieharder-3.31.1

make
make install

cd ~/stow
stow dieharder-3.31.1/