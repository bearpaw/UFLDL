#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget --no-check-certificate http://ufldl.stanford.edu/wiki/resources/stlSubset.zip

echo "Unzipping..."

unzip stlSubset.zip

# Creation is split out because leveldb sometimes causes segfault
# and needs to be re-created.

echo "Done."
