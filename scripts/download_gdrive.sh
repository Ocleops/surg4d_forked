#!/bin/bash
fileid="1zTcX80c1yrbntY9c6-EK2W2UVESVEug8"
filename="endo_file.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
