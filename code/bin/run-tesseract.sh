#!/bin/bash

filelist=list-chyron.$$.txt

[ -f "$filelist" ] && { printf "** error: temp file exists: $filelist\n\n"; exit 2; }
[ $# -eq 0 ] && { printf "** error: expecting chyrons.\n** usage: $0 [chyron1.tif chyron1.tif...]\n\n"; exit 2; }

for image; do
  [ ! -f "$image" ] && echo "** error: image not found: $image" && exit 2
  ls "$image"
done >>  "$filelist"

oneline(){
  tr '\n' ' ' | sed 's/  */ /g'
}

tesseract -psm 7 -c include_page_breaks=1 -c page_separator='' "$filelist" stdout | oneline
printf '\n\n'

#tesseract -psm 7 -c include_page_breaks=1 "$filelist" stdout

[ -f "$filelist" ] && rm -f "$filelist"

