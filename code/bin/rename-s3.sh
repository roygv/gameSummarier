#!/bin/bash


#  aws s3 mv "s3://game-summarizer/wba mci 1st half eng.mp4" "s3://game-summarizer/wba_mci_1st_half_eng.mp4"

tmp=list0.txt
[ -f "$tmp" ] && { echo "** tmp file exist: $tmp" ; exit 2; }
aws s3 ls s3://game-summarizer  | cut -c32- | egrep ' ' > "$tmp"
#aws s3 ls s3://game-summarizer  | cut -c32- | egrep ' ' | sed 's/^.*$/"&"/' > "$tmp"


s3="s3://game-summarizer/"
while IFS='' read -r file || [[ -n "$file" ]]; do
    echo "aws s3 mv \"${s3}${file}\" \"${s3}${file// /_}\""
done < $tmp

rm "$tmp"
