#!/bin/bash

: ${S3DATA_DIR:="$HOME/Downloads/videos/s3data/"} 
: ${TRAIN_CSV:="trainingData.csv"}

s3_files() {
  find $S3DATA_DIR -name "*.mp4" -printf "%f\n" | sort
}

csv_files() {
  egrep -iv '^#|^ *filename' $TRAIN_CSV | cut -f1 -d, | sort -u
}

print_header() {
  printf "\n=================================================\n"
  printf "=== ONLY IN $@\n"
  printf "=================================================\n\n"
}

# -1     suppress column 1 (lines unique to FILE1)
# -2     suppress column 2 (lines unique to FILE2)
# -3     suppress column 3 (lines that appear in both files)

print_header "$TRAIN_CSV"
comm -23 --check-order <( csv_files ) <( s3_files )

print_header "$S3DATA_DIR"
comm -13 --check-order <( csv_files ) <( s3_files )

