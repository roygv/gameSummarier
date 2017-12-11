#!/bin/bash

# You can specify number of channels, etc. as well, ex:
# ffmpeg -f u16le -ar 44100 -ac 1 -i input.raw output.wav

# The default for muxing into WAV files is pcm_s16le.
# You can change it by specifying the audio codec and using the WAV file extension:
# ffmpeg -i input -c:a pcm_s32le output.wav

# ffmpeg -f pcm_s16le -ar 44100 -ac 1 -i input.raw output.wav


# resample the audio on the fly using the "-ar" option,
# to set the bit rate and thus the quality of the resulting file with the "-ab" option,
# and to downmix stereo to mono with the "-ac" option.

output_dir=reduced
RUN=
pTIME='/usr/bin/time'
export TIME='%E real, %K mem, %M max, %t sz'
pFFMPEG='ffmpeg -hide_banner'

gen_audio() {
  for vid; do
      #base=$(basename "${vid}")
      suffix=${vid##*.}
      base=$(basename "${vid// /-}" ".${suffix}")

      a_out1="$output_dir/${base}.wav"
      a_out2="$output_dir/${base}.wav"
      v_out="$output_dir/${base}.mp4"
      foo="${base}--${sz}${crf/ /}${fps/ /}${vf//[\/ ]/_}${fmeta}.${suffix}"

      printf "\n#######################################\n"
      printf "###  Input: $vid\n"
      #ffprobe -hide_banner "$vid" 2>&1 | sed 's/^/#  /'

      printf "\n###  Output: $v_out\n"

      if [ -f "$v_out" ]; then
          echo "## ** WARNING: output file exists, skipping: $v_out"
      else
          ## gen audio file only
          #$RUN $pTIME $pFFMPEG -i "$vid" -vn -acodec copy "$a_out"
          #$RUN $pTIME $pFFMPEG -i "$vid" -acodec pcm_s16le -ac 1 -ar 22050 -ab 64k "$a_out1" 2>&1

          ## gen reduced video + updated audio
          #$RUN $pTIME $pFFMPEG -i "$vid"  \
          #    -c:a aac -ac 1 -ar 22050 -ab 64k \
          #    -c:v libx264 -r 5 -vf hue=s=0 -s 640x360 -movflags faststart -write_tmcd on "$v_out" 2>&1

          $RUN $pTIME $pFFMPEG -i "$vid"  \
              -c:a aac -ac 1 -ar 44100 -ab 64k \
              -c:v libx264 -r 15 -vf hue=s=0 -s 640x360 -crf 28 -b:v 2M -maxrate 2M -bufsize 1M -movflags faststart "$v_out" 2>&1

          # -c:v libx264 -r 15 -vf hue=s=0 -s 640x360 -crf 28 -b:v 2M -maxrate 2M -bufsize 1M -movflags faststart -write_tmcd on "$v_out" 2>&1
          # -movflags +faststart  # runs second pass, takes longer; faststart for web video.
          # for if videos are going to be viewed in a browser. This will move some information
          # to the beginning of your file and allow the video to begin playing before it is
          # completely downloaded by the viewer. Not required for a video service such as YouTube.
      fi

      printf "\n"
  done
}

[ "$1" = "-n" ] && { RUN=echo; shift; echo "## ** WARNING: dry-run, not generating output"; }
[ "$1" = "-h" ] && { printf "\n** USAGE: $0 video [video...]\n**   generate reduced audio/video, writes to: \"$output_dir\"\n\n"; exit 2; }

mkdir -p "$output_dir"
gen_audio "$@" 2>&1 | tee output-audio-convert.txt


