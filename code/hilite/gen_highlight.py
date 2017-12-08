#!/usr/bin/env python3

import argparse
import os
from os.path import isfile, join, abspath, expanduser, basename, splitext

import matplotlib.pyplot as plt
import matplotlib.style as ms
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import scipy.io.wavfile
from ggplot import *
# from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, afx, transfx, CompositeVideoClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from scipy import fftpack
from sklearn.decomposition import PCA


def load_edit_list(event_filename):
    print('Loading file: ', event_filename)
    event_data = pd.read_csv(event_filename, sep=',', header=0)
    print('Events loaded: ', len(event_data))

    # Compute offset of each clip in the new clips collection and add to DataFrame
    offset = np.cumsum(event_data["timeToCapture"])
    event_data["offset"] = offset - event_data["timeToCapture"]
    # Prepare a list of labels
    labels = [row["eventName"]
              for index, row in event_data.iterrows()
              for i in np.arange(row["offset"], row["offset"] + row["timeToCapture"])]

    return event_data


def concat_clips(event_data, transitions=None, filenames=None, dir_out=".", dir_input=None):
    clip_list = []
    video_path = None

    # #clips1 = [ clip1.fadein(.5).fadeout(.5),
    # #          clip2.fadein(.5).fadeout(.5) ]
    # # concatenate is old, deprecated version of concatenate_videoclips
    # final_clip = concatenate(xclips)
    # final_clip.write_videofile('concat3.mp4')

    # fade_duration = 1 # 1-second fade-in for each clip
    for index, row in event_data.iterrows():
        # print row['secondOffset'], row['timeToCapture']

        # full path to filename of full game is in CSV file;
        # if dir_input is given, ignore path in csv file.
        fname = row['fileName']  # default
        video_path = join(dir_input, basename(fname)) if dir_input else fname

        if isfile(video_path):
            videoClip = VideoFileClip(video_path).subclip(row['secondOffset'],
                                                          row['secondOffset'] + row['timeToCapture'])
            # clip_list.append(videoClip.crossfadein(fade_duration))
            clip_list.append(videoClip)
        else:
            print('** Error: file not found: %s' % video_path)
            if dir_input:
                print('** Input directory overridden as: %s' % dir_input)
                print('** Default path (from edit list): %s' % fname)

    # last video_path from csv file
    if video_path:

        if transitions:
            xclips = [c.fadein(.25).fadeout(.25) for c in clip_list]
        else:
            xclips = clip_list

        num_highlights = len(clip_list)
        file_basename = splitext(basename(video_path))[0]
        final_clip_path = join(dir_out, file_basename + '_highlight_' + str(num_highlights) + '.mp4')

        # final_clip = concatenate_videoclips(clip_list, padding = fade_duration)
        final_clip = concatenate_videoclips(xclips)
        final_clip.write_videofile(final_clip_path)
    else:
        print("** Error: no clips found to create highlight matching patterns: %s",
              ['*' if not filenames else filenames])
        raise ValueError("No clips found to created highlight.")


def main():
    # sample usage:
    #   gen_highlight.py --edit_list foo.csv -dir_out  /path/to/output
    #   generates a highlight clip from edit list (original video filenames
    #   are in the csv) and creates a highlight video in given target directory.
    #   Optionally, directory of source video can be overridden from what is in
    #   the edit list.

    parser = argparse.ArgumentParser(
        description='Given an edit list, generate single highlight clip.')

    parser.add_argument('-i', '--dir_input', metavar='{dir}', type=str, default=None,
                        help='input directory of full videos (default: path from edit list)')

    parser.add_argument('-d', '--dir_out', metavar='{dir}', type=str, default='.',
                        help='output directory for final video (default: "."')

    parser.add_argument('-e', '--edit_list', metavar='{file}', type=str, default='clips.csv',
                        help='filename of clip edit list (default: clips.csv)')

    parser.add_argument('-s', '--save_edits', action='store_true',
                        help='save edit list (default: deleted afterwards) [not implemented]')

    parser.add_argument('-t', '--trans', metavar='{effect}', choices=['fade', 'none'], default='fade',
                        help='enable scene transitions between clips; options: "fade", "none"')

    parser.add_argument('files', metavar='{files}', nargs='*',
                        help='input video file names (regexp) in the edit list to use (default: uses all entries)')

    args = parser.parse_args()

    print("pwd: {}".format(os.getcwd()))
    print("args:", args)
    print("  args.dir_input  : %s " % args.dir_input)
    print("  args.dir_out    : %s " % args.dir_out)
    print("  args.edit_list  : %s " % args.edit_list)
    print("  args.save_edits : %s " % args.save_edits)
    print("  args.trans      : %s " % args.trans)
    # print("  args.dir_temp   : %s " % args.dir_temp)

    if not os.path.isdir(args.dir_out):
        print("** Error: target directory not found: '%s'" % args.dir_out)
        raise ValueError("Target directory not found: '%s'" % args.dir_out)

    if args.dir_input and not os.path.isdir(args.dir_input):
        print("** Error: video input directory not found: '%s'" % args.dir_input)
        raise ValueError("Video input directory not found: '%s'" % args.dir_input)

    if not os.path.exists(args.edit_list):
        print("** Error: edit list not found: '%s'" % args.edit_list)
        raise ValueError("Edit list not found: '%s'" % args.edit_list)

    concat_clips(load_edit_list(args.edit_list),
                 args.trans,
                 args.files,
                 args.dir_out,
                 args.dir_input)

    print("Done.")


if __name__ == '__main__':
    main()
