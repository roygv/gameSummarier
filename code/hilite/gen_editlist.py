#!/usr/bin/env python3

import argparse
import glob
import os
from datetime import timedelta
from os.path import isfile, join, abspath, expanduser, basename, splitext
import pathlib

import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import scenedetect
import scipy.io.wavfile
from ggplot import *
from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from scipy import fftpack
from sklearn.decomposition import PCA
from sklearn.externals import joblib


def load_model(basedir='.'):
    def get_mpath(d, fname):
        mpath = join(d, fname)
        print("Loading model: ", mpath)
        return mpath

    gra_model = joblib.load(get_mpath(basedir, 'model_gra.pkl'))
    pca_model = joblib.load(get_mpath(basedir, 'model_pca.pkl'))
    # model = joblib.load('model_knn.pkl')

    return {'gra': gra_model, 'pca': pca_model}


def run_fft(video_filename):
    f = []
    # print row['secondOffset'], row['timeToCapture']

    videoClip = VideoFileClip(video_filename)
    fps = videoClip.fps
    audioClip = AudioFileClip(video_filename)
    samprate = audioClip.fps
    print(
        'Video duration: {0}, fps: {1}, audio sample rate {2}'.format(timedelta(seconds=round(audioClip.duration)), fps,
                                                                      samprate))
    wavClip = audioClip.to_soundarray(fps=samprate)
    wavClip = wavClip[:samprate * int(audioClip.duration), ]
    wavdata = wavClip.reshape(-1, samprate, 2)
    dims = wavdata.shape
    for sec in np.arange(dims[0]):
        ch1 = scipy.fftpack.fft(wavdata[sec, :, 0])[:samprate // 2]  # Left channel
        ch2 = scipy.fftpack.fft(wavdata[sec, :, 1])[:samprate // 2]  # Right channel
        ch = np.vstack([ch1, ch2])
        f.append(ch)
    #    del audioClip, videoClip

    f = np.absolute(f) / samprate
    f_db = 20 * np.log10(2 * f)  # Dimensions are (seconds, channels, samples)
    tmp_f = scipy.fftpack.fftfreq(samprate, 1.0 / samprate)[:samprate // 2]

    print("f_db: ", f_db.shape)
    f2 = f.reshape(f_db.shape[0], -1)
    print("f2: ", f2.shape)
    return f2, fps, videoClip


def predict_interest(f2, models):
    pca_model = models['pca']
    gra_model = models['gra']

    index = np.array(np.arange(f2.shape[0]))  # required by doPlotGame
    data = pca_model.transform(f2)
    N = 8  # Running average span
    print(data.shape)

    predicted_labels = gra_model.predict(data)
    predicted_proba = gra_model.predict_proba(data)

    # Calcualte running average
    # 0 means "goal", 1 is "near miss".
    running_predicted_labels = pd.rolling_mean(predicted_proba[:, 0] + predicted_proba[:, 1], N)

    # Assign the value to the first second.
    running_predicted_labels = np.concatenate((running_predicted_labels[N - 1:, ],
                                               np.array(
                                                   [np.average(predicted_proba[-i:, 0] + predicted_proba[-i:, 1]) for i
                                                    in np.arange(N - 1, 0, -1)])))

    print(predicted_proba.shape)
    print(running_predicted_labels.shape)
    return running_predicted_labels


def plotGame(index, g):
    import matplotlib.pyplot as plt
    import matplotlib.style as ms
    ms.use('seaborn-muted')

    # %matplotlib inline

    def on_plot_hover(event):
        for curve in plot.get_lines():
            if curve.contains(event)[0]:
                print("over %s" % curve.get_gid())

    fig = plt.figure(figsize=(15, 6))
    plot = plt.plot(index / 60, g, alpha=0.5)
    fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)
    plt.xlabel('Game Minutes')
    plt.ylabel('Goal Likelihood')
    plt.show()


def doPlotGame(index, running_predicted_labels):
    plotGame(index, running_predicted_labels)
    print("Total video length:", timedelta(seconds=len(running_predicted_labels)))


def distinctEvent(video_filename, running_predicted_labels, videoClip, fps, maxClipLength, tmpdir=None):
    from cv2 import ORB_create, xfeatures2d, imread

    indexed_running_predicted_labels = np.argsort(running_predicted_labels)
    eventsFounds = 0
    events = []
    gameMeanLikelihood = np.mean(running_predicted_labels)
    # for idx in reversed(indexed_running_predicted_labels[:len(running_predicted_labels)-(N-1)]):
    for idx in reversed(indexed_running_predicted_labels):
        found = False
        for e in events:
            if idx >= e["start"] and idx - 5 <= e["end"]:
                if found == True:
                    remember_e = e["end"]
                    remember_s = e["start"]
                    events.remove(e)
                    for e2 in events:
                        if idx >= e2["start"] and idx - 5 <= e2["end"]:
                            e2["start"] = min(e2["start"], remember_s)
                            e2["end"] = max(e2["end"], remember_e)
                else:
                    if idx > e["end"] and running_predicted_labels[idx] <= running_predicted_labels[e["end"]]:
                        e["end"] = idx
                    found = True
            elif idx + 1 == e["start"] and running_predicted_labels[idx] <= running_predicted_labels[e["start"]]:
                e["start"] = idx
                if found == True:
                    remember = e["end"]
                    events.remove(e)
                    for e2 in events:
                        if e2["end"] == idx:
                            e2["end"] = remember
                else:
                    found = True
            elif idx - 1 == e["end"] and running_predicted_labels[idx] <= running_predicted_labels[e["end"]]:
                e["end"] = idx
                if found == True:
                    remember = e["start"]
                    events.remove(e)
                    for e2 in events:
                        if e2["start"] == idx:
                            e2["start"] = remember
                else:
                    found = True
        if found == False:
            eventsFounds += 1
            event = {"peak": idx, "value": running_predicted_labels[idx], "start": idx, "end": idx}
            events.append(event)
        if running_predicted_labels[idx] <= gameMeanLikelihood:
            break

    # Prepare parameters for scene detection
    detector_list = [
        scenedetect.detectors.ContentDetector(threshold=30, min_scene_len=8),
        scenedetect.detectors.ThresholdDetector(threshold=70, min_percent=0.9)
    ]

    # Prepare objects for SURF (cound number of objects in a sample frame from each scene)
    orb = ORB_create()
    surf = xfeatures2d.SURF_create(hessianThreshold=10000, nOctaves=15, upright=1)
    clipTotal = 0
    for idx, event in enumerate(events[:10]):
        #    print("Peak:",timedelta(seconds=int(event['peak'])),"Value:",event['value'],chr(9),timedelta(seconds=int(event['start'])),'-',timedelta(seconds=int(event['end'])))
        #     if idx<=5: # temp for debug
        #         print("Peak:",timedelta(seconds=int(event['peak'])),"Value:",event['value'],chr(9),
        #           timedelta(seconds=int(event['start'])),'-',timedelta(seconds=int(event['end'])))
        #         continue
        # If this clip overlaps with a previous clip - skip it
        if len([1 for i in np.arange(idx) if event['start'] < events[i]['clipEndMs'] / 1000
                                             and event['end'] > events[i]['clipStartMs'] / 1000]) > 0:
            break
        scene_list = []
        video_framerate, frames_read = scenedetect.detect_scenes_file(
            video_filename, scene_list, detector_list, frame_skip=0, downscale_factor=3,
            timecode_list=[round((int(events[idx]['start']) - 30) * fps), round((int(events[idx]['end']) + 60) * fps),
                           -1])

        # create new list with scene boundaries in milliseconds instead of frame #
        scene_list_msec = sorted([(1000 * x) / float(video_framerate) for x in scene_list])
        events[idx]["clipStartMs"] = int(event['start']) * 1000
        events[idx]["clipEndMs"] = int(event['end']) * 1000
        for sceneIdx, scene in enumerate(scene_list_msec[:-1]):
            if event['peak'] - scene / 1000 > 30:
                events[idx]["clipStartMs"] = scene_list_msec[sceneIdx]
            else:
                events[idx]["clipEndMs"] = scene_list_msec[sceneIdx]

            # probably want a flag that disables retention of tmp image files [ToDo]
            if not tmpdir: tmpdir='.'
            dPath = join(tmpdir, splitext(basename(video_filename))[0])
            if not os.path.exists(dPath):
                print("creating temp directory for image files: '%s'" % dPath)
                pathlib.Path(dPath).mkdir(parents=True, exist_ok=True)

            imgPath = join(dPath, "event{:d}-frame{:d}.jpg".format(idx, sceneIdx))
            videoClip.save_frame(imgPath, t=scene / 1000.0 + 0.25)
            img = imread(imgPath, 0)
            kp, des = surf.detectAndCompute(img, None)

            # Make sure all the original clip is in, end with a scene with manu objects (replay?) after audio had settled
            if scene / 1000 > event['end'] and len(kp) > 100 and running_predicted_labels[
                event['end']] <= gameMeanLikelihood:
                break
            # If this scene gets us past maxClipLength then break
            if clipTotal + int(event['clipEndMs'] - event['clipStartMs']) / 1000 >= maxClipLength:
                break
        print("Peak:", timedelta(seconds=int(event['peak'])), "Value:", event['value'], "Scenes:", len(scene_list_msec),
              chr(9),
              timedelta(seconds=int(event['start'])), '-', timedelta(seconds=int(event['end'])), chr(9),
              timedelta(seconds=int(event['clipStartMs'] / 1000)), '-',
              timedelta(seconds=int(event['clipEndMs'] / 1000)),
              )
        clipTotal += int(event['clipEndMs'] - event['clipStartMs']) / 1000
        if clipTotal >= maxClipLength:
            break

    return events


def writeSummaryClipsIndexToCSV(events, video_filename, dir_edits, csv_filename):
    secondOffset = []
    timeToCapture = []
    # video_basename = basename(splitext(csv_filename)[0])

    for i in np.arange(10):
        if 'clipStartMs' in events[i]:
            secondOffset.append(events[i]["clipStartMs"] / 1000)
            timeToCapture.append((events[i]["clipEndMs"] - events[i]["clipStartMs"]) / 1000)
            # df.append({video_filename,events[i]["clipStartMs"]/1000,,(events[i]["clipEndMs"]-events[i]["clipStartMs"])/1000})
            print(events[i])
    df = pd.DataFrame({'fileName': video_filename,
                       'secondOffset': secondOffset,
                       'eventName': 'goal',
                       'timeToCapture': timeToCapture},
                      columns=['fileName',
                               'secondOffset',
                               'eventName',
                               'timeToCapture'])
    df.sort_index(axis=0, by='secondOffset', inplace=True)
    csv_filename = join(dir_edits, csv_filename)
    print("Writing edit list CSV to: %s", csv_filename)
    df.to_csv(csv_filename, index=False)


# ignore
def sceneDetect(video_filename, events, fps):
    # scenedetect -i mag_dor_1st_half_eng.mp4 -d content -t 28 -st 152s -et 231s -co scenes.csv -fs 1 -p 85 -df 3 -q

    eventIdx = 8
    scene_list = []  # Scenes will be added to this list in detect_scenes().

    # Usually use one detector, but multiple can be used.
    # scenedetect.detectors.ThresholdDetector(threshold = 8, min_percent = 0.85)
    detector_list = [
        scenedetect.detectors.ContentDetector(threshold=30, min_scene_len=8),
        scenedetect.detectors.ThresholdDetector(threshold=70, min_percent=0.9)
    ]

    # video_filename='/Users/Roy/Downloads/videos/MON_V_MONT_1H_ESP_highlight.mp4'
    video_framerate, frames_read = scenedetect.detect_scenes_file(
        #    video_filename, scene_list, detector_list, frame_skip = 3, downscale_factor = 3, timecode_list=[112*fps,172*fps,-1])
        video_filename, scene_list, detector_list, frame_skip=0, downscale_factor=3,
        timecode_list=[round((int(events[eventIdx]['start']) - 30) * fps),
                       round((int(events[eventIdx]['end']) + 90) * fps), -1])

    # scene_list now contains the frame numbers of scene boundaries.
    print([s / video_framerate for s in scene_list])

    # create new list with scene boundaries in milliseconds instead of frame #.
    scene_list_msec = sorted([(1000.0 * x) / float(video_framerate) for x in scene_list])

    # create new list with scene boundaries in timecode strings ("HH:MM:SS.nnn").
    scene_list_tc = [scenedetect.timecodes.get_string(x) for x in scene_list_msec]
    print(scene_list_tc)
    print("# scenes: {0}".format(len(scene_list_tc)))


def main():
    # sample usage:
    #   highlight.py --max_time 180 --max_clips 5  video-1.mp4 video-2.mp4 video-3.mp4
    #   creates an (approximately) 3-minute (180 second) clip with at
    #   most 5 highlights, from the 3 given videos.
    #   Specifing "max" and "min" clips to the same number is
    #   equivalent to setting "--num_clips".

    parser = argparse.ArgumentParser(
        description='Given full-length game video, generate single highlight clip.')

    parser.add_argument('-i', '--dir_input', metavar='{dir}', type=str, default='.',
                        help='input directory containing full game videos')

    parser.add_argument('--dir_edits', metavar='{dir}', type=str, default=None,
                        help='output directory for edit lists (default: {dir_input})')

    parser.add_argument('--dir_temp', metavar='{dir}', type=str, default=None,
                        help='output directory for temp jpg images (default: {dir_edits})')

    parser.add_argument('--save_clips', action='store_true',
                        help='save the generated clips used to make final video [not implemented]')

    parser.add_argument('--max_time', metavar='{seconds}', type=int, nargs=1, default=180,
                        help='max total time of final highlight video (in seconds) [not implemented]')

    parser.add_argument('--num_clips', metavar='N', type=int, default=None,
                        help='exact number of highlights to include in edit list [not implemented]')

    parser.add_argument('--max_clips', metavar='N', type=int, default=10,
                        help='maximum number of clips to include in edit list)')

    parser.add_argument('--min_clips', metavar='N', type=int, default=3,
                        help='minimum number of clips to include in edit list)')

    parser.add_argument('-m', '--dir_model', metavar='{dir}', type=str, default='.',
                        help='directory containing models (model*.pkl) (default: "."')

    parser.add_argument('files', metavar='{files}', nargs='+',
                        help='input file names of full-length video games to process')

    args = parser.parse_args()

    # for most args, just use args.{option}
    dir_edits = args.dir_edits if (args.dir_edits) else args.dir_input
    dir_temp = args.dir_temp if (args.dir_temp) else args.dir_edits

    print("pwd: {}".format(os.getcwd()))
    print("args:", args)
    print("  args.dir_input  : %s " % args.dir_input)
    print("  args.dir_edits  : %s " % args.dir_edits)
    print("  args.dir_temp   : %s " % args.dir_temp)
    print("  args.save_clips : %s " % args.save_clips)
    print("  args.max_time   : %s " % args.max_time)
    print("  args.num_clips  : %s " % args.num_clips)
    print("  args.max_clips  : %s " % args.max_clips)
    print("  args.min_clips  : %s " % args.min_clips)
    print("  args.dir_model  : %s " % args.dir_model)

    print("using:")
    print("  dir_edits  : %s " % dir_edits)
    print("  dir_temp   : %s " % dir_temp)

    for d in [dir_edits, args.dir_input, dir_temp]:
        if not os.path.isdir(d):
            print("** Error: directory does not exist: %s" % d)
            raise ValueError("Directory does not exist: %s" % d)

    video_files = []
    missing_files = []
    existing_files = []
    for arg in args.files:
        # print("\n====== arg: {}".format(arg))
        # merge glob w/ original string, in case glob returns none (if the filename has special chars)
        for filename in sorted(set(list(glob.glob(arg)) + [arg])):
            print("=== file: ", filename)
            video_path = join(args.dir_input, filename)
            if os.path.exists(video_path):
                csv_filename = basename(video_path).replace('.mp4', '.csv')
                highlight_filename = basename(video_path).replace('.mp4', '_highlight.mp4')
                # tuple: (path/to/video, basename csv, basename highlight)
                video_files.append((video_path, csv_filename, highlight_filename))
                print('video (input): "%s"' % video_path)
                print('    edit list (output): "%s"' % csv_filename)
                print('    highlight (output): "%s"' % highlight_filename)

                if os.path.exists(csv_filename):
                    print("** Error: file exists: %s" % csv_filename)
                    existing_files.append(csv_filename)
            else:
                missing_files.append(video_path)
                print("** Error: file not found: %s" % video_path)

    if missing_files:
        raise ValueError("Input file not found: %s" % missing_files)

    if existing_files:
        raise ValueError("Output file exists: %s" % existing_files)

    models = load_model(args.dir_model)
    for video_file, csv_file, highlight_file in video_files:
        print("============ processing: %s" % video_file)
        fft2, fps, videoClip = run_fft(video_file)
        running_predicted_labels = predict_interest(fft2, models)
        # doPlotGame(index, running_predicted_labels)
        # temp jpg files are written to dir_edits or dir_temp, if set
        # def distinctEvent(video_filename, running_predicted_labels, videoClip, fps, maxClipLength, dir_temp):
        events = distinctEvent(video_file, running_predicted_labels, videoClip, fps, args.max_time, dir_temp)
        writeSummaryClipsIndexToCSV(events, video_file, dir_edits, csv_file)

    print("============(done).")


if __name__ == '__main__':
    main()
