import ffmpy

path = "C:\\temp"
inputFile = "mal eng 1st half eng.mp4"
outputFile = inputFile.replace(".mp4", ".m4a")
ff = ffmpy.FFmpeg(
    inputs={path + '\\' + inputFile: None},
    outputs={path + '\\' + outputFile: '-vn -acodec copy'}
)

ff.run()
ffmpy
