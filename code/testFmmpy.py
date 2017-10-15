import ffmpy

path = "C:\\users\\roy\\Downloads"
inputFile = "West Ham United vs Tottenham 2017-09-23 2nd half ENG.mp4"
outputFile = inputFile.replace(".mp4", ".m4a")
ff = ffmpy.FFmpeg(
    inputs={path + '\\' + inputFile: None},
    outputs={path + '\\' + outputFile: '-vn -acodec copy'}
)

ff.run()
