@echo off

call python positive.py
echo Generated positives.
call python negative.py
echo Generated negatives.
call C:\Users\wuwsh\opencv\build\x64\vc15\bin\opencv_createsamples.exe -info pos.txt -w 32 -h 32 -num 1000 -vec pos.vec
echo Created Samples.
call C:\Users\wuwsh\opencv\build\x64\vc15\bin\opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -precalcValBufSize 6000 -precalcIdxBufSize 6000 -numPos 200 -numNeg 1000 -numStages 12 -w 32 -h 32 -maxFalseAlarmRate 0.4 -minHitRate 0.999
echo Done.