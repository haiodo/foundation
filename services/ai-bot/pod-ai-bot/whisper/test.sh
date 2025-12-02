#!/bin/bash
curl -X POST -H "content-type: multipart/form-data" -F "audio_file=@./tests/1764653810474.wav" "127.0.0.1:9007/asr?output=json&language=ru&encode=false"
