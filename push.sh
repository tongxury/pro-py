#! /bin/bash

docker buildx build --platform=linux/amd64 -t usernx/voiceagent-py:1.0.1 . --push   