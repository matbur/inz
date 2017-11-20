#!/usr/bin/env bash

docker rm inz
git pull
docker build -t inz .
docker run --name inz -d -v $PWD:/code inz python main.py