#!/bin/bash

# initial check

if [ "$#" != 1 ]; then
    echo "$# parameters given. Only 1 expected. Use -h to view command format"
    exit 1
fi

if [ "$1" == "-h" ]; then
  echo "Usage: $(basename "$0") [file to evaluate upon]"
  exit 1
fi

test_path=$1

# delete old docker if exists
docker ps -q --filter "name=nlp2023-hw3" | grep -q . && docker stop nlp2023-hw3
docker ps -aq --filter "name=nlp2023-hw3" | grep -q . && docker rm nlp2023-hw3

# build docker file
docker build . -f Dockerfile -t nlp2023-hw3

# bring model up
docker run -d -p 12345:12345 --name nlp2023-hw3 nlp2023-hw3

# perform evaluation
/usr/bin/env python hw3/evaluate.py "$test_path"

# stop container
docker stop nlp2023-hw3

# dump container logs
docker logs -t nlp2023-hw3 > logs/server.stdout 2> logs/server.stderr

# remove container
docker rm nlp2023-hw3