FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
