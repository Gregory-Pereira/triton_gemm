# FROM registry.access.redhat.com/ubi9/python-311:1-77.1729767833
FROM nvidia/cuda:12.6.2-cudnn-devel-ubi9

USER root

RUN mkdir -p /opt/app-root/src

WORKDIR /opt/app-root/src

ADD src /opt/app-root/src

RUN dnf install -y python3.11 python3.11-pip python3.11-devel nano

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python -m pip install -r /opt/app-root/src/requirements.txt

ENTRYPOINT ["python", "main.py"]
