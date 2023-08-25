FROM tensorflow/tensorflow:nightly-gpu-jupyter
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
STOPSIGNAL SIGINT
WORKDIR /tf