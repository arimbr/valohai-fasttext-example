FROM python:3.6

RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  python-dev \
  python-numpy \
  python-scipy

RUN git clone https://github.com/facebookresearch/fastText.git /tmp/fastText && \
  rm -rf /tmp/fastText/.git* && \
  cd /tmp/fastText && \
  make

COPY requirements.txt .

RUN pip3 install -r requirements.txt

WORKDIR /
