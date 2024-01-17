# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential curl gcc unzip && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY pistachio/ pistachio/
#COPY data/ data/

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin


WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV PYTHONPATH pistachio/

RUN gsutil -m cp -r gs://mlops97_data_storage/data/ .

# Pull data from DVC remote
# RUN dvc pull

ENTRYPOINT ["python", "-u", "pistachio/src/models/lightning_train.py", "/data/data/raw" "&&", "gsutil", "-m", "cp", "-r", "pistachio/models/pistachio_model.pth", "gs://mlops97_data_storage/model"]



