# Use an official Python runtime as a parent image
FROM python:3.8-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential curl gcc unzip && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY pistachio/ pistachio/
COPY app/ app/


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

#ENV PYTHONPATH pistachio/
# Define environment variable
ENV PORT=8080

EXPOSE 8080
# Command to run the application
CMD ["/bin/sh", "-c" , "gsutil -m cp -r gs://mlops97_data_storage/model/transfer_learning_model.pth app/models/ && uvicorn pistachio_inference:app --host 0.0.0.0 --port 8080 --workers 1"]
