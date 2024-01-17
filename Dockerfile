# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc unzip && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY pistachio/ pistachio/
#COPY data/ data/


WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV PYTHONPATH pistachio/

# Pull data from DVC remote
# RUN dvc pull



ENTRYPOINT ["python", "-u", "pistachio/src/models/lightning_train.py"]
CMD ["gs://mlops97_data_storage/data/data/raw"]



