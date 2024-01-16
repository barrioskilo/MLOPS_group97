# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY pistachio/ pistachio/
#COPY data/ data/

RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc *.dvc
RUN dvc config core.no_scm true
RUN dvc pull


WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV PYTHONPATH pistachio/

# Pull data from DVC remote
# RUN dvc pull

# Install DVC with GCP support
RUN pip install 'dvc[gs]'

ENTRYPOINT ["python", "-u", "pistachio/src/models/lightning_train.py"]
CMD ["data/raw"]



