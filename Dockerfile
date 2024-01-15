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
COPY data/ data/


WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV PYTHONPATH pistachio/

RUN python pistachio/src/data/make_dataset.py data/raw data/processed/processed_data.pt

ENTRYPOINT ["python", "-u", "pistachio/src/models/train_model.py"]
CMD ["train", "--lr", "1e-4"]