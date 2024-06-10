FROM python:3.10

RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY . /code




RUN chmod +x /code/src

RUN pip install -r code/src/requirements.txt

EXPOSE 8080

WORKDIR /code/src


ENV PYTHONPATH "${PYTHONPATH}:/code/src"

CMD pip install -e .