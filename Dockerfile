FROM python:3.7.4

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# RUN mkdir src

# COPY src src

# COPY main.py main.py

# ENTRYPOINT ["python", "main.py"]

