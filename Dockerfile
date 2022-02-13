FROM python:3.7

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

ADD src src

COPY main.py .

CMD python main.py
