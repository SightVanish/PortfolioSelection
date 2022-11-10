FROM ubuntu:20.04
LABEL maintainer="wuchen"

RUN apt-get update -y && apt-get install -y python3-pip python3-dev
WORKDIR /app
COPY ./requirements.txt /app/
COPY ./test.py /app/
RUN python3 -m pip install -r ./requirements.txt

ENTRYPOINT ["python3"]

CMD ["/app/test.py"]
