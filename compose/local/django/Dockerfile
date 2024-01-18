FROM python:3.10
RUN apt-get update -y

# Install graphviz
RUN apt-get install -y graphviz

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

COPY . /app

WORKDIR /app

COPY ./compose/local/django/start /start
RUN sed -i 's/\r//' /start
RUN chmod +x /start

COPY ./compose/local/django/start_scheduler /start_scheduler
RUN sed -i 's/\r//' /start_scheduler
RUN chmod +x /start_scheduler

ENTRYPOINT ["/start"]
