FROM python:3.10
RUN apt-get update -y

# Install graphviz
RUN apt-get install -y graphviz

COPY . /app
RUN pip3 install -r /app/requirements.txt

WORKDIR /app

COPY ./compose/local/django/start /start
RUN sed -i 's/\r//' /start
RUN chmod +x /start

ENTRYPOINT ["/start"]
