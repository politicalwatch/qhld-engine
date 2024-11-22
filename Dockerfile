FROM python:3.10-slim

RUN apt-get update && apt-get install -y git gcc cron poppler-utils tesseract-ocr tesseract-ocr-spa tesseract-ocr-cat antiword
RUN pip install pip==24.0

COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

COPY . /app

RUN touch /var/log/cron.log

CMD /etc/init.d/cron start && tail -f /var/log/cron.log
