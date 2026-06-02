FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir mlflow pandas scikit-learn numpy

CMD ["mlflow", "run", "."]
