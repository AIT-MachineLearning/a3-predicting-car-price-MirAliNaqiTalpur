FROM python:3.11

WORKDIR /root/code

RUN pip install --no-cache-dir flask pandas joblib==1.4.2 scikit-learn==1.5.1 numpy pytest pytest-depends pytest-timeout mlflow==2.15.1


COPY .code /root/code
EXPOSE 5000

CMD ["python", "/root/code/app.py"]