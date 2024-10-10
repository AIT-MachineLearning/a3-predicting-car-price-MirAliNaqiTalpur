FROM python:3.11

WORKDIR /code

#Intalling dependencies
# RUN pip3 install flask==3.0.3 pandas==1.5.3 joblib==1.4.2 scikit-learn==1.5.1 numpy==1.26.4 pytest pytest-depends mlflow==2.15.1
RUN pip install --no-cache-dir flask pandasjoblib==1.4.2 scikit-learn==1.5.1 numpy pytest pytest-depends pytest-timeout mlflow==2.15.1


COPY ./code /code/
EXPOSE 5000

CMD ["python", "app.py"]