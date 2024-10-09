FROM python:3.11.4-bookworm


WORKDIR /code

#Intalling dependencies
RUN pip3 install flask pandas joblib scikit-learn numpy pytest pytest-depends mlflow 


COPY ./code /code/
EXPOSE 5000

CMD ["python", "app.py"]