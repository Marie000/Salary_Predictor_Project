FROM python:3.10

WORKDIR /src

COPY . /src

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/Marie000/Salary_Predictor_Project.git -U

EXPOSE 5000

CMD ["python", "training_pipeline.py"]