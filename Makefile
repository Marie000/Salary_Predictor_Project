
env:
	python3 -m venv env
	source  ml_package/bin/activate

install:
	pip install -r requirements.txt

setup:
	python setup.py sdist bdist_wheel

update-package:
	pip install git+https://github.com/Marie000/Salary_Predictor_Project.git -U
