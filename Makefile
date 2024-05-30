
env:
	python3 -m venv env
	source  ml_package/bin/activate

install:
	pip install -r requirements.txt

setup:
	python setup.py sdist bdist_wheel
