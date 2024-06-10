build-docker:
	sudo docker build -t predictor-image .

run-docker:
	sudo docker run -d -it --name predictor -p 8080:8080 predictor-image bash

api-docker:
	sudo docker exec -d -w /code predictor main.py