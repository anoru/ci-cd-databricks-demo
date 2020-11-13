.PHONY: virtual install build-requirements black isort flake8

PROJECT_NAME = lab-project-demo

clean-pyc:
    find . -name '*.pyc' -exec rm --force {} +
    find . -name '*.pyo' -exec rm --force {} +
   name '*~' -exec rm --force  {}

clean-build:
    rm --force --recursive build/
    rm --force --recursive dist/
    rm --force --recursive *.egg-info

virtual: # Creates an isolated python 3 environment
    virtualenv -p /usr/bin/python3 .venv

install:
	.venv/bin/pip install -Ur requirements.txt

update-requirements: install
	.venv/bin/pip freeze > requirements.txt

# black
.venv/bin/black: # Installs black code formatter
	.venv/bin/pip install -U black

black: .venv/bin/black # Formats code with black
	.venv/bin/black *.py

# isort
.venv/bin/isort: # Installs isort to sort imports
	.venv/bin/pip install -U isort

isort: .venv/bin/isort # Sorts imports using isort
	.venv/bin/isort *.py

# flake8
.venv/bin/flake8: # Installs flake8 code linter
	.venv/bin/pip install -U flake8

flake8: .venv/bin/flake8 # Lints code using flake8
	.venv/bin/flake8 *.py

# Optional if you need jupyter notebook
project_name = your-project-name

jupyter: .venv/bin/ipython
	.venv/bin/jupyter notebook

.venv/bin/ipython:
	.venv/bin/pip install -U ipykernel jupyter
	.venv/bin/ipython kernel install --user --name=$(PROJECT_NAME)

# build a docker image for the project
# TO DO
#docker-run:
#    docker build \
#      --file=./Dockerfile \
#      --tag=my_project ./
#    docker run \
#      --detach=false \
#      --name=my_project \
#      --publish=$(HOST):8080 \
#      my_project
