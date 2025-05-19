.PHONY: build run stop clean test lint docs push dev validate-env-dev validate-env-prod run-dev run-prod build-dev build-prod test-all

-include .env

# Set the environment to development by default development or production
APP_ENV ?= production

ifeq ($(APP_ENV),development)
DOCKER_COMPOSE = docker-compose -f dev-docker-compose.yml
else
DOCKER_COMPOSE = docker-compose -f docker-compose.yml
endif
DEV_DOCKER_COMPOSE=docker-compose -f dev-docker-compose.yml

# Docker compose commands
build:
	$(DOCKER_COMPOSE) build

run:
	$(DOCKER_COMPOSE) up -d 

stop:
	$(DOCKER_COMPOSE) down

clean:
	@echo "Cleaning up development Docker resources..."
	docker image prune -f
	$(DOCKER_COMPOSE) down --remove-orphans

clean-all: clean
	@echo "Cleaning up development files..."
ifeq ($(OS),Windows_NT)
	del /s /q *.pyc
	for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
	for /d /r . %d in (.pytest_cache) do @if exist "%d" rd /s /q "%d"
	for /d /r . %d in (*.egg-info) do @if exist "%d" rd /s /q "%d"
else
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name "*.egg-info" -exec rm -r {} +
endif

# Cleanup commands
clean-dev:
	@echo "Cleaning up development Docker resources..."
	docker image prune -f
	docker-compose -f dev-docker-compose.yml down --remove-orphans

clean-all: clean
	@echo "Performing deep cleanup of all Docker resources..."
	docker system prune -af
	docker volume prune -f
	@echo "Cleaning up development files..."
ifeq ($(OS),Windows_NT)
	del /s /q *.pyc
	for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
	for /d /r . %d in (.pytest_cache) do @if exist "%d" rd /s /q "%d"
	for /d /r . %d in (*.egg-info) do @if exist "%d" rd /s /q "%d"
else
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name "*.egg-info" -exec rm -r {} +
endif

# Development mode (hot reload)
dev: clean-dev
	@echo "Starting development environment..."
	$(DEV_DOCKER_COMPOSE) up --build

# Quick development mode without cleanup
dev-quick:
	@echo "Starting development environment (without cleanup)..."
	$(DEV_DOCKER_COMPOSE) up --build

# Testing commands
test-all:
    @services=$$(ls services/); \
    pids=""; \
    for service in $$services; do \
		if [ -f "services/$$service/Dockerfile" ]; then \
			echo "Building and testing $$service..."; \
			( \
				docker build --target test -t $$service-test -f services/$$service/Dockerfile . && \
				docker run --rm \
					-e OPENAI_API_KEY=${OPENAI_API_KEY} \
					-e APP_ENV=test \
					-e LOG_LEVEL=DEBUG \
					$$service-test \
			) & \
			pids="$$pids $$!"; \
		fi \
	done; \
	echo "Waiting for all tests to complete..."; \
	for pid in $$pids; do \
		wait $$pid || exit 1; \
	done

test-chat-service:
	docker build --target test -t chat-service-test -f services/chat-service/Dockerfile .
	docker run --rm -e OPENAI_API_KEY=${OPENAI_API_KEY} -e APP_ENV=test -e LOG_LEVEL=DEBUG chat-service-test

test-chat-service-int:
	docker build --target test -t chat-service-test-int -f services/chat-service/Dockerfile .
	docker run --rm \
		-e OPENAI_API_KEY=${OPENAI_API_KEY} \
		-e APP_ENV=test \
		-e LOG_LEVEL=DEBUG \
		-e RUN_INTEGRATION_TESTS=1 \
		chat-service-test-int \
		pytest -vvs tests/int/test_chat_completion_int.py

# Code quality
lint:
	black .
	flake8 .
	mypy .

# Documentation
docs:
	pip install pdoc3
	pdoc --html --output-dir docs/ shared/advisor_utils/

# Docker image management
push:
	docker-compose build
	docker-compose push

# Environment validation
validate-env-dev:
	python scripts/validate_env.py .env.dev

validate-env-prod:
	python scripts/validate_env.py .env.prod

# Environment-specific commands
run-dev: validate-env-dev
	docker-compose --env-file .env.dev up

run-prod: validate-env-prod
	docker-compose --env-file .env.prod up

build-dev: validate-env-dev
	docker-compose --env-file .env.dev build

build-prod: validate-env-prod
	docker-compose --env-file .env.prod build
