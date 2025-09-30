cat > Makefile <<'MAKE'
.PHONY: up down logs ingest query reset-db fmt lint check

PY=cd rag && . .venv/bin/activate;

up:
	cd infra && docker compose up -d

down:
	cd infra && docker compose down

logs:
	cd infra && docker compose logs -f

ingest:
	$(PY) python ingest.py

query:
	$(PY) python query.py "¿Qué es MCP?"

reset-db:
	cd infra && docker exec -it $$(docker ps -qf "ancestor=ankane/pgvector") \
	  psql -U postgres -d ragdb -c "DROP TABLE IF EXISTS documents;"

fmt:
	$(PY) ruff check --fix .
	$(PY) ruff format .

lint:
	$(PY) ruff check .

check: lint
MAKE
