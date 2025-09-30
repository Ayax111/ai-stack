.RECIPEPREFIX := >
.PHONY: up down logs ingest query reset-db fmt lint check venv deps run pip shell

PY=cd rag && . .venv/bin/activate;

up:
> cd infra && docker compose up -d

down:
> cd infra && docker compose down

logs:
> cd infra && docker compose logs -f

ingest:
> $(PY) python ingest.py

# Ahora 'make query Q="tu pregunta"' funciona
query:
> $(PY) python query.py $(ARGS) "$(Q)"

reset-db:
> cd infra && docker exec -it $$(docker ps -qf "ancestor=ankane/pgvector") \
>   psql -U postgres -d ragdb -c "DROP TABLE IF EXISTS documents;"

fmt:
> $(PY) ruff check --fix .
> $(PY) ruff format .

lint:
> $(PY) ruff check .

check: lint

# ---- VENV helpers ----
venv:
> cd rag && [ -d .venv ] || python -m venv .venv
> cd rag && . .venv/bin/activate && python -m pip install --upgrade pip

deps: venv
> cd rag && . .venv/bin/activate && pip install \
>   openai httpx python-dotenv sqlalchemy "psycopg[binary]" pgvector \
>   langchain pypdf sentence-transformers numpy ruff

# Ejecuta cualquier comando dentro del venv:
# uso: make run cmd="python query.py '¿Qué es MCP?'"
run:
> cd rag && . .venv/bin/activate && $(cmd)

# pip dentro del venv:
# uso: make pip args="install fastapi"
pip:
> cd rag && . .venv/bin/activate && pip $(args)

# Abre una shell interactiva con el venv activo (sal con 'exit')
shell:
> cd rag && bash -lc 'source . .venv/bin/activate && exec bash -i'
