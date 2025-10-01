¡claro! Estos son los ARGS que entiende query.py (los pasas como ARGS="..." en make query o directamente en CLI). Entre paréntesis te pongo la variable de .env que pisan si las usas.

Parámetros disponibles

--k <int> (TOP_K)
Cuántos pasajes finales entran en el contexto del LLM (tras filtros/diversificación).
Sube k → más cobertura (más tokens, más latencia); baja k → más precisión.
Rango típico: 3–8.

--threshold <float> (SCORE_THRESHOLD)
Umbral de similitud (0..1). Descarta pasajes con score bajo.
Más alto = más estricto (puede quedarse sin contexto); más bajo = más ruido.
Rango típico: 0.20–0.35.

--max-per-doc <int> (MAX_CHUNKS_PER_DOC)
Máximo de pasajes por documento en el contexto (evita monopolios).
Rango típico: 1–3.

--no-round-robin (ROUND_ROBIN_PER_DOC=0)
Desactiva la alternancia entre documentos. Con esto, si un doc domina, puede aportar varios pasajes seguidos (dentro de --max-per-doc).

--rerank / --no-rerank (RERANKER_ENABLED)
Fuerza activar o desactivar el reranker (cross-encoder).
Activarlo mejora precisión, pero añade latencia.
Recuerda: el nº de candidatos previos lo marca RERANKER_CANDIDATES en .env.

--max-ctx <int> (MAX_CONTEXT_CHARS)
Límite de caracteres para el bloque de contexto (safety contra prompts gigantes).
Si se supera, corta los pasajes sobrantes.

Tip: --help te muestra la ayuda integrada.

Cómo usarlos
Con make:
# pregunta libre usando valores de .env
make query Q="Resume MCP en 5 puntos"

# sube cobertura y baja umbral
make query Q="Mejores prácticas RAG" ARGS="--k 7 --threshold 0.22"

# limita monopolio por documento y desactiva round-robin
make query Q="Instalación paso a paso" ARGS="--max-per-doc 1 --no-round-robin"

# fuerza reranker y recorta el contexto
make query Q="Comparativa de modelos de embeddings" ARGS="--rerank --max-ctx 3000"

Guía rápida de ajuste (reglas prácticas)

Si el LLM mete paja/alucina: sube --threshold (0.28–0.32) y baja --k (4–5).

Si la respuesta omite cosas: baja --threshold (0.22–0.25) o sube --k (6–7).

Si un PDF domina: --max-per-doc 1 y (opcional) --no-round-robin 0 para alternar.

Si hay ruido en los top-k: prueba --rerank (mejor precisión) y quizá baja --k.
