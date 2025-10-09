cat > rag/README.md <<'MD'
![CI](https://github.com/Ayax111/ai-stack/actions/workflows/ci.yml/badge.svg)
![pre-commit](https://github.com/Ayax111/ai-stack/actions/workflows/pre-commit.yml/badge.svg)

# RAG + LM Studio + pgvector (con evaluación y reranker)

Guía de puesta en marcha y funcionamiento de un **sistema RAG** local que:
- Indexa documentos (`data/`) en **Postgres + pgvector**.
- Consulta un **LLM local en LM Studio** (API OpenAI-compatible).
- Recupera pasajes relevantes, **opcionalmente los reranquea** con un modelo cross-encoder.
- Evalúa la calidad del retrieve (hit@k, MRR, cobertura) con scripts ligeros.

## Índice
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Estructura](#estructura)
- [Configuración (.env)](#configuración-env)
- [Puesta en marcha](#puesta-en-marcha)
- [Flujo: ingesta → consulta](#flujo-ingesta--consulta)
- [Reranker (opcional)](#reranker-opcional)
- [Evaluación (baseline vs rerank)](#evaluación-baseline-vs-rerank)
- [Uso rápido (Makefile)](#uso-rápido-makefile)
- [Solución de problemas](#solución-de-problemas)
- [Buenas prácticas y seguridad](#buenas-prácticas-y-seguridad)
- [Tooling: pre-commit y CI](#tooling-pre-commit-y-ci)

---

## Arquitectura
