# FINKI Chat Bot

RAG chat bot for the [`FCSE Students`](https://discord.gg/finki-studenti-810997107376914444) Discord server, powered by [LangChain](https://github.com/langchain-ai/langchain) and [FastAPI](https://github.com/fastapi/fastapi). Uses [PostgreSQL](https://github.com/postgres/postgres) and [pgvector](https://github.com/pgvector/pgvector) for keeping documents. Has support for multiple LLMs.

It currently works on a dataset of documents (FAQ) by conditionally injecting them in the context if they are related to the prompt.

## Services

This project comes as a monorepo of microservices:

- API (`/api`) for managing documents, links and chatting
- GPU API (`/gpu-api`) for locally executing GPU accelerated tasks like embeddings generation

## Quick Setup (Production)

It's highly recommended to do this in Docker.

To run the chat bot:

1. Download [`compose.prod.yaml`](./compose.prod.yaml)
2. Download [`.env.sample`](.env.sample), rename it to `.env` and change it to your liking
3. Run `docker compose -f compose.prod.yaml up -d`

The API will be running on port `8880`. This also brings up a `pgAdmin` instance. You may use it to view or create documents. It's accesible on port `5555` by default.

## Quick Setup (Development)

Requires Python >= 3.13 and [`uv`](https://github.com/astral-sh/uv).

1. Clone the repository: `git clone https://github.com/finki-hub/finki-chat-bot.git`
2. Install dependencies: `cd api && uv sync`
3. Prepare env. variables by copying `env.sample` to `.env` - minimum setup requires the database configuration, it can be left as is
4. Run it: `docker compose up -d`

This also brings up an OpenAPI instance at `localhost:8880/docs`.

## Endpoints

This is an incomplete list. You may view all available endpoints on the OpenAPI documentation.

- `/questions/list` - get all questions
- `/questions/name/<name>` - get a question by its name
- `/questions/embed` - generate embeddings for all questions for a given model

## License

This project is licensed under the terms of the MIT license.
