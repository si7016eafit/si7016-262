# SI7016 NLP — Class 03 (2026-2)

Laboratorios:
1. `class03a-langchain-agent.ipynb`: agente, tools y memoria.
2. `class03a-hugginfface-agent.ipynb`: agente, tools y memoria.
3. `class03b-rag-qdrant.ipynb`: RAG con embeddings locales + Qdrant.
4. `class03c-mcp-intro.ipynb`: MCP Python SDK v2 (`MCPServer` + `Client`).

## Ejemplos locales: Instalación
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Puede tambien crear el docker y corren un jupyterhub local.

Configure variables para `ANTHROPIC_API_KEY` u `OPENAI_API_KEY`, `HUGGING_FACE`, `TAVILY`, `LANGSMITH` entre otras

Ojo: algunas requiren pago

Puede definir `HF_MODEL` para desacoplar el lab de una versión concreta del modelo.


