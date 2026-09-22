# Nota (revisión sept. 2026): este archivo es idéntico a mcp_host_example.py.
# Se dejan ambos sin eliminar (no había autorización para borrar archivos del
# repo del profesor), igual que se hizo en class05/3-agents/ con main.py y
# customerservice.py en una ronda anterior de este mismo proyecto.
#
# El patrón (Responses API + tool remoto "mcp") sigue vigente: se verificó contra
# la documentación oficial de OpenAI (developers.openai.com/api/docs/guides/
# tools-connectors-mcp). "gpt-5" es un alias que siempre apunta a la última
# variante estable de GPT-5, así que no requiere fijarse a una fecha — si se
# prefiere un modelo explícito, las alternativas vigentes en 2026 son
# "gpt-5.6-terra" (reemplazo oficial de los modelos gpt-3.5/gpt-5 retirados) o
# el nuevo buque insignia "gpt-6-astra" (lanzado el 3 de septiembre de 2026).

from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-5",
    tools=[
        {
            "type": "mcp",
            "server_label": "openai_docs",
            "server_description": "Servidor MCP oficial de documentación de OpenAI",
            "server_url": "https://developers.openai.com/mcp",
            "require_approval": "never"
        }
    ],
    input=(
        "Busca en la documentación de OpenAI qué es MCP y "
        "resúmeme las diferencias entre MCP Host, MCP Client y MCP Server."
    ),
)

print(resp.output_text)
