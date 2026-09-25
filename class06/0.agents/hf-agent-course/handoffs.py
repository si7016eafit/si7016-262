# Universidad EAFIT - SI7016 - NLP - Lecture 05c
# Ejemplo companero de class05c.ipynb (OpenAI Agents SDK) - handoffs (delegar
# la conversación completa a otro agente especializado, no solo llamar una tool)
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model="gpt-5.6",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model="gpt-5.6",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
    model="gpt-5.6",
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)
    # ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?


if __name__ == "__main__":
    asyncio.run(main())
