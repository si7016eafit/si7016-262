# Universidad EAFIT - SI7016 - NLP - Lecture 05c
# Ejemplo companero de class05c.ipynb (OpenAI Agents SDK)
# pip install openai-agents
import asyncio
from agents import Agent, Runner

# Modelo de frontera 2026 explícito (si se omite, el Agents SDK usa un
# modelo por defecto configurado a nivel de librería/entorno).
agent = Agent(name="Assistant", instructions="You are a helpful assistant", model="gpt-5.6")

async def main():
    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)

# Ejecutar el programa de forma correcta
asyncio.run(main())

