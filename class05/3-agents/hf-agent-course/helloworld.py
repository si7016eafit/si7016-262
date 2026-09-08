# Universidad EAFIT - SI7016 - NLP - Lecture 05c
# Ejemplo companero de class05c.ipynb (OpenAI Agents SDK)
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant", model="gpt-5.6")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
