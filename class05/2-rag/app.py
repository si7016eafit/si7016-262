#
# Universidad EAFIT
# 2026-2
# SI7016 - NLP - Lecture 05b - Chatbot RAG con Streamlit (actualizado 2026-2)
#
import uuid

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

# 1. Vector store: un solo cliente (langchain_chroma) para escribir y leer,
#    en vez de mezclar un chromadb.PersistentClient crudo con un Chroma de
#    LangChain apuntando al mismo directorio pero a una colección distinta
#    (bug del notebook original: los documentos añadidos por el cliente crudo
#    quedaban invisibles para el retriever de LangChain).
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(
    collection_name="chatbot_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever()

# 2. Modelo de chat. La API key se toma de la variable de entorno
#    OPENAI_API_KEY (export OPENAI_API_KEY="..." antes de correr streamlit) -
#    nunca hardcodeada en el código.
llm = ChatOpenAI(model="gpt-5.6")


# 3. Grafo de LangGraph: recuperar contexto y generar respuesta, con memoria
#    de la conversación vía checkpointer (reemplaza a ConversationBufferMemory
#    + ConversationalRetrievalChain, ambos deprecados en LangChain).
def retrieve_and_generate(state: MessagesState):
    user_message = state["messages"][-1].content
    docs = retriever.invoke(user_message)
    context = "\n\n".join(doc.page_content for doc in docs) or "(sin contexto relevante)"
    system = SystemMessage(
        content=f"Responde la pregunta del usuario usando este contexto cuando sea relevante:\n\n{context}"
    )
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("retrieve_and_generate", retrieve_and_generate)
graph_builder.add_edge(START, "retrieve_and_generate")
graph_builder.add_edge("retrieve_and_generate", END)

chat_chain = graph_builder.compile(checkpointer=InMemorySaver())

# Interfaz en Streamlit
st.title("Chatbot RAG con LangGraph (2026)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# thread_id único por sesión de navegador, para que las conversaciones de
# distintos usuarios no se mezclen en la memoria del checkpointer
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Escribe tu mensaje:")

if user_input:
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    result = chat_chain.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    response = result["messages"][-1].content

    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "assistant", "content": response})

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
