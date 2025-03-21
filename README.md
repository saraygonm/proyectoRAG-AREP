# Aplicación de recuperación de generación aumentada (RAG)

## Descripción

Este proyecto implementa una aplicación de Recuperación de Generación Aumentada (RAG) utilizando el framework LangChain.
RAG permite a los modelos de lenguaje acceder a fuentes externas de información en tiempo real,
mejorando así la precisión y contexto de las respuestas generadas. La aplicación realiza tareas de generación de texto mientras se apoya en un sistema de
recuperación para proporcionar información contextual.

---

## Arquitectura y Componentes del Proyecto

- **Carga de Documentos:** Carga documentos desde una fuente web (por ejemplo, una entrada de blog).
- **División de Texto:** Divide los documentos en fragmentos más pequeños para facilitar una recuperación eficiente.
- **Almacenamiento Vectorial:** Almacena los embeddings de los documentos para realizar búsquedas por similitud.
- **Recuperación:** Recupera fragmentos relevantes de documentos según las consultas de los usuarios.
- **Generación:** Utiliza un modelo de lenguaje para generar respuestas basadas en los documentos recuperados.
- **Gestión del Estado:** Maneja el historial y contexto de las conversaciones en interacciones de múltiples turnos.
- **Agente:** Maneja consultas complejas dividiéndolas en tareas más pequeñas.


# Aplicación de Recuperación de Generación Aumentada (RAG)

## Descripción

Este proyecto implementa una aplicación de **Recuperación de Generación Aumentada (RAG)** utilizando el framework LangChain. La aplicación recupera información relevante de una base de datos de documentos y utiliza un modelo de lenguaje para generar respuestas a consultas del usuario. Este proyecto se divide en dos partes:

- **Parte 1:** 
Configura el pipeline básico de RAG, que incluye la carga de documentos, división, indexación y recuperación.
- **Parte 2:** 
Expande la aplicación con la gestión del historial de chat con estado y el manejo de consultas complejas usando agentes.

---

## Arquitectura y Componentes del Proyecto

La aplicación está compuesta por los siguientes componentes:

- **Carga de Documentos:** Carga documentos desde una fuente web (por ejemplo, una entrada de blog).
- **División de Texto:** Divide los documentos en fragmentos más pequeños para facilitar la recuperación eficiente.
- **Almacenamiento Vectorial:** Almacena los embeddings de los documentos para realizar búsquedas de similitud.
- **Recuperación:** Recupera fragmentos relevantes de documentos según las consultas de los usuarios.
- **Generación:** Utiliza un modelo de lenguaje para generar respuestas basadas en el contenido recuperado.
- **Gestión del Estado:** Maneja el historial y contexto de las conversaciones en interacciones de múltiples turnos.
- **Agente:** Maneja consultas complejas dividiéndolas en tareas más pequeñas.

---

## 📍 Comenzando

Estas instrucciones te permitirán obtener una copia del proyecto en funcionamiento en tu máquina local para propósitos de desarrollo y pruebas.

### 🔧 Prerrequisitos

- Python 3.7 o superior.
- Una clave API válida de OpenAI.
- Una clave API de Pinecone (para almacenamiento vectorial).

### ⚙️ Instrucciones paso a paso

1) **Instalar Librerías Requeridas**

   Primero, se debe instalar LangChain y las dependencias necesarias usando el siguiente comando:

   ```bash
   pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph beautifulsoup4 langchain-openai langchain-pinecone pinecone-notebooks
   ```

2) **Configurar las Claves API**

   Se deben configurar las claves API de OpenAI y Pinecone como variables de entorno.

   **Clave API de OpenAI:**
   ```python
   import getpass
   import os

   if not os.environ.get("OPENAI_API_KEY"):
       os.environ["OPENAI_API_KEY"] = getpass.getpass("Introduce tu clave API para OpenAI: ")
   ```

   **Clave API de Pinecone:**
   ```python
   if not os.getenv("PINECONE_API_KEY"):
       os.environ["PINECONE_API_KEY"] = getpass.getpass("Introduce tu clave API de Pinecone: ")
   ```

3) **Inicializar el Índice de Pinecone**

   Se debe crear un índice en Pinecone para almacenar los embeddings de los documentos.

   ```python
   from pinecone import Pinecone, ServerlessSpec

   pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
   index_name = "langchain-test-index"  # Se puede cambiar si se desea

   existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
   if index_name not in existing_indexes:
       pc.create_index(
           name=index_name,
           dimension=3072,
           metric="cosine",
           spec=ServerlessSpec(cloud="aws", region="us-east-1"),
       )
       while not pc.describe_index(index_name).status["ready"]:
           time.sleep(1)

   index = pc.Index(index_name)
   ```

4) **Inicializar los Embeddings**

   Se utilizan los embeddings de OpenAI para codificar los documentos.

   ```python
   from langchain_openai import OpenAIEmbeddings

   embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
   ```

5) **Configurar el Almacenamiento Vectorial**

   Se configura Pinecone como el almacenamiento vectorial.

   ```python
   from langchain_pinecone import PineconeVectorStore

   vector_store = PineconeVectorStore(index=index, embedding=embeddings)
   ```

---

## ⚙️ Ejecución del Código

### Parte 1: Pipeline Básico de RAG

1) **Cargar y Dividir Documentos**

   Se carga un artículo web y se divide el contenido en fragmentos más pequeños para facilitar la recuperación.

   ```python
   import bs4
   from langchain_community.document_loaders import WebBaseLoader
   from langchain_text_splitters import RecursiveCharacterTextSplitter

   loader = WebBaseLoader(
       web_paths=("<https://lilianweng.github.io/posts/2023-06-23-agent/>",),
       bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
   )
   docs = loader.load()

   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   all_splits = text_splitter.split_documents(docs)
   ```

2) **Indexar los Documentos**

   Se añaden los fragmentos de los documentos al almacenamiento vectorial (Pinecone).

   ```python
   document_ids = vector_store.add_documents(documents=all_splits)
   ```

3) **Definir el Prompt de RAG**

   Se utiliza un prompt predefinido para la tarea de preguntas y respuestas.

   ```python
   from langchain import hub

   prompt = hub.pull("rlm/rag-prompt")
   ```

4) **Recuperar y Generar Respuestas**

   Se recuperan los documentos relevantes y se genera una respuesta usando el modelo de lenguaje.

   ```python
   from langchain_core.documents import Document
   from typing_extensions import List, TypedDict

   class State(TypedDict):
       question: str
       context: List[Document]
       answer: str

   def retrieve(state: State):
       retrieved_docs = vector_store.similarity_search(state["question"])
       return {"context": retrieved_docs}

   def generate(state: State):
       docs_content = "\\n\\n".join(doc.page_content for doc in state["context"])
       messages = prompt.invoke({"question": state["question"], "context": docs_content})
       response = llm.invoke(messages)
       return {"answer": response.content}

   from langgraph.graph import START, StateGraph

   graph_builder = StateGraph(State).add_sequence([retrieve, generate])
   graph_builder.add_edge(START, "retrieve")
   graph = graph_builder.compile()

   result = graph.invoke({"question": "What is Task Decomposition?"})
   print(result["answer"])
   ```

### Parte 2: Gestión del Historial de Chat y Agentes

1) **Gestión del Estado**

   Se utiliza un punto de control de memoria para gestionar el historial de chat.

   ```python
   from langgraph.checkpoint.memory import MemorySaver

   memory = MemorySaver()
   graph = graph_builder.compile(checkpointer=memory)
   ```

2) **Manejo de Consultas Complejas con Agentes**

   Se utiliza un agente para manejar consultas complejas.

   ```python
   from langgraph.prebuilt import create_react_agent

   agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
   ```

3) **Ejecutar el Agente**

   Se obtienen respuestas del agente en modo streaming.

   ```python
   config = {"configurable": {"thread_id": "abc123"}}

   for event in agent_executor.stream(
       {"messages": [{"role": "user", "content": "What is Task Decomposition?"}]},
       stream_mode="values",
       config=config,
   ):
       event["messages"][-1].pretty_print()
   ```
---

## 👨🏼‍💻 Autora

**Saray Alieth Mendivelso Gonzalez**

---
