import networkx as nx
import nx_arangodb as nxadb
from arango import ArangoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import re, os, json, time, random, yaml, pandas
from langgraph.prebuilt import create_react_agent
from langchain.tools.base import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.graphs import ArangoGraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_core.tools import tool
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry,get_registry
import colorlog, logging
from tqdm import trange
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
    '%(log_color)s %(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

class GraphAgent:
    def __init__(self):
        # remote db
        self.db = ArangoClient(hosts=CONFIG["DATABASE_HOST"]
                  ).db(username=CONFIG["DATABASE_USERNAME"], 
                       password=CONFIG["DATABASE_PASSWORD"], 
                       verify=True)

        # networkx graph, Re-connect to the a Graph, which persists in the database
        self.G_adb = nxadb.Graph(name=CONFIG["GRAPH_NAME"], db=self.db)

        # arangodb,
        self.arango_graph = ArangoGraph(self.db)

        # llm
        self.text_embedder = get_registry().get('openai').create(name=CONFIG["openai_embedder"])
        self.llm = ChatOpenAI(model_name=CONFIG["openai_model"])
        self.verbose = CONFIG["VERBOSE"]
        self.MAX_ATTEMPTS = CONFIG["MAX_ATTEMPTS"]

        # agent
        self.history = []
        self.agent = create_react_agent(self.llm, 
                                        tools=[StructuredTool.from_function(self.text_to_nx_algorithm_to_text), 
                                                 StructuredTool.from_function(self.text_to_aql_to_text), 
                                                 StructuredTool.from_function(self.whoami),
                                                 StructuredTool.from_function(self.vector_qa)])
        # vector db
        class Schema(LanceModel):
            _id: str | int # primary key for the node
            description: str = self.text_embedder.SourceField() # what to embed
            metadata: str | list[str]  # what to search
            #embedding: list[float] = None  # pre-computed embedding, such as image embedding
            vector: Vector(self.text_embedder.ndims()) = self.text_embedder.VectorField() # output embedding from the 'description' field
                
        self.db = lancedb.connect(uri='../data/temp')
        self.table = self.db.create_table(CONFIG["GRAPH_NAME"], schema=Schema, on_bad_vectors='drop', mode="overwrite")
        self.top_k = CONFIG["top_k"]
        #self.create_vector_db()

    def create_vector_db(self):
        # create vector db
        # run this after G_adb is created

        logger.info("Creating vector database")
        df = pandas.DataFrame([data for _, data in self.G_adb.nodes(data=True)])
        df['node_type'] = df['_id'].apply(lambda x: x.split('/')[0])
        col_names = df.columns.tolist()

        # vector index
        batch_size = 128
        for start_idx in trange(0, len(df), batch_size):
            batch = df.iloc[start_idx:min(start_idx+batch_size, len(df))]

            self.table.add(data=[{
                '_id': str(row['_id']),
                'description': f"{str(row['name'])} : {str(row['description'])}", # make sure there is a description field in the dataframe
                'metadata': str([f"{col} : {str(row[col])}" for col in col_names if col not in ['_id', 'description']])
            } for _, row in batch.iterrows()])
        # full text search index
        self.table.create_fts_index(['description', 'metadata'], replace=True)

        return


    def route_query(self, query):
        pass

    #@tool
    def whoami(self, query: str):
        """
        I am a graph agent.
        An overview of the database collections and graph schema you can find here:
        """
        return str(self.arango_graph.schema)

    #@tool
    def vector_qa(self, query:str):
        """ This tool is available to invoke the vector search on all nodes.
        semantic similarity between the query and the description or metadata of nodes.
        """

        retrieved_nodes = (self.table
                    .search(query=query, query_type='hybrid')
                    #.rerank(reranker=reranker) # not used for now, slow down the search
                    .limit(self.top_k)
                    .to_pandas()
                    .drop(columns=['vector', '_relevance_score'])
                    #.drop_duplicates(subset=['_id'], keep='first')
                    )
        
        
        retrieved_context = str(retrieved_nodes.to_csv(index=False, sep='|'))
        
        response = self.llm.invoke(f"""
            You are a helpful assistant that can answer questions about a graph.
            The retrieved context are a list of nodes and their descriptions.
            Answer the question based on the context only. and cite the source.

            Question: 
            {query}
            Context: 
            {retrieved_context}
            Your response:
        """).content
        return response
    
    #@tool
    def text_to_aql_to_text(self, query: str):
        """This tool is available to invoke the
        ArangoGraphQAChain object, which enables you to
        translate a Natural Language Query into AQL, execute
        the query, and translate the result back into Natural Language.
        """

        chain = ArangoGraphQAChain.from_llm(
            llm=self.llm,
            graph=self.arango_graph,
            verbose=self.verbose,
            allow_dangerous_requests=True
        )
        
        result = chain.invoke(query)
        return str(result["result"])

    #@tool
    def text_to_nx_algorithm_to_text(self, query:str):
        """This tool is available to invoke a NetworkX Algorithm on
        the ArangoDB Graph. You are responsible for accepting the
        Natural Language Query, establishing which algorithm needs to
        be executed, executing the algorithm, and translating the results back
        to Natural Language, with respect to the original query.

        If the query (e.g traversals, shortest path, etc.) can be solved using the Arango Query Language, then do not use
        this tool.
        """

        attempt = 1
        FINAL_RESULT = None
        feedback = ""
        while attempt <= self.MAX_ATTEMPTS:
            text_to_nx = self.llm.invoke(f"""
            I have a NetworkX Graph called `G_adb`. It has the following schema: {self.arango_graph.schema}
            Answer the following query: {query}.

            Generate the Python Code using the `G_adb` object.
            - Be very precise on the NetworkX algorithm you select to answer this query. Think step by step.
            - Only assume that networkx is installed, and other base python dependencies.
            - Write code in one block, and do not include any instructions.
            - Always set the last variable as `FINAL_RESULT`, which represents the answer to the original query.
            - Only provide python code that I can directly execute via `exec()`. Do not provide any instructions.
            - Make sure that `FINAL_RESULT` stores a short & consice answer. Avoid setting this variable to a long sequence.
            - Previous runtime feedback: {feedback}

            Your code:
            """).content
            try:
                text_to_nx_cleaned = re.sub(r"^```python\n|```$", "", text_to_nx, flags=re.MULTILINE).strip()
            
                if self.verbose:
                    print("\n* Generating NetworkX code")
                    print(text_to_nx_cleaned)

                global_vars = {"G_adb": self.G_adb, "nx": nx}
                local_vars = {}
                exec(text_to_nx_cleaned, global_vars, local_vars)
                text_to_nx_final = text_to_nx
                FINAL_RESULT = local_vars.get("FINAL_RESULT")
                if FINAL_RESULT:
                    break
            except Exception as e:
                feedback += f"ERROR: {e}"

            attempt += 1
            
        if self.verbose:
            print("\n* Executing code")
            print(f"FINAL_RESULT: {FINAL_RESULT}")
            print("\n* Formulating final answer")

        nx_to_text = self.llm.invoke(f"""

            I have the following graph analysis query: {query}.
            and executed the following python code to help me answer my query:
            ---
            {text_to_nx_final}
            ---

            The `FINAL_RESULT` variable is set to the following: {FINAL_RESULT}.

            Based on my original Query and FINAL_RESULT, generate a short and concise response to
            answer my query.
            
            Your response:
        """).content

        return nx_to_text

    def vector_search(self, query):
        pass

    # Note: Consider implementing a hybrid tool that combines both AQL & NetworkX Algorithms!
    def query_graph(self, query):
        # 
        messages = [{"role": "user", "content": query}]
        final_state = self.agent.invoke({"messages": messages})
        self.history.append({"messages": messages, "final_state": final_state})

        return final_state["messages"][-1].content

if __name__ == "__main__":
    graph_agent = GraphAgent()
    print(graph_agent.query_graph("who are you?"))
    print(graph_agent.query_graph("give me a random event, and its time and location"))
    print(graph_agent.query_graph("how many events are there in the graph?"))
    print(graph_agent.query_graph("what is the most impactful node of type event in the graph? explain why"))
    print(graph_agent.query_graph("how many nodes are connected to the event named 'Orma youth attack journalists'? use networkx code"))