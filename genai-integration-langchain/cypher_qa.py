import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain

# Initialize the LLM
model = init_chat_model(
    "gpt-4o", 
    model_provider="openai",
    temperature=0.0 # 输出是确定性和精确的
)

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"), 
    password=os.getenv("NEO4J_PASSWORD"),
)

# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=model, 
    allow_dangerous_requests=True,
    verbose=True, # 详细，打印生成的Cypher查询和用于生成答案的完整上下文
)

# Invoke the chain
question = "How many movies are in the Sci-Fi genre?"
response = cypher_qa.invoke({"query": question})
print(response["result"])

# 限制架构可以通过以下方式帮助生成更好的 Cypher：
# 降低生成的 Cypher 查询的复杂性。
# 帮助 LLM 专注于图表的相关部分。
# 排除可能混淆 LLM 的不相关或不需要的图部分。


# 限制架构: --- 包含 include_types 或 排除 exclude_types

# cypher_qa = GraphCypherQAChain.from_llm(
#     graph=graph, 
#     llm=model, 
#     include_types=["Movie", "ACTED_IN", "Person"],
#     exclude_types=["User", "RATED"],
#     allow_dangerous_requests=True,
#     verbose=True, 
# )


# 提供具体的说明和示例查询，以改进 Cypher 查询生成。
from langchain_core.prompts.prompt import PromptTemplate

# Cypher template
# Cypher template with examples
# Cypher template with examples
cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, for example "The 39 Steps" becomes "39 Steps, The".

Schema:
{schema}
Examples:
1. Question: Get user ratings?
   Cypher: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE u.name = "User name" RETURN r.rating AS userRating
2. Question: Get average rating for a movie?
   Cypher: MATCH (m:Movie)<-[r:RATED]-(u:User) WHERE m.title = 'Movie Title' RETURN avg(r.rating) AS userRating
3. Question: Get movies for a genre?
   Cypher: MATCH ((m:Movie)-[:IN_GENRE]->(g:Genre) WHERE g.name = 'Genre Name' RETURN m.title AS movieTitle

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], 
    template=cypher_template
)