import os
import psycopg2
import random
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

# loading environment variables from .env file
load_dotenv()

current_investorID = random.randint(1, 20)

# DB connection 
def get_connection():
    # 1. Save the connection to a variable first
    connection = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )

    connection.autocommit = True 

    # 2. Set the RLS variable (using 'app.' to match your policy)
    cursor = connection.cursor()
    cursor.execute("SELECT set_config('app.current_investorID', %s, false);", (str(current_investorID),))
    cursor.close()
    
    # 3. Now return the connection
    return connection


llm = ChatOpenAI(
     model="gpt-5.6-luna",
     reasoning_effort="none",
    )

# tools
@tool
def sql_db_list_tables() -> str:
    """Input is an empty string, output is a comma-separated list of tables in the database."""
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        tables = [row[0] for row in cursor.fetchall()]
        return ", ".join(tables)
    finally:
        connection.close()


@tool
def sql_db_schema(table_names: str) -> str:
    """Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling sql_db_list_tables first!
    Example Input: table1, table2, table3"""
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        valid_tables = {row[0] for row in cursor.fetchall()}
        results = []
        for table in table_names.split(","):
            table = table.strip()
            if table not in valid_tables:
                results.append(f"Error: table_names {{{table!r}}} not found in database")
                continue
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
            """, (table,))
            columns = cursor.fetchall()
            if columns:
                col_defs = ",\n".join(
                    f'    "{name}" {dtype}' + ("" if nullable == "YES" else " NOT NULL")
                    for name, dtype, nullable in columns
                )
                results.append(f'CREATE TABLE "{table}" (\n{col_defs}\n)')
                try:
                    quoted_table = '"' + table.replace('"', '""') + '"'
                    cursor.execute(f"SELECT * FROM {quoted_table} LIMIT 3;")
                    rows = cursor.fetchall()
                    if rows:
                        col_names = [description[0] for description in cursor.description]
                        results.append(
                            f"/*\n3 rows from {table} table:\n"
                            + "\t".join(col_names)
                            + "\n"
                            + "\n".join("\t".join(str(x) for x in row) for row in rows)
                            + "\n*/"
                        )
                except Exception as e:
                    connection.rollback()
                    results.append(f"Error fetching sample rows: {e}")
        return "\n\n".join(results)
    finally:
        connection.close()

@tool

def sql_db_query(query: str) -> str:
    """Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    If you encounter an issue with 'column "xxxx" does not exist', use sql_db_schema to query the correct table fields."""
    connection = get_connection()

    try:
        cursor = connection.cursor()
        cursor.execute(query)
        res = cursor.fetchall()
        return str(res)
    
    except Exception as e:
        return f"Error: {e}"

    finally:
        connection.close()


@tool
def sql_db_query_checker(query: str) -> str:
    """Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!"""
    trigger_prompt = """{query}
Double check the PostgreSQL query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers (mixed-case names need double quotes)
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Columns in SELECT missing from GROUP BY
- Case-sensitive string comparisons (consider ILIKE)

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """.format(query=query)

    response = llm.invoke(trigger_prompt)
    return response.content.strip()


tools = [sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker] 

# Define LLM with bound tools
llm_with_tools = llm.bind_tools(tools)
    

# Nodes
sys_msg = SystemMessage(content="""You are an investment information assistant backed by a PostgreSQL database containing investor profiles and fund data.

What you can do:
- Answer basic investor questions using SQL-based retrieval.
- Retrieve investor-specific information: risk profile, portfolio holdings, allocation, and investment objectives.
- Retrieve available funds with their risk levels, categories, and key performance metrics.
- Handle basic chit-chat and general investment-related questions (no database lookup needed for these).
- Give brief, factual commentary based on the retrieved investor and fund data.

Database tables:
- funds: available funds (name, category, risk level, minimum investment, performance metrics)
- investors: investor profiles (risk profile, investment objectives)
- investor_portfolios: each investor's holdings and allocation
- transactions: investor transaction history

How to query:
- For any database question, call sql_db_list_tables first, then sql_db_schema for the relevant tables, before writing a query.
- then, ALWAYS call sql_db_query_checker on your drafted query before executing it with sql_db_query.
- Write PostgreSQL syntax. Use double quotes for mixed-case table or column names.
- Only run read-only SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, or ALTER.
- Limit results to 10 rows unless asked otherwise.
- If a query errors, read the error, fix the query, and retry.

Rules:
- Answer only from retrieved data. If the data isn't in the database, say so. Don't guess or invent figures.
- Do NOT make investment recommendations or tell the user what to buy, sell, or hold. You may describe the data, compare funds factually, and note how a fund's risk level relates to an investor's stated risk profile.
- When giving commentary, keep it neutral and tie it to the data you retrieved.
- Only share data for the investor being asked about. Don't expose other investors' information.
- Keep answers clear and concise.
- Do NOT ask the user for their investor ID. The database automatically restricts all data to the current logged-in investor via Row-Level Security. 
""")


def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


check_msg = SystemMessage(content="""You are a compliance reviewer. Check the draft answer.
PASS if it only states facts or data (e.g. an investor's profile, fund details) or is basic chit-chat.
FAIL only if it tells the user what to buy, sell, hold, or invest in, or predicts or guarantees performance.
Reply with "PASS", or "FAIL: <one-sentence reason>".""")


def validator(state: MessagesState):
    last = state["messages"][-1]
    if last.tool_calls:  # tool-call turn, nothing to judge yet
        return {}
    prompt = f"""{check_msg.content}

Draft answer:
\"\"\"{last.content}\"\"\""""
    result = llm.invoke(prompt).content.strip()
    if result.upper().startswith("PASS"):
        return {}
    return {"messages": [HumanMessage(content=f"[Reviewer] {result} Rewrite using only the retrieved data.")]}


def route_after_check(state: MessagesState):
    return "assistant" if isinstance(state["messages"][-1], HumanMessage) else END

# Graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("validator", validator)

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",tools_condition,
    {"tools":"tools", END:"validator"}
)
builder.add_conditional_edges(
    "validator", route_after_check,
    {"assistant": "assistant", END: END})
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
