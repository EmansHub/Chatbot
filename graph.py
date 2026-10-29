import os
import psycopg2
import random
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, ToolMessage, AIMessage 
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
    temperature=0.0,
    )

'''
llm = ChatAnthropic(
    model="claude-haiku-5-5",
    temperature=0.0,
    max_tokens=2048,
)
'''

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

class State(MessagesState):
    summary: str    

# Nodes
sys_msg = SystemMessage(content="""You are an investment information assistant backed by a PostgreSQL database containing investor profiles and fund data.

What you can do:
- Answer basic investor questions using SQL-based retrieval.
- Retrieve investor-specific information: risk profile, portfolio holdings, allocation, and investment objectives.
- Retrieve available funds with their risk levels, categories, and key performance metrics.
- Handle basic chit-chat and general investment-related questions (no database lookup needed for these).
- Give brief, factual commentary based on the retrieved investor and fund data.

How to query:
- For any database question, call sql_db_list_tables first, then sql_db_schema for the relevant tables, before writing a query.
- then, ALWAYS call sql_db_query_checker on your drafted query before executing it with sql_db_query.
- Write PostgreSQL syntax. Use double quotes for mixed-case table or column names.
- Only run read-only SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, or ALTER.
- If a query errors, read the error, fix the query, and retry.

Rules:
- Answer only from retrieved data. If the data isn't in the database, say so. Don't guess or invent figures.
- LANGUAGE MIRRORING (STRICT): Always respond in the exact same language the user used. Applies to all output including refusals and "no data" replies.
For Arabic, respond strictly in Gulf/Khaleeji dialect with zero Latin characters anywhere in the response.
- Do NOT make investment recommendations or tell the user what to buy, sell, or hold. Do not give any future predictions or calculations. You may describe the data, compare funds factually, and note how a fund's risk level relates to an investor's stated risk profile. 
- When giving commentary, keep it neutral and tie it to the data you retrieved.
- Only share data for the investor being asked about. Don't expose other investors' information.
- Keep answers clear and concise.
- NEVER mention SQL, database tables, schemas, column names, or backend processes (like RLS) to the user. Speak naturally about their portfolio, funds, and profile.
If the user explicitly asks for the database schema, table structure, or available fields, you MUST politely refuse and DO NOT call any tools to inspect them.
- STRICTLY LIMIT your responses to the scope of this app (investments, portfolios, and funds). If the user asks an unrelated question (e.g., recipes, coding), 
politely refuse and redirect them to your purpose. Do not answer off-topic requests.
- Align investor risk profiles with fund risk levels using this strict mapping:
  - Very Conservative: Match "Very Low" and "Low" funds.
  - Conservative: Match "Very Low" and "Low" funds.
  - Moderate: Match "Low", "Moderate", and "High" funds.
  - Aggressive: Match "Moderate", "High", and "Very High" funds.
  - Very Aggressive: Match "High" and "Very High" funds.
  - When asked what funds align with a profile, ALWAYS query the database for the mapped risk levels 
  using an OR condition (e.g., WHERE risk_level IN ('Moderate', 'High', 'Very High')).
  - If the user mixes two languages, answer each part in the same language it was asked. For example, if they ask "What funds match my risk profile وايش هي المخاطر", answer the first part in English and the second part in Arabic.
  - Do not give any id number. For example, do not say fund 1 or fund 2. Instead, use the fund name or a descriptive phrase. If the user asks for a fund by number, politely refuse and ask for the fund name instead. ID's are internal database identifiers and should never be exposed to the user.
""")


# Node
def assistant(state: State):
    summary = state.get("summary", "")
    system_messages = [sys_msg]
    if summary:
        system_messages.append(
            SystemMessage(content=f"Summary of conversation earlier: {summary}")
        )
    return {"messages": [llm_with_tools.invoke(system_messages + state["messages"])]}


# Summarization Node
# How many full user turns to keep verbatim. Everything older gets summarized.
N_KEEP_TURNS = 90000

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    messages = state["messages"]

    # Indices of real user turns (exclude the reviewer's injected HumanMessage)
    human_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, HumanMessage) and not m.content.startswith("[Reviewer]")
    ]

    if len(human_indices) <= N_KEEP_TURNS:
        return {}

    # Cut AT a real HumanMessage — guarantees no tool call / ToolMessage is split
    cut_index = human_indices[-N_KEEP_TURNS]
    to_summarize = messages[:cut_index]
    kept = messages[cut_index:]

    # ---- decide what tool noise / reviewer noise to strip from KEPT turns ----
    strip_ids = set()
    for i, m in enumerate(kept):
        # (a) Strip tool outputs
        if isinstance(m, ToolMessage):
            strip_ids.add(m.id)
            continue

        # (b) Strip AI messages that only exist to request tool calls
        if isinstance(m, AIMessage) and m.tool_calls:
            strip_ids.add(m.id)
            continue

        # (c) Strip [Reviewer] messages AND the rejected AI response right before them
        if isinstance(m, HumanMessage) and m.content.startswith("[Reviewer]"):
            strip_ids.add(m.id)
            if (
                i > 0
                and isinstance(kept[i - 1], AIMessage)
                and not kept[i - 1].tool_calls
            ):
                strip_ids.add(kept[i - 1].id)

    # ---- Summarize the old turns ----
    if summary:
        prompt = (
            f"This is summary of the conversation to date: {summary}\n\n"
            '''Extend the summary by taking into account the new messages above,
            do not provide any commentary, only summarize. Do not delete information 
            from the old summary.'''
        )
    else:
        prompt = '''Create a summary of the conversation above; do not provide any extra commentary,
        only summarize.'''

    response = llm.invoke(to_summarize + [HumanMessage(content=prompt)])

    # ---- Delete old turns + stripped noise ----
    delete_messages = [RemoveMessage(id=m.id) for m in to_summarize]
    delete_messages += [RemoveMessage(id=i) for i in strip_ids]

    return {"summary": response.content, "messages": delete_messages}


# Validator Node
check_msg = SystemMessage(content="""You are a compliance reviewer. Check the draft answer.
PASS if it only states facts or data (e.g. an investor's profile, fund details) or is basic chit-chat.
FAIL only if it tells the user what to buy, sell, hold, or invest in, or predicts or guarantees performance.
FAIL if it mentions database tables, SQL, schemas, or backend processes.
FAIL if it answers a question unrelated to investments, portfolios, or funds.
Reply with "PASS", or "FAIL: <one-sentence reason>".""")


def validator(state: State):
    last = state["messages"][-1]
    if last.tool_calls:  # tool-call turn, nothing to judge yet
        return {}
    prompt = f"""{check_msg.content}

Draft answer:
\"\"\"{last.content}\"\"\""""
    result = llm.invoke(prompt).content.strip()
    if result.upper().startswith("PASS"):
        return {}
    return {"messages": [HumanMessage(content=f"[Reviewer] {result} rewrite your response to address this feedback and strictly follow the system instructions.")]}


def route_after_check(state: State):
    last = state["messages"][-1]

    # FAIL branch: reviewer injected a HumanMessage → send back to assistant
    if isinstance(last, HumanMessage):
        return "assistant"

    # PASS branch: check whether we now exceed N_KEEP_TURNS real user turns
    turn_count = sum(
        1 for m in state["messages"]
        if isinstance(m, HumanMessage) and not m.content.startswith("[Reviewer]")
    )
    if turn_count > N_KEEP_TURNS:
        return "summarize"

    return END

# Graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("validator", validator)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",tools_condition,
    {"tools":"tools", END:"validator"}
)
builder.add_conditional_edges(
    "validator", route_after_check,
    {"assistant": "assistant", "summarize": "summarize", END: END}
)
builder.add_edge("tools", "assistant")
builder.add_edge("summarize", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
