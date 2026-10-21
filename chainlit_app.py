import uuid
import chainlit as cl
from numpy import random
import graph
from langchain_core.messages import HumanMessage


@cl.on_chat_start
async def start():
    current_id = graph.current_investorID

    cl.user_session.set("thread_id", str(uuid.uuid4()))
    
    await cl.Message(content=f"Hi investor {current_id}, how can I help you today?").send()


@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}
    
    # Call the graph
    result = await graph.graph.ainvoke(
        {"messages": [HumanMessage(content=message.content)]},
        config
    )
    
    # Send response back to Chainlit UI
    await cl.Message(content=result["messages"][-1].content).send()
