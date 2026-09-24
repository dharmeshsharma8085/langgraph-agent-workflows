import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# state 
class pipelinestate(TypedDict, total=False):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


llm = ChatOpenAI(
    model="gpt-5",
    temperature=0.7
)


# Stage 1
def editor_node(state: pipelinestate) -> dict:
    print("Stage 1 ---------------------------------------------------\n")
    """
    Stage 1: Cleans grammar, removes typos and refines the tone.
    """

    prompt = f"""
You are a text editor.
Check the text for grammar, spelling, punctuation, and sentence structure.
Do not change the context or meaning.
Make only necessary corrections and return the edited text.

Text:
{state["raw_input"]}
"""

    response = llm.invoke(prompt)

    return {
        "edited_text": response.content.strip()
    }


# Stage 2
def scriptwriter_node(state: pipelinestate) -> dict:
    print("Stage 2 ---------------------------------------------------\n")
    """
    Stage 2: Transforms the refined content into a clear,
    engaging, and well-structured script.
    """

    prompt = f"""
You are a professional scriptwriter.
Turn the user's input into a clear, engaging, and well-structured script.
Preserve the original meaning and context.
Improve the flow and structure.
Return only the final script.

Text:
{state["edited_text"]}
"""

    response = llm.invoke(prompt)

    return {
        "script_text": response.content.strip()
    }


# Stage 3
def translator_node(state: pipelinestate) -> dict:
    
    print("Stage 3  -------------------------------------------------------\n")
    """
    Stage 3: Translates the script into natural Hinglish.
    """

    prompt = f"""
You are a professional Hinglish translator.
Convert the given script into natural, easy-to-understand Hinglish.
Preserve the original meaning, context, tone, and structure.
Do not add or remove information.
Return only the translated script.

Text:
{state["script_text"]}
"""

    response = llm.invoke(prompt)

    return {
        "final_output": response.content.strip()
    }


# Create graph
graph = StateGraph(pipelinestate)

graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)


# Edges
graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)


# Compile
app = graph.compile()


# Run
result = app.invoke({
    "raw_input": """
Artificial Intelligence is changing the way we work and learn.
In this video, we will understand how AI is transforming our daily lives.
"""
})


print("Your result is:\n")
print(result["final_output"])