import os
from typing import TypedDict ,Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph , START , END

load_dotenv()
llm = ChatOpenAI(
    model = "gpt-5",
    temperature=0.1
)

def merge_score_dicts( existing : dict , new_update : dict) -> dict:
    if existing is None:
        return new_update
    
    return {**existing , **new_update}

# cretae a state

class AnalyzerState(TypedDict):
    raw_text : str
    #score of all splited node ex A,B,C safety score overwrite agian and again there fore we use reducer 
    safety_score : Annotated[dict[str,int] , merge_score_dicts]
    
    
    
#nodes
def toxicity_node(state: AnalyzerState):
    
    print("\n😡 [Branch 1] Analyzing Toxicity and Hate Comment or Speech .....")
    prompt = f"""
You are a toxicity and hate-speech detection system.

Analyze the following text and assign a toxicity score from 0 to 100.

0 = Completely safe/neutral
10-20 = Very mild negativity or rudeness
30-40 = Mildly offensive or insulting
50-60 = Clearly toxic, abusive, or strongly offensive
70-80 = Highly toxic, hateful, threatening, or severely abusive
90-100 = Extremely toxic, hateful, threatening, or violent

Consider insults, harassment, aggressive profanity, hate speech,
threats, dehumanizing language, and severe abuse.

Do not give a high score just because a swear word appears.
Consider the context and intended use.

IMPORTANT:
Return ONLY one integer from 0 to 100.
Do not return any explanation or additional text.

Text:
{state["raw_text"]}
"""

    response = llm.invoke(prompt)

    print("😡 RAW TOXICITY RESPONSE:", repr(response.content))

    try:
        score = int(response.content.strip())
        score = max(0, min(100, score))
    except ValueError:
        score = 0

    return {
        "safety_score": {
            "toxicity_level": score
        }
    }
    
    
# nodes
def copyright_node(state: AnalyzerState):

    print("\n©️ [Branch 2] Analyzing Copyright and Potential Copyright Infringement .....")

    prompt = f"""
You are a copyright detection and content analysis system.

Analyze the following text and assign a copyright infringement risk score
from 0 to 100.

🟢 0 = No apparent copyright concern; original, generic, or factual content
🟢 10-20 = Very low copyright risk
🟡 30-40 = Some similarity or potentially copyrighted expression
🟠 50-60 = Moderate copyright infringement risk
🔴 70-80 = High likelihood of copyrighted content being reproduced
🔴 90-100 = Extremely high likelihood of substantial copyrighted content being reproduced

Consider:
- Direct reproduction of copyrighted text
- Long or substantial portions of books, articles, scripts, lyrics, or other creative works
- Requests or content that reproduce protected material substantially
- Distinctive creative expressions that appear copied from an existing work
- Large verbatim excerpts from copyrighted sources

Do NOT consider the following as copyright infringement by themselves:
- General facts or information
- Common knowledge
- Short generic phrases
- Original ideas without copied expression
- User-generated original content
- Proper attribution alone as proof of infringement

Focus on the actual text and the likelihood that it reproduces
protected creative expression.

IMPORTANT:
- Return ONLY one integer from 0 to 100.
- Do NOT return any explanation.
- Do NOT return a percentage sign.
- Do NOT return any additional text.

Text to analyze:
{state["raw_text"]}
"""

    response = llm.invoke(prompt)

    print("😡 RAW COPYRIGHT RESPONSE:", repr(response.content))

    try:
        score = int(response.content.strip())
        score = max(0, min(100, score))
    except ValueError:
        score = 0

    return {
        "safety_score": {
            "copyright_level": score
        }
    }
    

# nodes
def cultural_guide_node(state: AnalyzerState):

    print("\n🌍 [Branch 3] Analyzing Cultural Sensitivity and Respect .....")

    prompt = f"""
You are a cultural sensitivity and cultural respect analysis system.

Analyze the following text and assign a cultural sensitivity risk score
from 0 to 100.

🟢 0 = Completely respectful, neutral, and culturally appropriate
🟢 10-20 = Very low cultural sensitivity concern
🟡 30-40 = Mild cultural insensitivity, stereotype, or misunderstanding
🟠 50-60 = Clearly culturally insensitive, disrespectful, or stereotypical
🔴 70-80 = Highly disrespectful, offensive, or discriminatory toward a culture
🔴 90-100 = Extremely hateful, degrading, or dehumanizing toward a cultural,
religious, ethnic, or traditional group

Consider:
- Cultural stereotypes
- Insults toward cultures or traditions
- Disrespectful descriptions of cultural practices
- Mocking or degrading cultural identities
- Cultural appropriation concerns
- Generalizations about people based on their culture
- Discriminatory or hateful statements toward cultural groups
- Offensive use of cultural symbols, traditions, or practices

IMPORTANT:
- Consider the context and intent of the text.
- Do not flag respectful discussion, education, criticism, or neutral
  descriptions of cultural practices as highly problematic.
- Distinguish cultural criticism from attacks against people or groups.
- Do not assign a high score simply because a culture, religion, ethnicity,
  or tradition is mentioned.

Return a score based on the severity of the cultural sensitivity concern.

IMPORTANT:
- Return ONLY one integer from 0 to 100.
- Do NOT return any explanation.
- Do NOT return a percentage sign.
- Do NOT return any additional text.

Text to analyze:
{state["raw_text"]}
"""

    response = llm.invoke(prompt)

    print("😡 RAW CULTURAL RESPONSE:", repr(response.content))

    try:
        score = int(response.content.strip())
        score = max(0, min(100, score))
    except ValueError:
        score = 0

    return {
        "safety_score": {
            "cultural_sensitivity_level": score
        }
    }
    
    
    
# connect

builder = StateGraph(AnalyzerState)

builder.add_node("toxicity_node" , toxicity_node )

builder.add_node("copyright_check" , copyright_node)

builder.add_node("cultural_node" , cultural_guide_node)


# now use fan in and fan out

builder.add_edge(START , "toxicity_node")

builder.add_edge(START , "copyright_check")

builder.add_edge(START , "cultural_node")

builder.add_edge("toxicity_node" , END)

builder.add_edge("copyright_check" , END)

builder.add_edge("cultural_node" , END)


app = builder.compile()

sample_script = """
You people are completely stupid and worthless. Your entire culture is
backward and inferior, and everyone from your country is lazy and
uncivilized. I hate people like you and they should not be respected.

Here is a large portion of a copyrighted article reproduced word for word:
"The quick brown fox jumps over the lazy dog. The quick brown fox jumps
over the lazy dog. This is an example of reproduced copyrighted wording
that is being copied directly rather than explained in my own words."
"""

intial_state = {
    "raw_text" : sample_script,
    "safety_score" : {} # Intialiazed as an empty dict
}
    
final_test = app.invoke(intial_state)


print(final_test)