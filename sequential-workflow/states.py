# so now we aew crating a graph
# so we have to create a state

import os

#1 using type DICT most common 
from typing import TypedDict
class State(TypedDict):
    topic : str 
    summmary : str 
    score : str 
    
    
#2 USING PYDANTIC 

# it is good at data validation type checking at runtime


from pydantic import BaseModel , field_validator

class State(BaseModel):
    topic : str 
    summmary : str 
    score : str  = ""
    
    
    @field_validator 
    def score_postive(cls ,v):
        if v<0:
            raise ValueError(" Socrte must  be grater than 0 ")
        
        
        
#3 python data clasess  use very rarely


from dataclasses import dataclass , field

@dataclass
class State:
    topic : str = " "
    summary : str = " "
    score : str = " "
    messages = list = field(default_factory=list)
    
    
    
#using  langgraph


from langgraph.graph import MessagesState

class State(MessagesState):
    user_name : str 
    # messages fiel is already is include the wuth add_messages reducer
    # just add your extra field
    
    

 