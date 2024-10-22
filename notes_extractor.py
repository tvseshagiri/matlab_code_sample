from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class MeetingNotes(BaseModel):
    meeting_date_time: datetime = Field(description="The date and time of the meeting")
    competitors: Optional[List[str]] = Field(
        description="The list of competitors mentioned in the notes"
    )
    action_items: Optional[List[str]] = Field(
        description="The list of action items mentioned in the notes"
    )
    critical_items: Optional[List[str]] = Field(
        description="The list of critical items mentioned in the notes"
    )
    ifx_participants: Optional[List[str]] = Field(
        description="The list of IFX participants mentioned in the notes"
    )
    customer_contact: Optional[str] = Field(
        description="The customer contact mentioned in the notes"
    )
    highlight_lowlights: Optional[List[str]] = Field(
        description="The list of highlight lowlights mentioned in the notes"
    )
    other_items: Optional[List[str]] = Field(
        description="The list of other items like risks etc mentioned in the notes"
    )


from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview", temperature=0
).with_structured_output(MeetingNotes)

resp = llm.invoke(
    """
    Date: 2023-09-23 12:34 PM
    Agenda: 60KW Charger GAN Promption
    60KW Charger Project Status 5th Board release on the week 8th 2015. 
    PFC 6X120 R 20 MH.6X9574                  
    """
)

print(resp)
