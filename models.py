from sqlalchemy import Column, Integer, String, Boolean
from database import Base

class Bill(Base): #This class defines the structure of the "bills" table in the database. Each instance of this class represents a row in the "bills" table, with columns for id, title, pdf_url, local_path, and processed status.
    __tablename__ = "bills"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    pdf_url = Column(String, unique=True)
    local_path = Column(String)
    processed = Column(Boolean, default=False)