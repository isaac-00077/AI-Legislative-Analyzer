from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base

from pgvector.sqlalchemy import Vector


class Bill(Base):
    __tablename__ = "bills"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    pdf_url = Column(String, unique=True)
    local_path = Column(String)
    processed = Column(Boolean, default=False)

 
    title_embedding = Column(Vector(384))

    
    chunks = relationship("Chunk", back_populates="bill")
    
    summary = Column(Text)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)

    bill_id = Column(Integer, ForeignKey("bills.id"))

    original_text = Column(Text)
    compressed_text = Column(Text)

    
    embedding = Column(Vector(384))

    bill = relationship("Bill", back_populates="chunks")