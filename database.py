from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://postgres:K%23v8%21nZ2%24mPq9@db.yhtiigzfuaqqxbtknwjx.supabase.co:5432/postgres"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()
