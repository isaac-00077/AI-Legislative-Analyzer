import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


# Use DATABASE_URL from the environment so the same code works
# locally and on Render/Supabase. Fall back to a sensible local
# default if the variable is not set.
DATABASE_URL = os.getenv(
	"DATABASE_URL",
	"postgresql://postgres:postgres@localhost/bills_db",
)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()
