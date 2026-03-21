from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./bills.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}) #engine is like road connecting system to database file (bills.db)

SessionLocal = sessionmaker(bind=engine) #Its like a vehicle(session) which we use on the road (engine) to interact with the database. We create sessions from this SessionLocal to perform database operations.

Base = declarative_base() #All tables are in this Base. We will define our models (tables) by inheriting from this Base class. It provides the necessary functionality to create and manage database tables based on our model definitions.