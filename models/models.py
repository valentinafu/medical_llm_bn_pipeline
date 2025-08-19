from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy import JSON

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)  # optionally hashed
    role = Column(String)  # 'patient' or 'expert'
    created_at = Column(DateTime, default=datetime.utcnow)

class Evaluation(Base):
    __tablename__ = 'evaluations'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    symptoms = Column(Text)
    prediction = Column(Text)
    target_node = Column(String)
    category = Column(String, nullable=True)
    status = Column(String, default='pending')
    llm_response = Column(JSON, nullable=True)  # or Column(Text)


class ExpertFeedback(Base):
    __tablename__ = 'expert_feedback'
    id = Column(Integer, primary_key=True)
    evaluation_id = Column(Integer, ForeignKey('evaluations.id'))
    expert_id = Column(Integer, ForeignKey('users.id'))
    adjusted_probs = Column(Text)
    comment = Column(Text)
    submitted_at = Column(DateTime, default=datetime.utcnow)
