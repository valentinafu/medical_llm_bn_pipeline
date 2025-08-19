from db.database import SessionLocal
from models.models import Evaluation

def get_all_evaluations():
    db = SessionLocal()
    history = db.query(Evaluation)\
        .order_by(Evaluation.timestamp.desc())\
        .all()
    db.close()
    return history

def get_evaluation_by_id(eval_id):
    db = SessionLocal()
    evaluation = db.query(Evaluation).filter(Evaluation.id == eval_id).first()
    db.close()
    return evaluation
