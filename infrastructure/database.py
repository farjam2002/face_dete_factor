from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def init_db(config: dict, base_dir):
    # برای اینکه مدل‌های آینده ثبت شوند
    try:
        from domain import models  # noqa: F401
    except Exception:
        pass

    data_dir = base_dir / config["app"]["data_dir"]
    db_path = data_dir / config["storage"]["db_file"]

    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False
    )

    SessionLocal.configure(bind=engine)

    Base.metadata.create_all(bind=engine)

    return engine