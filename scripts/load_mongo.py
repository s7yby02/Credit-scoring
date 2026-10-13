"""Load generated JSONL files into MongoDB. Idempotent: rerunning replaces documents by _id.

Usage: python scripts/load_mongo.py data/train
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from pymongo import ASCENDING, MongoClient, ReplaceOne

DATE_FIELDS = {"accounts": ["fetched_at"], "applications": ["application_date"], "outcomes": ["outcome_date"]}
BATCH = 200


def load(folder: Path, db) -> None:
    for collection, date_fields in DATE_FIELDS.items():
        path = folder / f"{collection}.jsonl"
        if not path.exists():
            print(f"{collection}: no file, skipped")
            continue
        ops = []
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                for field in date_fields:
                    doc[field] = datetime.fromisoformat(doc[field])  # real BSON dates, so range queries work
                ops.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
                if len(ops) == BATCH:
                    db[collection].bulk_write(ops, ordered=False)
                    ops = []
        if ops:
            db[collection].bulk_write(ops, ordered=False)
        print(f"{collection}: {db[collection].count_documents({})} documents")


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/train")
    db = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))[os.getenv("MONGO_DB", "credit_scoring")]
    load(folder, db)
    db.accounts.create_index([("user_id", ASCENDING)])
    db.accounts.create_index([("fetched_at", ASCENDING)])
    db.applications.create_index([("application_date", ASCENDING)])


if __name__ == "__main__":
    main()
