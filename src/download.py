"""Download the four PhysioNet databases with the wfdb package."""
import argparse
import wfdb

DBS = {"mitdb": "mitdb", "incartdb": "incartdb", "svdb": "svdb", "sddb": "sddb"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=list(DBS), required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    wfdb.dl_database(DBS[a.db], dl_dir=f"{a.out}/{a.db}")
    print(f"downloaded {a.db} to {a.out}/{a.db}")
