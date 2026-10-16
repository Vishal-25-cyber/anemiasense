"""
AnemiaSense Database & History Manager
Connects to MongoDB (with automatic local JSON fallback)
Stores patient screening records, biomarkers, telemetry, and thumbnails.
"""

import os
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

try:
    import pymongo
    from bson import ObjectId
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False
    ObjectId = None

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FALLBACK_FILE = DATA_DIR / "history.json"

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = os.getenv("MONGODB_DB", "anemiasense")
MONGO_COLLECTION_NAME = "screenings"


def create_thumbnail_base64(img_rgb: np.ndarray, max_dim: int = 300) -> str:
    """Creates a lightweight compressed JPEG base64 data URI for storage and previews."""
    if img_rgb is None or img_rgb.size == 0:
        return ""
    try:
        h, w = img_rgb.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"[AnemiaSense DB] Thumbnail error: {e}")
        return ""


class HistoryDatabase:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._connected = False
        self._init_mongo()

    def _init_mongo(self):
        if not HAS_PYMONGO:
            print("[AnemiaSense DB] pymongo not installed, using JSON file store.")
            return

        try:
            self.client = pymongo.MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000
            )
            # Verify connectivity
            self.client.admin.command('ping')
            self.db = self.client[MONGO_DB_NAME]
            self.collection = self.db[MONGO_COLLECTION_NAME]
            # Create indexes for fast lookup and chronological sorting
            self.collection.create_index([("created_at", pymongo.DESCENDING)])
            self.collection.create_index([("report_id", pymongo.ASCENDING)], unique=True)
            self._connected = True
            print(f"[AnemiaSense DB] Connected successfully to MongoDB at {MONGO_URI} (db: {MONGO_DB_NAME})")
        except Exception as e:
            self._connected = False
            print(f"[AnemiaSense DB] MongoDB connection error ({e}); using JSON fallback at {FALLBACK_FILE}")

    def is_connected(self) -> bool:
        if not self._connected or self.client is None:
            # Retry connection once if previously disconnected
            self._init_mongo()
        return self._connected

    def get_status(self) -> Dict[str, Any]:
        connected = self.is_connected()
        total_count = self.count_screenings()
        return {
            "mongodb_connected": connected,
            "mongodb_uri": MONGO_URI if connected else None,
            "database": MONGO_DB_NAME if connected else None,
            "collection": MONGO_COLLECTION_NAME if connected else None,
            "storage_type": "MongoDB (Primary)" if connected else "Local JSON Store (Fallback)",
            "total_records": total_count,
        }

    def _generate_report_id(self) -> str:
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        rand_suffix = os.urandom(3).hex().upper()
        return f"AS-{date_str}-{rand_suffix}"

    def save_screening(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves a screening record into MongoDB (and fallback JSON).
        Returns the saved summary including report_id and record_id.
        """
        now = datetime.now()
        report_id = data.get("report_id") or self._generate_report_id()

        doc = {
            "report_id":         report_id,
            "created_at":        now.isoformat(),
            "created_at_human":  now.strftime("%b %d, %Y • %I:%M %p"),
            "mode":              data.get("mode", "eye"),
            "patient":           data.get("patient", {}),
            "screening_results": data.get("screening_results", {}),
            "quality_assurance": data.get("quality_assurance", {}),
            "telemetry":         data.get("telemetry", {}),
            "nutrition_theme":   (data.get("nutritional_guidance") or {}).get("plan_theme", ""),
            "crop_coords":       data.get("crop_coords", None),
            "has_crop":          bool(data.get("crop_coords")),
            "image_thumbnail":   data.get("image_thumbnail", ""),
            "gradcam_base64":    data.get("gradcam_base64", ""),
            "disclaimer":        data.get("disclaimer", "Research screening estimate only."),
        }

        record_id = None

        # 1. Save to MongoDB
        if self.is_connected() and self.collection is not None:
            try:
                res = self.collection.insert_one(doc.copy())
                record_id = str(res.inserted_id)
            except Exception as e:
                print(f"[AnemiaSense DB] MongoDB insert error: {e}")

        # 2. Backup to JSON File
        try:
            records = self._read_fallback_json()
            json_doc = doc.copy()
            json_doc["_id"] = record_id or f"local_{int(now.timestamp()*1000)}"
            records.insert(0, json_doc)
            # Keep up to 200 items in fallback JSON
            if len(records) > 200:
                records = records[:200]
            with open(FALLBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            if not record_id:
                record_id = json_doc["_id"]
        except Exception as e:
            print(f"[AnemiaSense DB] Fallback JSON write error: {e}")

        return {
            "record_id":  record_id or report_id,
            "report_id":  report_id,
            "created_at": doc["created_at"],
            "created_at_human": doc["created_at_human"],
            "mongodb_stored": self.is_connected(),
        }

    def _read_fallback_json(self) -> List[Dict[str, Any]]:
        if not FALLBACK_FILE.exists():
            return []
        try:
            with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def count_screenings(self) -> int:
        if self.is_connected() and self.collection is not None:
            try:
                return self.collection.count_documents({})
            except Exception:
                pass
        return len(self._read_fallback_json())

    def list_screenings(self, limit: int = 50, skip: int = 0, severity: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists screenings, sorted chronologically descending."""
        if self.is_connected() and self.collection is not None:
            try:
                query: Dict[str, Any] = {}
                if severity and severity.upper() != "ALL":
                    query["screening_results.severity_level"] = severity.upper()
                if search:
                    regex = {"$regex": search, "$options": "i"}
                    query["$or"] = [
                        {"report_id": regex},
                        {"patient.gender": regex},
                        {"mode": regex},
                    ]

                cursor = self.collection.find(query).sort("created_at", pymongo.DESCENDING).skip(skip).limit(limit)
                results = []
                for doc in cursor:
                    doc["_id"] = str(doc["_id"])
                    results.append(doc)
                return results
            except Exception as e:
                print(f"[AnemiaSense DB] MongoDB list error ({e}), falling back to JSON.")

        # Fallback reading
        records = self._read_fallback_json()
        if severity and severity.upper() != "ALL":
            records = [r for r in records if r.get("screening_results", {}).get("severity_level") == severity.upper()]
        if search:
            s_lower = search.lower()
            records = [r for r in records if s_lower in r.get("report_id", "").lower() or s_lower in r.get("mode", "").lower()]
        return records[skip : skip + limit]

    def get_screening(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Finds screening by report_id or MongoDB ObjectId."""
        if self.is_connected() and self.collection is not None:
            try:
                doc = self.collection.find_one({"report_id": identifier})
                if not doc and HAS_PYMONGO and ObjectId and ObjectId.is_valid(identifier):
                    doc = self.collection.find_one({"_id": ObjectId(identifier)})
                if doc:
                    doc["_id"] = str(doc["_id"])
                    return doc
            except Exception as e:
                print(f"[AnemiaSense DB] MongoDB get error: {e}")

        # Fallback search
        records = self._read_fallback_json()
        for r in records:
            if r.get("report_id") == identifier or str(r.get("_id")) == identifier:
                return r
        return None

    def delete_screening(self, identifier: str) -> bool:
        """Deletes a screening by report_id or MongoDB ObjectId."""
        deleted = False
        if self.is_connected() and self.collection is not None:
            try:
                res = self.collection.delete_one({"report_id": identifier})
                if res.deleted_count > 0:
                    deleted = True
                elif HAS_PYMONGO and ObjectId and ObjectId.is_valid(identifier):
                    res = self.collection.delete_one({"_id": ObjectId(identifier)})
                    if res.deleted_count > 0:
                        deleted = True
            except Exception as e:
                print(f"[AnemiaSense DB] MongoDB delete error: {e}")

        # Also remove from fallback JSON
        try:
            records = self._read_fallback_json()
            initial_len = len(records)
            records = [r for r in records if r.get("report_id") != identifier and str(r.get("_id")) != identifier]
            if len(records) < initial_len:
                deleted = True
                with open(FALLBACK_FILE, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[AnemiaSense DB] Fallback JSON delete error: {e}")

        return deleted

    def get_statistics(self) -> Dict[str, Any]:
        """Aggregates summary statistics for display."""
        screenings = self.list_screenings(limit=1000)
        total = len(screenings)
        if total == 0:
            return {
                "total_screenings": 0,
                "average_hb": 0.0,
                "anemia_count": 0,
                "anemia_percentage": 0.0,
                "severity_counts": {
                    "NORMAL": 0,
                    "MILD_ANEMIA": 0,
                    "MODERATE_ANEMIA": 0,
                    "SEVERE_ANEMIA": 0,
                },
                "mongodb_connected": self.is_connected(),
            }

        hb_vals = []
        sev_counts = {"NORMAL": 0, "MILD_ANEMIA": 0, "MODERATE_ANEMIA": 0, "SEVERE_ANEMIA": 0}
        anemia_count = 0

        for s in screenings:
            res = s.get("screening_results", {})
            hb = res.get("estimated_hemoglobin_g_dl")
            if hb is not None and isinstance(hb, (int, float)):
                hb_vals.append(hb)
            sev = res.get("severity_level", "NORMAL")
            if sev in sev_counts:
                sev_counts[sev] += 1
            if sev != "NORMAL":
                anemia_count += 1

        avg_hb = round(float(np.mean(hb_vals)), 2) if hb_vals else 0.0
        anemia_pct = round((anemia_count / total) * 100, 1)

        return {
            "total_screenings": total,
            "average_hb": avg_hb,
            "anemia_count": anemia_count,
            "anemia_percentage": anemia_pct,
            "severity_counts": sev_counts,
            "mongodb_connected": self.is_connected(),
        }


# Singleton instance
db_manager = HistoryDatabase()
