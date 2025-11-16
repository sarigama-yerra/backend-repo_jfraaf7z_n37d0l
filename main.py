import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents
from schemas import User, Project, Bug, Commit, ALL_MODELS

# -----------------------------
# App & Middleware
# -----------------------------
app = FastAPI(title="BugSage API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers
# -----------------------------
from bson import ObjectId

def oid(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID")


def to_obj(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = {**doc}
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    # Convert datetime to isoformat
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        if isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    new_list.append(to_obj(item))
                elif isinstance(item, datetime):
                    new_list.append(item.isoformat())
                else:
                    new_list.append(item)
            d[k] = new_list
    return d


# -----------------------------
# Root & Health
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "BugSage Backend is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    return response


# -----------------------------
# Schema Introspection (for viewers)
# -----------------------------
@app.get("/schema")
def get_schema():
    return {name: model.model_json_schema() for name, model in ALL_MODELS.items()}


# -----------------------------
# Projects
# -----------------------------
class CreateProject(BaseModel):
    name: str
    key: str
    description: Optional[str] = None
    repo_url: Optional[str] = None


@app.post("/api/projects")
def create_project(payload: CreateProject):
    data = Project(**payload.model_dump())
    project_id = create_document("project", data)
    doc = db.project.find_one({"_id": ObjectId(project_id)})
    return to_obj(doc)


@app.get("/api/projects")
def list_projects():
    docs = db.project.find().limit(100)
    return [to_obj(d) for d in docs]


# -----------------------------
# Bugs
# -----------------------------
class CreateBug(BaseModel):
    title: str
    description: str
    project_id: Optional[str] = None
    reporter_id: Optional[str] = None
    assignee_id: Optional[str] = None
    priority: str = Field("medium")
    severity: str = Field("minor")
    module_path: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    steps_to_reproduce: Optional[str] = None
    environment: Optional[str] = None
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    logs: Optional[str] = None


@app.post("/api/bugs")
def create_bug(payload: CreateBug):
    bug = Bug(**payload.model_dump())
    # initialize history
    bug_dict = bug.model_dump()
    now = datetime.now(timezone.utc)
    bug_dict.update({
        "status": "triage",
        "history": [{
            "from_status": None,
            "to_status": "triage",
            "at": now,
            "by": bug_dict.get("reporter_id")
        }],
        "reopened_count": 0,
        "created_at": now,
        "updated_at": now,
    })
    bug_id = db.bug.insert_one(bug_dict).inserted_id
    doc = db.bug.find_one({"_id": bug_id})
    return to_obj(doc)


@app.get("/api/bugs")
def list_bugs(project_id: Optional[str] = None, status: Optional[str] = None, assignee_id: Optional[str] = None, q: Optional[str] = None):
    filt: Dict[str, Any] = {}
    if project_id:
        filt["project_id"] = project_id
    if status:
        filt["status"] = status
    if assignee_id:
        filt["assignee_id"] = assignee_id
    if q:
        filt["$or"] = [
            {"title": {"$regex": q, "$options": "i"}},
            {"description": {"$regex": q, "$options": "i"}},
            {"tags": {"$regex": q, "$options": "i"}},
        ]
    docs = db.bug.find(filt).sort("created_at", -1).limit(200)
    return [to_obj(d) for d in docs]


class UpdateBug(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    assignee_id: Optional[str] = None
    priority: Optional[str] = None
    severity: Optional[str] = None
    module_path: Optional[str] = None
    tags: Optional[List[str]] = None
    steps_to_reproduce: Optional[str] = None
    environment: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    logs: Optional[str] = None


@app.patch("/api/bugs/{bug_id}")
def update_bug(bug_id: str, payload: UpdateBug):
    update = {k: v for k, v in payload.model_dump(exclude_none=True).items()}
    if not update:
        return to_obj(db.bug.find_one({"_id": oid(bug_id)}))
    update["updated_at"] = datetime.now(timezone.utc)
    res = db.bug.update_one({"_id": oid(bug_id)}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Bug not found")
    return to_obj(db.bug.find_one({"_id": oid(bug_id)}))


class TransitionPayload(BaseModel):
    to_status: str
    by: Optional[str] = None
    note: Optional[str] = None


@app.post("/api/bugs/{bug_id}/transition")
def transition_bug(bug_id: str, payload: TransitionPayload):
    bug = db.bug.find_one({"_id": oid(bug_id)})
    if not bug:
        raise HTTPException(status_code=404, detail="Bug not found")
    now = datetime.now(timezone.utc)
    from_status = bug.get("status")
    history_item = {
        "from_status": from_status,
        "to_status": payload.to_status,
        "at": now,
        "by": payload.by,
        "note": payload.note,
    }
    update: Dict[str, Any] = {
        "status": payload.to_status,
        "updated_at": now,
    }
    if payload.to_status == "closed":
        update["resolved_at"] = now
    if payload.to_status == "reopened":
        update["reopened_count"] = int(bug.get("reopened_count", 0)) + 1
    db.bug.update_one({"_id": oid(bug_id)}, {"$set": update, "$push": {"history": history_item}})
    return to_obj(db.bug.find_one({"_id": oid(bug_id)}))


# -----------------------------
# Commits Ingest (for analytics/prediction)
# -----------------------------
class IngestCommit(BaseModel):
    project_id: Optional[str] = None
    module_path: Optional[str] = None
    author: Optional[str] = None
    message: str
    additions: int = 0
    deletions: int = 0
    files: List[str] = Field(default_factory=list)


@app.post("/api/commits")
def ingest_commit(payload: IngestCommit):
    commit = Commit(**payload.model_dump())
    commit_dict = commit.model_dump()
    now = datetime.now(timezone.utc)
    commit_dict.update({"created_at": now, "updated_at": now})
    cid = db.commit.insert_one(commit_dict).inserted_id
    return to_obj(db.commit.find_one({"_id": cid}))


@app.post("/api/commits/bulk")
def ingest_commits_bulk(payload: List[IngestCommit]):
    now = datetime.now(timezone.utc)
    docs = []
    for p in payload:
        d = Commit(**p.model_dump()).model_dump()
        d.update({"created_at": now, "updated_at": now})
        docs.append(d)
    if docs:
        db.commit.insert_many(docs)
    return {"inserted": len(docs)}


# -----------------------------
# Metrics & Analytics
# -----------------------------
@app.get("/api/metrics/overview")
def metrics_overview(project_id: Optional[str] = None):
    filt = {"project_id": project_id} if project_id else {}
    bugs = list(db.bug.find(filt))
    commits = list(db.commit.find(filt))

    total = len(bugs)
    resolved = [b for b in bugs if b.get("resolved_at")]
    reopen_count = sum(int(b.get("reopened_count", 0)) for b in bugs)

    # MTTR in hours
    mttr_hours = None
    deltas = []
    for b in resolved:
        created = b.get("created_at") or b.get("history", [{}])[0].get("at")
        resolved_at = b.get("resolved_at")
        if created and resolved_at:
            try:
                delta = (resolved_at - created).total_seconds() / 3600.0
                deltas.append(delta)
            except Exception:
                pass
    if deltas:
        mttr_hours = sum(deltas) / len(deltas)

    # Bug density: bugs per 100 commits (rough proxy)
    commit_count = max(len(commits), 1)
    bug_density = (total / commit_count) * 100.0

    # Reopen rate
    reopen_rate = (sum(1 for b in bugs if int(b.get("reopened_count", 0)) > 0) / total) * 100.0 if total else 0.0

    # Trend: inflow/outflow by week (last 8 weeks)
    from collections import defaultdict
    def week_key(dt: datetime):
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    inflow: Dict[str, int] = defaultdict(int)
    outflow: Dict[str, int] = defaultdict(int)
    for b in bugs:
        c = b.get("created_at") or datetime.now(timezone.utc)
        inflow[week_key(c)] += 1
        if b.get("resolved_at"):
            outflow[week_key(b["resolved_at"])]+=1

    # Normalize to last 8 weeks
    from datetime import timedelta
    now_ts = datetime.now(timezone.utc)
    weeks = []
    for i in range(7, -1, -1):
        wdt = now_ts - timedelta(weeks=i)
        weeks.append(week_key(wdt))
    inflow_list = [{"week": w, "count": inflow.get(w, 0)} for w in weeks]
    outflow_list = [{"week": w, "count": outflow.get(w, 0)} for w in weeks]

    # Module-wise quality score: fewer bugs per commit => higher score
    module_stats: Dict[str, Dict[str, Any]] = {}
    for b in bugs:
        m = b.get("module_path") or "unknown"
        module_stats.setdefault(m, {"bugs": 0, "commits": 0})
        module_stats[m]["bugs"] += 1
    for c in commits:
        m = c.get("module_path") or "unknown"
        module_stats.setdefault(m, {"bugs": 0, "commits": 0})
        module_stats[m]["commits"] += 1
    module_scores = []
    for m, st in module_stats.items():
        commits_c = max(st["commits"], 1)
        density = st["bugs"] / commits_c
        score = max(0, 100 - density * 100)
        module_scores.append({"module": m, "quality_score": round(score, 1), "bugs": st["bugs"], "commits": st["commits"]})
    module_scores.sort(key=lambda x: x["quality_score"])  # worst first

    return {
        "totals": {"bugs": total, "resolved": len(resolved), "reopened": reopen_count},
        "mttr_hours": round(mttr_hours, 2) if mttr_hours is not None else None,
        "bug_density": round(bug_density, 2),
        "reopen_rate": round(reopen_rate, 2),
        "trend": {"inflow": inflow_list, "outflow": outflow_list},
        "module_scores": module_scores[:20],
    }


# -----------------------------
# AI-ish Predictions & Suggestions (heuristic MVP)
# -----------------------------
@app.get("/api/predictions/modules")
def predict_risky_modules(project_id: Optional[str] = None):
    filt = {"project_id": project_id} if project_id else {}
    bugs = list(db.bug.find(filt))
    commits = list(db.commit.find(filt))

    from collections import defaultdict
    stats = defaultdict(lambda: {"bugs": 0, "churn": 0, "recent": 0})
    now = datetime.now(timezone.utc)
    for b in bugs:
        m = b.get("module_path") or "unknown"
        stats[m]["bugs"] += 1
    for c in commits:
        m = c.get("module_path") or "unknown"
        churn = int(c.get("additions", 0)) + int(c.get("deletions", 0))
        stats[m]["churn"] += churn
        ts = c.get("timestamp") or c.get("created_at") or now
        # weight recency (last 30 days)
        if isinstance(ts, datetime) and (now - ts).days <= 30:
            stats[m]["recent"] += 1

    results = []
    for m, s in stats.items():
        # Simple risk score: bugs*3 + log(churn+1)*2 + recent*1
        import math
        score = s["bugs"] * 3 + math.log(max(s["churn"], 0) + 1) * 2 + s["recent"] * 1
        results.append({
            "module": m,
            "risk_score": round(score, 2),
            "bugs": s["bugs"],
            "churn": s["churn"],
            "recent_commits": s["recent"],
            "heat": "high" if score > 10 else ("medium" if score > 4 else "low"),
        })
    results.sort(key=lambda x: x["risk_score"], reverse=True)
    return {"modules": results[:25]}


class SuggestionRequest(BaseModel):
    project_id: Optional[str] = None


@app.post("/api/suggestions")
def smart_suggestions(payload: SuggestionRequest):
    project_id = payload.project_id
    filt = {"project_id": project_id} if project_id else {}
    bugs = list(db.bug.find(filt))
    commits = list(db.commit.find(filt))

    # Recommend test cases based on frequent tags/keywords
    from collections import Counter
    tag_counter = Counter()
    keyword_counter = Counter()
    for b in bugs:
        for t in b.get("tags", []):
            tag_counter[t.lower()] += 1
        text = (b.get("title", "") + " " + b.get("description", "")).lower()
        for kw in ["ui", "backend", "api", "performance", "security", "race", "null", "timeout", "memory", "db", "sql", "auth"]:
            if kw in text:
                keyword_counter[kw] += 1
    top_tags = [t for t, _ in tag_counter.most_common(5)]
    top_keywords = [k for k, _ in keyword_counter.most_common(5)]

    test_case_ideas = [
        f"Regression suite for {t} related flows" for t in top_tags
    ] + [
        f"Add {k} stress tests" if k in ["performance", "timeout", "memory"] else f"Add {k} focused unit/integration tests" for k in top_keywords
    ]

    # Suggest reviewers: most active commit authors on risky modules
    risky = predict_risky_modules(project_id).get("modules", [])
    risky_modules = {r["module"] for r in risky[:5]}
    author_counter = Counter()
    for c in commits:
        if (c.get("module_path") or "unknown") in risky_modules:
            if c.get("author"):
                author_counter[c["author"]] += 1
    suggested_reviewers = [a for a, _ in author_counter.most_common(5)]

    return {
        "test_case_ideas": test_case_ideas[:8],
        "suggested_reviewers": suggested_reviewers,
        "top_tags": top_tags,
        "top_keywords": top_keywords,
    }


class AutoTagRequest(BaseModel):
    title: str
    description: str


@app.post("/api/bugs/autotag")
def auto_tag_bug(payload: AutoTagRequest):
    text = (payload.title + " " + payload.description).lower()
    tag_map = {
        "ui": ["button", "layout", "css", "ux", "modal", "render"],
        "backend": ["api", "server", "db", "sql", "endpoint"],
        "performance": ["slow", "lag", "timeout", "latency", "memory"],
        "security": ["xss", "csrf", "auth", "inject", "token"],
        "database": ["query", "index", "mongo", "postgres", "deadlock"],
    }
    tags: List[str] = []
    for tag, kws in tag_map.items():
        if any(k in text for k in kws):
            tags.append(tag)
    if not tags:
        tags = ["general"]
    return {"tags": tags}


# -----------------------------
# Integrations config (placeholders)
# -----------------------------
class IntegrationPayload(BaseModel):
    provider: str
    settings: Dict[str, Any] = Field(default_factory=dict)


@app.post("/api/integrations/config")
def upsert_integration(payload: IntegrationPayload):
    if not payload.provider:
        raise HTTPException(status_code=400, detail="provider required")
    now = datetime.now(timezone.utc)
    db.integrationconfig.update_one(
        {"provider": payload.provider},
        {"$set": {"provider": payload.provider, "settings": payload.settings, "updated_at": now}, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    doc = db.integrationconfig.find_one({"provider": payload.provider})
    return to_obj(doc)


@app.get("/api/integrations/config")
def list_integrations():
    docs = db.integrationconfig.find().limit(50)
    return [to_obj(d) for d in docs]


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
