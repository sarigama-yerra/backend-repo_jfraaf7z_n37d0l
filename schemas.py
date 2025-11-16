"""
Database Schemas for BugSage

Each Pydantic model represents a MongoDB collection. The collection name is the lowercase of the class name.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr

# -----------------------------
# Core Entities
# -----------------------------

class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    role: str = Field("developer", description="Role: admin, developer, qa, manager")
    avatar_url: Optional[str] = Field(None, description="Profile image URL")
    is_active: bool = Field(True, description="Whether user is active")

class Project(BaseModel):
    name: str = Field(..., description="Project display name")
    key: str = Field(..., description="Short unique key, e.g., BUGSAGE")
    description: Optional[str] = Field(None)
    repo_url: Optional[str] = Field(None, description="VCS repository URL")
    members: List[Dict[str, Any]] = Field(default_factory=list, description="List of {user_id, role}")

class Attachment(BaseModel):
    name: str
    url: str
    type: str = Field("file", description="file, image, log")

class StatusChange(BaseModel):
    from_status: Optional[str] = None
    to_status: str
    at: datetime = Field(default_factory=datetime.utcnow)
    by: Optional[str] = None
    note: Optional[str] = None

class Comment(BaseModel):
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Bug(BaseModel):
    title: str
    description: str
    project_id: Optional[str] = None
    reporter_id: Optional[str] = None
    assignee_id: Optional[str] = None
    priority: str = Field("medium", description="low, medium, high, urgent")
    severity: str = Field("minor", description="minor, major, critical, blocker")
    status: str = Field("triage", description="triage, fix, verify, closed, reopened")
    module_path: Optional[str] = Field(None, description="Path or name of the affected module")
    tags: List[str] = Field(default_factory=list)
    steps_to_reproduce: Optional[str] = None
    environment: Optional[str] = None
    attachments: List[Attachment] = Field(default_factory=list)
    logs: Optional[str] = None
    comments: List[Comment] = Field(default_factory=list)
    history: List[StatusChange] = Field(default_factory=list)
    reopened_count: int = 0
    resolved_at: Optional[datetime] = None
    due_date: Optional[datetime] = None

class Commit(BaseModel):
    project_id: Optional[str] = None
    module_path: Optional[str] = None
    author: Optional[str] = None
    message: str
    additions: int = 0
    deletions: int = 0
    files: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Optional: lightweight configuration entities
class IntegrationConfig(BaseModel):
    provider: str
    settings: Dict[str, Any] = Field(default_factory=dict)

# The Flames database viewer will read these via GET /schema
ALL_MODELS = {
    "user": User,
    "project": Project,
    "bug": Bug,
    "commit": Commit,
    "integrationconfig": IntegrationConfig,
}
