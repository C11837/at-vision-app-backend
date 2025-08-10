"""
FastAPI application serving the Vision Dashboard backend.

The service exposes endpoints for user authentication, model metadata,
business metrics, drift detection and more.  All data returned from the
endpoints is randomly generated at runtime to simulate the shape of
responses from a real system.  Authentication is implemented using
a simple token mechanism returned from the `/login` endpoint; production
systems should replace this with SSO or another secure method.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Vision Dashboard API", version="0.1.0")

# Allow frontend running on different ports to access the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------------------------------------------------
# Authentication
# -------------------------------------------------------------------

# In-memory user store; do not use this in production.
fake_users_db: Dict[str, str] = {
    "user": "password"
}

# In-memory token store mapping tokens to usernames and expiration times.
tokens_db: Dict[str, Dict[str, datetime]] = {}

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str
    expires_in: int = Field(description="Seconds until the token expires")

def get_current_username(authorization: Optional[str] = Header(None)) -> str:
    """Extract and validate the token supplied in the Authorization header."""
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    try:
        scheme, token = authorization.split()
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization scheme must be Bearer")
    record = tokens_db.get(token)
    if not record or record["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return record["username"]

@app.post("/login", response_model=TokenResponse)
def login(data: LoginRequest) -> TokenResponse:
    """
    Authenticate a user and return a token.

    For demonstration a single default user exists (`user` / `password`).  A real
    implementation should verify credentials against an identity provider or
    SSO system.
    """
    stored_pw = fake_users_db.get(data.username)
    if stored_pw is None or stored_pw != data.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    # Create token valid for 1 hour
    token = uuid.uuid4().hex
    expires_at = datetime.utcnow() + timedelta(hours=1)
    tokens_db[token] = {"username": data.username, "expires_at": expires_at}
    return TokenResponse(token=token, expires_in=int((expires_at - datetime.utcnow()).total_seconds()))

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class CreateUserRequest(BaseModel):
    username: str
    password: str

class SimpleMessage(BaseModel):
    message: str

@app.post("/change-password", response_model=SimpleMessage)
def change_password(data: ChangePasswordRequest, username: str = Depends(get_current_username)) -> SimpleMessage:
    """Change the current user's password."""
    current_pw = fake_users_db.get(username)
    if current_pw != data.old_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Old password is incorrect")
    fake_users_db[username] = data.new_password
    return SimpleMessage(message="Password updated successfully")

@app.get("/users", response_model=List[str])
def list_users(_: str = Depends(get_current_username)) -> List[str]:
    """Return a list of all usernames."""
    return list(fake_users_db.keys())

@app.post("/users", response_model=SimpleMessage)
def create_user(data: CreateUserRequest, _: str = Depends(get_current_username)) -> SimpleMessage:
    """Add a new user to the in-memory user store."""
    if data.username in fake_users_db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    fake_users_db[data.username] = data.password
    return SimpleMessage(message=f"User {data.username} created successfully")

# -------------------------------------------------------------------
# Models Endpoint
# -------------------------------------------------------------------

class ModelStatistics(BaseModel):
    feature_names: List[str]
    coefficients: List[float]
    feature_means: List[float]
    coefficient_variance: List[float]

class ModelInfo(BaseModel):
    id: str
    name: str
    labels: List[str]
    stats: ModelStatistics

@app.get("/models", response_model=List[ModelInfo])
def get_models(_: str = Depends(get_current_username)) -> List[ModelInfo]:
    """Return a list of models and associated statistics."""
    models: List[ModelInfo] = []
    # Generate three dummy models
    for i in range(3):
        num_features = random.randint(3, 6)
        feature_names = [f"feature_{j+1}" for j in range(num_features)]
        coefficients = np.random.randn(num_features).round(3).tolist()
        feature_means = np.random.rand(num_features).round(3).tolist()
        # For simplicity coefficient variance is positive values derived from random numbers
        coefficient_variance = (np.random.rand(num_features) * 0.5).round(3).tolist()
        model = ModelInfo(
            id=str(uuid.uuid4()),
            name=f"Model {i+1}",
            labels=["class_0", "class_1"],
            stats=ModelStatistics(
                feature_names=feature_names,
                coefficients=coefficients,
                feature_means=feature_means,
                coefficient_variance=coefficient_variance,
            ),
        )
        models.append(model)
    return models

# -------------------------------------------------------------------
# Business Metrics Endpoint
# -------------------------------------------------------------------

class MetricSeries(BaseModel):
    name: str
    timestamps: List[str]
    values: List[float]

class BusinessMetricsResponse(BaseModel):
    cost: MetricSeries
    resource_utilization: MetricSeries
    performance: MetricSeries

@app.get("/business-metrics", response_model=BusinessMetricsResponse)
def business_metrics(_: str = Depends(get_current_username)) -> BusinessMetricsResponse:
    """Return business metrics such as cost, resource utilisation and model performance."""
    # Create a common time range (last 12 months)
    now = datetime.utcnow()
    months = [now - timedelta(days=30 * i) for i in reversed(range(12))]
    timestamps = [m.strftime("%Y-%m-%d") for m in months]

    def random_series(name: str, baseline: float) -> MetricSeries:
        values = (baseline + 0.1 * np.random.randn(12)).round(3).tolist()
        return MetricSeries(name=name, timestamps=timestamps, values=values)

    return BusinessMetricsResponse(
        cost=random_series("Cost ($k)", 100.0),
        resource_utilization=random_series("Resource Utilization (%)", 70.0),
        performance=random_series("Model Accuracy (%)", 85.0),
    )

# -------------------------------------------------------------------
# Drift Detection Endpoint
# -------------------------------------------------------------------

class DriftResponse(BaseModel):
    drift_score: float
    drift_detected: bool
    details: Dict[str, float]

@app.get("/drift", response_model=DriftResponse)
def drift(_: str = Depends(get_current_username)) -> DriftResponse:
    """Return dummy drift statistics."""
    score = random.random()
    details = {f"feature_{i+1}": round(random.random(), 3) for i in range(5)}
    return DriftResponse(
        drift_score=round(score, 3),
        drift_detected=score > 0.7,
        details=details,
    )

# -------------------------------------------------------------------
# Connectors Endpoint
# -------------------------------------------------------------------

class Connector(BaseModel):
    name: str
    type: str
    status: str

@app.get("/connectors", response_model=List[Connector])
def connectors(_: str = Depends(get_current_username)) -> List[Connector]:
    """List available connectors and their status."""
    return [
        Connector(name="Databricks", type="cloud", status="connected"),
        Connector(name="AWS S3", type="cloud", status="pending"),
        Connector(name="Azure Blob", type="cloud", status="disconnected"),
        Connector(name="On-Prem SQL", type="onprem", status="connected"),
    ]

# -------------------------------------------------------------------
# Monitoring and Notifications
# -------------------------------------------------------------------

class MonitoringStatus(BaseModel):
    service: str
    status: str
    last_checked: str

@app.get("/monitoring", response_model=List[MonitoringStatus])
def monitoring(_: str = Depends(get_current_username)) -> List[MonitoringStatus]:
    """Return the health status of various services."""
    now_str = datetime.utcnow().isoformat()
    return [
        MonitoringStatus(service="Drift Detection Engine", status="healthy", last_checked=now_str),
        MonitoringStatus(service="Notification Service", status="healthy", last_checked=now_str),
        MonitoringStatus(service="Model Metadata Registration", status="degraded", last_checked=now_str),
    ]

class Notification(BaseModel):
    id: str
    title: str
    message: str
    timestamp: str

@app.get("/notifications", response_model=List[Notification])
def notifications(_: str = Depends(get_current_username)) -> List[Notification]:
    """Return a list of notifications."""
    base_time = datetime.utcnow()
    notifs: List[Notification] = []
    messages = [
        ("Model drift detected", "Drift score exceeded threshold for model 1"),
        ("New model registered", "Model 3 has been added to the registry"),
        ("Connector update", "AWS S3 connector is now pending approval"),
    ]
    for i, (title, msg) in enumerate(messages):
        notifs.append(
            Notification(
                id=str(uuid.uuid4()),
                title=title,
                message=msg,
                timestamp=(base_time - timedelta(minutes=10 * i)).isoformat(),
            )
        )
    return notifs