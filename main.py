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
from typing import Dict, List, Optional, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Import generation utilities
from data_generator import (
    generate_model_info,
    generate_business_metrics,
    generate_drift_metrics,
    calculate_metrics_from_paths,
)

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
    drift: Optional[Dict[str, Any]] = None  # optional drift metrics
    restricted: Optional[bool] = None  # whether model was registered via client module
    connector: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # additional registration metadata
    business_metrics: Optional[Dict[str, Any]] = None  # metrics specific to this model

# In-memory list of models registered by users.  If empty a default set is returned.
registered_models: List[ModelInfo] = []

@app.get("/models", response_model=List[ModelInfo])
def get_models(_: str = Depends(get_current_username)) -> List[ModelInfo]:
    """Return a list of models and associated statistics.

    If the user has registered models via the model metadata workflow these are
    returned.  Otherwise a default set of banking‑related models is generated.
    """
    if registered_models:
        return registered_models
    # Otherwise generate default models
    models_data = generate_model_info(3)
    return [
        ModelInfo(
            id=m["id"],
            name=m["name"],
            labels=m["labels"],
            stats=ModelStatistics(**m["stats"]),
        )
        for m in models_data
    ]

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
    metrics = generate_business_metrics()
    return BusinessMetricsResponse(
        cost=MetricSeries(**metrics["cost"]),
        resource_utilization=MetricSeries(**metrics["resource_utilization"]),
        performance=MetricSeries(**metrics["performance"]),
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
    drift_data = generate_drift_metrics(5)
    return DriftResponse(
        drift_score=drift_data["drift_score"],
        drift_detected=drift_data["drift_detected"],
        details=drift_data["details"],
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

# -------------------------------------------------------------------
# Model-specific Business Metrics and Drift
# -------------------------------------------------------------------

@app.get("/business-metrics/{model_id}", response_model=BusinessMetricsResponse)
def business_metrics_for_model(model_id: str, _: str = Depends(get_current_username)) -> BusinessMetricsResponse:
    """Return business metrics for a specific model."""
    # Find the model in the registered list
    for m in registered_models:
        if m.id == model_id:
            # Use stored metrics if available
            if m.business_metrics:
                bm = m.business_metrics
                return BusinessMetricsResponse(
                    cost=MetricSeries(**bm["cost"]),
                    resource_utilization=MetricSeries(**bm["resource_utilization"]),
                    performance=MetricSeries(**bm["performance"]),
                )
    # If not found or no metrics, return generated metrics
    bm = generate_business_metrics()
    return BusinessMetricsResponse(
        cost=MetricSeries(**bm["cost"]),
        resource_utilization=MetricSeries(**bm["resource_utilization"]),
        performance=MetricSeries(**bm["performance"]),
    )


@app.get("/drift/{model_id}", response_model=DriftResponse)
def drift_for_model(model_id: str, _: str = Depends(get_current_username)) -> DriftResponse:
    """Return drift metrics for a specific model."""
    for m in registered_models:
        if m.id == model_id and m.drift:
            d = m.drift
            return DriftResponse(
                drift_score=d["drift_score"],
                drift_detected=d["drift_detected"],
                details=d["details"],
            )
    # Otherwise generate dummy
    d = generate_drift_metrics(5)
    return DriftResponse(
        drift_score=d["drift_score"],
        drift_detected=d["drift_detected"],
        details=d["details"],
    )

# -------------------------------------------------------------------
# Model Metadata Registration
# -------------------------------------------------------------------

class ModelRegisterRequest(BaseModel):
    restricted: bool = Field(..., description="Whether the model data is restricted and must be processed client side")
    model_name: str = Field(..., description="Name of the model to register")
    connector: str = Field(..., description="Name of the connector where the model/data resides")
    connector_details: Optional[Dict[str, str]] = Field(
        default=None, description="Additional parameters required for the connector (e.g. bucket, path)"
    )
    training_data_path: Optional[str] = Field(default=None, description="Path to the training data (if unrestricted)")
    production_data_path: Optional[str] = Field(default=None, description="Path to the production data (if unrestricted)")

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Additional model metadata (version, description, author, training parameters, deployment info, compliance, etc.)"
        ),
    )

class ModelRegisterResponse(BaseModel):
    message: str
    instructions: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class ClientMetricsRequest(BaseModel):
    model_name: str
    metrics: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

@app.post("/model-metadata/register", response_model=ModelRegisterResponse)
def register_model(data: ModelRegisterRequest, username: str = Depends(get_current_username)) -> ModelRegisterResponse:
    """
    Register a model and optionally compute metrics.

    If the model is restricted (data cannot leave the client environment) the
    endpoint returns instructions for running the client module locally.  If
    unrestricted the server will calculate metrics on the provided data
    immediately and store the result.
    """
    if data.restricted:
        instructions = (
            "Data is marked as restricted. Please download the client module from /client-module "
            "and run it on the machine where the data resides using the following command:\n"
            "python client_module.py --server-url <server_url> --token <access_token> "
            f"--model-name '{data.model_name}' --training-data <training_path> --production-data <production_path>"
        )
        # For restricted models we don't compute metrics here; metrics will be uploaded later.
        return ModelRegisterResponse(message="Model registration initiated in restricted mode", instructions=instructions)
    # Unrestricted: compute metrics on the server using provided paths
    if not data.training_data_path or not data.production_data_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="training_data_path and production_data_path are required for unrestricted models")
    metrics = calculate_metrics_from_paths(data.training_data_path, data.production_data_path)
    new_model = ModelInfo(
        id=str(uuid.uuid4()),
        name=data.model_name,
        labels=["no", "yes"],
        stats=ModelStatistics(**metrics["stats"]),
        drift=metrics["drift"],
        restricted=False,
        connector=data.connector,
        metadata=data.metadata,
        business_metrics=generate_business_metrics(),
    )
    registered_models.append(new_model)
    return ModelRegisterResponse(message="Model registered successfully", metrics=metrics)

@app.post("/model-metadata/submit", response_model=ModelRegisterResponse)
def submit_client_metrics(data: ClientMetricsRequest, username: str = Depends(get_current_username)) -> ModelRegisterResponse:
    """
    Submit metrics computed on a client machine.

    This endpoint allows a client running in a restricted environment to upload
    the aggregated metrics for a model.  The server stores the model and
    acknowledges receipt.
    """
    # Extract metrics from payload
    metrics = data.metrics
    # Build ModelInfo
    new_model = ModelInfo(
        id=str(uuid.uuid4()),
        name=data.model_name,
        labels=["no", "yes"],
        stats=ModelStatistics(**metrics["stats"]),
        drift=metrics.get("drift"),
        restricted=True,
        connector=None,
        metadata=data.metadata,
        business_metrics=generate_business_metrics(),
    )
    registered_models.append(new_model)
    return ModelRegisterResponse(message="Metrics received and model registered")

@app.get("/client-module")
def download_client_module(_: str = Depends(get_current_username)):
    """Provide the client module for download."""
    # The client module is located in the same directory as this script
    module_path = __file__.replace("main.py", "client_module.py")
    return FileResponse(module_path, media_type="text/x-python", filename="client_module.py")