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

# Import the LLM helper function.  This function calls an external API (e.g.
# OpenAI) to generate responses.  If the API key is not configured or the
# request fails the helper will return a string prefixed with "LLM analysis
# unavailable".  See llm_response.py for details.
try:
    from llm_response import get_llm_response  # type: ignore
except Exception:
    # Provide a fallback in case the module cannot be imported
    def get_llm_response(system_prompt: str, user_prompt: str, max_tokens: int = 600, temperature: float = 0) -> str:
        return "LLM analysis unavailable: llm_response module missing"

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
# Each entry contains password, persona, profile, LOB and sub‑LOB information.
#
# The "profile" field determines a user's access level.  Two profiles are
# recognised:
#   * "admin" – full access to connectors, monitoring and access management
#   * "standard" – limited access scoped to the models the user owns
#
users_info: Dict[str, Dict[str, any]] = {
    # username: details
    #
    # For demonstration the dashboard includes two accounts per persona: one
    # with a regular ("user") profile and one with administrative privileges.
    # Personas include Data Scientist, Product Owner, Site Reliability Engineer
    # (SRE), Machine Learning Engineer and Application Owner.  The profile
    # determines which navigation items appear in the UI; admins can access
    # additional pages such as Connectors, Monitoring and Access Management.
    "datascientist_user": {
        "password": "ds_user123",
        "persona": "Data Scientist",
        "profile": "user",
        "lob": "Analytics",
        "sub_lob": "Fraud",
    },
    "datascientist_admin": {
        "password": "ds_admin123",
        "persona": "Data Scientist",
        "profile": "admin",
        "lob": "Analytics",
        "sub_lob": "Fraud",
    },
    "productowner_user": {
        "password": "po_user123",
        "persona": "Product Owner",
        "profile": "user",
        "lob": "Product",
        "sub_lob": "Payments",
    },
    "productowner_admin": {
        "password": "po_admin123",
        "persona": "Product Owner",
        "profile": "admin",
        "lob": "Product",
        "sub_lob": "Payments",
    },
    "sre_user": {
        "password": "sre_user123",
        "persona": "SRE",
        "profile": "user",
        "lob": "Operations",
        "sub_lob": "Infrastructure",
    },
    "sre_admin": {
        "password": "sre_admin123",
        "persona": "SRE",
        "profile": "admin",
        "lob": "Operations",
        "sub_lob": "Infrastructure",
    },
    "mlengineer_user": {
        "password": "ml_user123",
        "persona": "ML Engineer",
        "profile": "user",
        "lob": "Engineering",
        "sub_lob": "Models",
    },
    "mlengineer_admin": {
        "password": "ml_admin123",
        "persona": "ML Engineer",
        "profile": "admin",
        "lob": "Engineering",
        "sub_lob": "Models",
    },
    "appowner_user": {
        "password": "app_user123",
        "persona": "Application Owner",
        "profile": "user",
        "lob": "Applications",
        "sub_lob": "Retail",
    },
    "appowner_admin": {
        "password": "app_admin123",
        "persona": "Application Owner",
        "profile": "admin",
        "lob": "Applications",
        "sub_lob": "Retail",
    },
}

# In-memory token store mapping tokens to usernames, expiration and additional claims.
tokens_db: Dict[str, Dict[str, any]] = {}

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str
    expires_in: int = Field(description="Seconds until the token expires")
    persona: str
    profile: str
    username: str

class UserInfoResponse(BaseModel):
    username: str
    persona: str
    profile: str
    lob: str
    sub_lob: str


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
    if not record or record["expires_at"] < datetime.now():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return record["username"]

@app.get("/me", response_model=UserInfoResponse)
def get_user_info(username: str = Depends(get_current_username)) -> UserInfoResponse:
    """Return the current user's persona, profile and LOB information."""
    info = users_info.get(username)
    if not info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserInfoResponse(
        username=username,
        persona=info["persona"],
        profile=info["profile"],
        lob=info["lob"],
        sub_lob=info["sub_lob"],
    )

@app.post("/login", response_model=TokenResponse)
def login(data: LoginRequest) -> TokenResponse:
    """
    Authenticate a user and return an access token.

    The supplied credentials are validated against the in-memory `users_info`
    dictionary.  Upon successful authentication a unique token is created and
    stored along with the user's persona, profile and other metadata.  The
    token expires after one hour.  Clients should include the token in the
    `Authorization: Bearer <token>` header when calling other endpoints.
    """
    user_record = users_info.get(data.username)
    if user_record is None or user_record["password"] != data.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    # Create token valid for 1 hour
    token = uuid.uuid4().hex
    expires_at = datetime.now() + timedelta(hours=1)
    # Store additional claims on the token for convenience
    tokens_db[token] = {
        "username": data.username,
        "persona": user_record["persona"],
        "profile": user_record["profile"],
        "lob": user_record["lob"],
        "sub_lob": user_record["sub_lob"],
        "expires_at": expires_at,
    }
    # Include persona, profile and username in the token response so that
    # the frontend can immediately determine role‑based access without
    # making a separate `/me` call.  These fields are also stored on
    # the server as part of the token record.
    return TokenResponse(
        token=token,
        expires_in=int((expires_at - datetime.now()).total_seconds()),
        persona=user_record["persona"],
        profile=user_record["profile"],
        username=data.username,
    )

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
    current = users_info.get(username)
    if not current or current["password"] != data.old_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Old password is incorrect")
    users_info[username]["password"] = data.new_password
    return SimpleMessage(message="Password updated successfully")

@app.get("/users", response_model=List[str])
def list_users(_: str = Depends(get_current_username)) -> List[str]:
    """Return a list of all usernames."""
    return list(users_info.keys())

@app.post("/users", response_model=SimpleMessage)
def create_user(data: CreateUserRequest, username: str = Depends(get_current_username)) -> SimpleMessage:
    """Add a new user to the in-memory user store.  Only admin users may create accounts."""
    # verify caller is admin
    token = None
    # Determine profile via token
    for tok, rec in tokens_db.items():
        if rec["username"] == username:
            token = rec
            break
    if token is None or token.get("profile") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins may create users")
    if data.username in users_info:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    # Default persona and profile for new users: Data Scientist / standard
    users_info[data.username] = {
        "password": data.password,
        "persona": "Data Scientist",
        # New users receive a 'user' profile by default.  The label 'standard'
        # was used in earlier versions but has been replaced with 'user'.
        "profile": "user",
        "lob": "Unknown",
        "sub_lob": "Unknown",
    }
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
    owner: Optional[str] = None  # username of the model registrar
    platform: Optional[str] = None  # platform such as Databricks, AWS, Mercury, Neptune
    connector: Optional[str] = None  # connector type (e.g. S3, Volumes)
    connector_details: Optional[Dict[str, Any]] = None  # JSON details for connector configuration
    metadata: Optional[Dict[str, Any]] = None  # additional registration metadata
    business_metrics: Optional[Dict[str, Any]] = None  # metrics specific to this model
    training_pipeline: Optional[Dict[str, Any]] = None  # dummy training pipeline info
    production_pipeline: Optional[Dict[str, Any]] = None  # dummy production pipeline info

# In-memory list of models registered by users.  If empty a default set is returned.
registered_models: List[ModelInfo] = []

# -------------------------------------------------------------------
# Demo Data Initialisation
# -------------------------------------------------------------------

# At import time populate the user store with one model per user.  This
# demonstrates the 360‑degree model view on the home page and exercises
# the model‑specific metrics and drift endpoints.  Each model is given
# sensible banking‑related names and platforms/connectors.

def _initialise_demo_models() -> None:
    """Create demo models for each user if no models are registered."""
    if registered_models:
        return  # Avoid re‑initialising if models already exist
    # Generate a pool of model definitions; rotate through names for variety
    base_models = generate_model_info(len(users_info))
    # Predefine platform and connector mappings per persona/user
    platform_connector_map = {
        "datascientist_user": ("Databricks", "Unity Catalog", {"catalog": "fraud_models"}),
        "datascientist_admin": ("Databricks", "Volumes", {"volume": "analytics_volume"}),
        "productowner_user": ("AWS", "S3", {"bucket": "churn-data"}),
        "productowner_admin": ("AWS", "RDS Postgres", {"database": "product_db"}),
        "sre_user": ("AWS", "RDS Postgres", {"database": "operations_db"}),
        "sre_admin": ("AWS", "S3", {"bucket": "infra-logs"}),
        "mlengineer_user": ("Mercury", "Storage", {"cluster": "ml-storage"}),
        "mlengineer_admin": ("Mercury", "SQL", {"database": "mlops"}),
        "appowner_user": ("Neptune", "Gremlin", {"endpoint": "retail-graph"}),
        "appowner_admin": ("Neptune", "OpenCypher", {"endpoint": "app-graph"}),
    }
    # Iterate through users and assign a model
    for idx, (username, info) in enumerate(users_info.items()):
        model_def = base_models[idx % len(base_models)]
        # Create drift and business metrics
        drift = generate_drift_metrics(len(model_def["stats"]["feature_names"]))
        business = generate_business_metrics()
        # Determine platform and connector based on username mapping; default to None
        platform, connector, connector_details = platform_connector_map.get(
            username, (None, None, None)
        )
        # Compose metadata dictionary; minimal metadata by default
        metadata = {
            "platform": platform,
            "connector": connector,
            "version": "1.0",
            "description": f"Auto‑generated model for {info['persona']} user",
            "author": username,
            "training_dataset": "2024‑Q1",
            "training_parameters": "Default",
            "deployment_environment": "staging",
            "approval_status": "Approved",
            "last_modified": datetime.now().isoformat(),
        }
        # Build training and production pipeline details
        now_iso = datetime.now().isoformat()
        training_pipeline = {
            "data_source": f"/data/{username}/train.csv",
            "last_run": now_iso,
            "status": "success",
        }
        production_pipeline = {
            "data_source": f"/data/{username}/prod.csv",
            "last_run": now_iso,
            "status": "success",
        }
        registered_models.append(
            ModelInfo(
                id=model_def["id"],
                name=model_def["name"],
                labels=model_def["labels"],
                stats=ModelStatistics(**model_def["stats"]),
                drift=drift,
                restricted=False,
                owner=username,
                platform=platform,
                connector=connector,
                connector_details=connector_details,
                metadata=metadata,
                business_metrics=business,
                training_pipeline=training_pipeline,
                production_pipeline=production_pipeline,
            )
        )


# Initialise demonstration models once on import
_initialise_demo_models()

@app.get("/models", response_model=List[ModelInfo])
def get_models(username: str = Depends(get_current_username)) -> List[ModelInfo]:
    """Return a list of models visible to the current user.

    Admin users can see all registered models, whereas non‑admin users see
    only the models they have registered.  If no models are registered yet
    a default set of banking-related models is generated for demonstration.
    """
    # Determine caller's profile
    token_info = next((rec for rec in tokens_db.values() if rec["username"] == username), None)
    # If no token information is available default to 'user'
    profile = token_info.get("profile") if token_info else "user"
    if registered_models:
        if profile == "admin":
            return registered_models
        # filter by owner
        return [m for m in registered_models if m.owner == username]
    # Otherwise generate default models for demonstration
    models_data = generate_model_info(3)
    return [
        ModelInfo(
            id=m["id"],
            name=m["name"],
            labels=m["labels"],
            stats=ModelStatistics(**m["stats"]),
            owner=None,
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
    now_str = datetime.now().isoformat()
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
    base_time = datetime.now()
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
# LLM Integration
# -------------------------------------------------------------------

class LLMQuery(BaseModel):
    query: str
    model_id: Optional[str] = None

class LLMResponse(BaseModel):
    response: str

@app.post("/llm/drift", response_model=LLMResponse)
def llm_drift(data: LLMQuery, username: str = Depends(get_current_username)) -> LLMResponse:
    """
    Query the LLM for drift analysis.  The query should describe the model
    or drift scenario to be analysed.  Optionally a model_id can be provided
    so that the server can embed metadata or context; for now this has no
    effect on the dummy implementation but could be used to craft a more
    specific prompt.
    """
    system_prompt = (
        "You are a machine learning expert specialising in drift detection for banking models. "
        "Answer the user's question concisely and clearly."
    )
    user_prompt = data.query
    # Call the helper; handle failure gracefully
    try:
        answer = get_llm_response(system_prompt, user_prompt, max_tokens=400, temperature=0)
    except Exception:
        answer = "LLM analysis unavailable due to an internal error"
    return LLMResponse(response=answer)

class FAQQuery(BaseModel):
    question: str

@app.post("/llm/faq", response_model=LLMResponse)
def llm_faq(data: FAQQuery, username: str = Depends(get_current_username)) -> LLMResponse:
    """
    Chat endpoint for Vision AI frequently asked questions.  The backend stores
    a simple dictionary of FAQs.  If the question is not recognised a generic
    response is generated by the LLM helper.
    """
    # Simple FAQ database; keys are lowercased questions
    faqs = {
        "what is vision ai": "Vision AI is an open, platform‑independent monitoring dashboard for machine learning models.",
        "how do i register a model": "Navigate to the Model Metadata page, fill in the model details and choose whether the data is restricted.",
        "what connectors are supported": "Supported connectors include Databricks (Volumes, Unity Catalog, DBFS), AWS (S3, RDS Postgres), Mercury and Neptune.",
        "how does drift detection work": "Drift detection compares the distribution of new data with baseline training data using statistical tests such as PSI or KS tests.",
    }
    q = data.question.strip().lower()
    if q in faqs:
        return LLMResponse(response=faqs[q])
    # Fall back to LLM
    system_prompt = (
        "You are a helpful assistant answering frequently asked questions about the Vision AI dashboard. "
        "If you don't know the answer, respond politely that the information is not available."
    )
    try:
        answer = get_llm_response(system_prompt, data.question, max_tokens=200, temperature=0)
    except Exception:
        answer = "LLM analysis unavailable due to an internal error"
    return LLMResponse(response=answer)

# -------------------------------------------------------------------
# Access Management
# -------------------------------------------------------------------

class AccessRecord(BaseModel):
    username: str
    persona: str
    profile: str
    lob: str
    sub_lob: str

@app.get("/access-management", response_model=List[AccessRecord])
def access_management(username: str = Depends(get_current_username)) -> List[AccessRecord]:
    """
    Return a list of users and their access information.  Only admins may call this endpoint.
    """
    token_info = next((rec for rec in tokens_db.values() if rec["username"] == username), None)
    if not token_info or token_info.get("profile") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins may view access information")
    records: List[AccessRecord] = []
    for user, info in users_info.items():
        records.append(
            AccessRecord(
                username=user,
                persona=info["persona"],
                profile=info["profile"],
                lob=info.get("lob", ""),
                sub_lob=info.get("sub_lob", ""),
            )
        )
    return records

class AccessCreateRequest(BaseModel):
    username: str
    password: str
    persona: str
    profile: str
    lob: str
    sub_lob: str

@app.post("/access-management", response_model=SimpleMessage)
def add_access(data: AccessCreateRequest, username: str = Depends(get_current_username)) -> SimpleMessage:
    """
    Create a new user with the specified persona, profile and LOB information.  Only admins may call this.
    """
    token_info = next((rec for rec in tokens_db.values() if rec["username"] == username), None)
    if not token_info or token_info.get("profile") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins may modify access")
    if data.username in users_info:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    users_info[data.username] = {
        "password": data.password,
        "persona": data.persona,
        "profile": data.profile,
        "lob": data.lob,
        "sub_lob": data.sub_lob,
    }
    return SimpleMessage(message=f"User {data.username} created successfully")

@app.delete("/access-management/{target_username}", response_model=SimpleMessage)
def delete_access(target_username: str, username: str = Depends(get_current_username)) -> SimpleMessage:
    """
    Delete an existing user.  Only admins may call this endpoint.  Prevent deleting oneself.
    """
    token_info = next((rec for rec in tokens_db.values() if rec["username"] == username), None)
    if not token_info or token_info.get("profile") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins may modify access")
    if target_username == username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot delete your own account")
    if target_username not in users_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    del users_info[target_username]
    # Remove any tokens belonging to the deleted user
    to_delete = [tok for tok, rec in tokens_db.items() if rec["username"] == target_username]
    for tok in to_delete:
        del tokens_db[tok]
    return SimpleMessage(message=f"User {target_username} removed successfully")

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


@app.get("/models/{model_id}", response_model=ModelInfo)
def get_model_details(model_id: str, username: str = Depends(get_current_username)) -> ModelInfo:
    """
    Return the full model information for a specific model.  Non‑admin users can
    only access details for models they own.
    """
    token_info = next((rec for rec in tokens_db.values() if rec["username"] == username), None)
    profile = token_info.get("profile") if token_info else "user"
    for m in registered_models:
        if m.id == model_id:
            # Check permissions
            if profile != "admin" and m.owner != username:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this model")
            return m
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")


@app.get("/drift/{model_id}", response_model=DriftResponse)
def drift_for_model(
    model_id: str,
    drift_type: Optional[str] = None,
    time_range: Optional[str] = None,
    scan_type: Optional[str] = None,
    _: str = Depends(get_current_username),
) -> DriftResponse:
    """
    Return drift metrics for a specific model.

    Query parameters allow the client to specify the drift type (data, feature,
    concept or all), the time range (yearly, quarterly, monthly, daily) and the
    scan type (quick scan, smart scan, deep scan).  The current implementation
    ignores these parameters and returns randomly generated metrics.  A real
    implementation would alter the calculation based on these choices.
    """
    for m in registered_models:
        if m.id == model_id and m.drift:
            d = m.drift
            return DriftResponse(
                drift_score=d["drift_score"],
                drift_detected=d["drift_detected"],
                details=d["details"],
            )
    # Otherwise generate dummy.  Number of features could be tuned based on scan type.
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
    platform: Optional[str] = Field(None, description="Platform where the model resides (Databricks, AWS, Mercury, Neptune)")
    connector: Optional[str] = Field(None, description="Name of the connector used for accessing the model/data")
    connector_details: Optional[Dict[str, Any]] = Field(
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
    # Consolidate the metadata provided by the client.  Always ensure that
    # platform and connector are stored within the metadata dictionary so that
    # downstream users can view this information in a single place.  If the
    # metadata field is omitted it is initialised to an empty dict.
    metadata_fields: Dict[str, any] = dict(data.metadata or {})
    if data.platform:
        metadata_fields.setdefault("platform", data.platform)
    if data.connector:
        metadata_fields.setdefault("connector", data.connector)

    # If the data is marked restricted the server does not compute metrics.
    # Instead return instructions for running the client module.  We still
    # record the model skeleton in the registry so that the user can see it in
    # their dashboard.
    if data.restricted:
        instructions = (
            "Data is marked as restricted. Please download the client module from /client-module "
            "and run it on the machine where the data resides using the following command:\n"
            "python client_module.py --server-url <server_url> --token <access_token> "
            f"--model-name '{data.model_name}' --training-data <training_path> --production-data <production_path>"
        )
        # Register a placeholder model record for restricted mode
        placeholder_model = ModelInfo(
            id=str(uuid.uuid4()),
            name=data.model_name,
            labels=["no", "yes"],
            stats=ModelStatistics(**generate_model_info(1)[0]["stats"]),
            drift=None,
            restricted=True,
            owner=username,
            platform=data.platform,
            connector=data.connector,
            connector_details=data.connector_details,
            # Persist consolidated metadata including platform/connector
            metadata=metadata_fields,
            business_metrics=generate_business_metrics(),
            training_pipeline=None,
            production_pipeline=None,
        )
        registered_models.append(placeholder_model)
        return ModelRegisterResponse(message="Model registration initiated in restricted mode", instructions=instructions)
    # Unrestricted: compute metrics on the server using provided paths.  Both paths are required
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
        owner=username,
        platform=data.platform,
        connector=data.connector,
        connector_details=data.connector_details,
        # Persist consolidated metadata including platform/connector
        metadata=metadata_fields,
        business_metrics=generate_business_metrics(),
        # Generate dummy pipeline info; in a real system this would come from your ML pipeline
        training_pipeline={
            "data_source": data.training_data_path,
            "last_run": datetime.now().isoformat(),
            "status": "success",
        },
        production_pipeline={
            "data_source": data.production_data_path,
            "last_run": datetime.now().isoformat(),
            "status": "success",
        },
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
    # Build ModelInfo with the owner information
    new_model = ModelInfo(
        id=str(uuid.uuid4()),
        name=data.model_name,
        labels=["no", "yes"],
        stats=ModelStatistics(**metrics["stats"]),
        drift=metrics.get("drift"),
        restricted=True,
        owner=username,
        platform=None,
        connector=None,
        connector_details=None,
        metadata=data.metadata,
        business_metrics=generate_business_metrics(),
        training_pipeline=None,
        production_pipeline=None,
    )
    registered_models.append(new_model)
    return ModelRegisterResponse(message="Metrics received and model registered")

@app.get("/client-module")
def download_client_module(_: str = Depends(get_current_username)):
    """Provide the client module for download."""
    # The client module is located in the same directory as this script
    module_path = __file__.replace("main.py", "client_module.py")
    return FileResponse(module_path, media_type="text/x-python", filename="client_module.py")