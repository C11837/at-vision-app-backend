# Vision Dashboard Backend (Python)

This directory contains a lightweight Python API used by the Vision Dashboard frontend.  The service is built on **FastAPI** so it is easy to run locally and extend.  It does **not** expose any sensitive information – all data returned from the endpoints is dummy data generated at runtime.  The endpoints and authentication model are deliberately simple to make the architecture easy to understand and modify.

## Features

- **Authentication** – A `/login` endpoint accepts a username and password.  On success it returns a short‑lived access token which must be supplied in the `Authorization` header on subsequent requests.  In a production environment this could be replaced with SSO or any other authentication mechanism.

  Ten sample users are provided out of the box, covering five personas.  For each persona there is both a **user** and an **admin** profile so you can explore the role‑based access controls implemented in the dashboard:

  | Username                 | Password       | Persona            | Profile | Line of Business | Sub‑LOB        | Notes                                                                                  |
  |--------------------------|---------------:|--------------------|---------|------------------|----------------|----------------------------------------------------------------------------------------|
  | `datascientist_user`     | ds_user123     | Data Scientist     | user    | Analytics        | Fraud          | Owns a fraud detection model.  Sees only model‑centric pages.                          |
  | `datascientist_admin`    | ds_admin123    | Data Scientist     | admin   | Analytics        | Fraud          | Sees all models plus admin pages like Monitoring, Connectors and Access Management.    |
  | `productowner_user`      | po_user123     | Product Owner      | user    | Product          | Payments       | Owns a customer churn model.  Limited to model features.                               |
  | `productowner_admin`     | po_admin123    | Product Owner      | admin   | Product          | Payments       | Can view all models and manage connectors and access.                                   |
  | `sre_user`               | sre_user123    | SRE                | user    | Operations       | Infrastructure | Owns an operations model.  Cannot access admin functionality.                          |
  | `sre_admin`              | sre_admin123   | SRE                | admin   | Operations       | Infrastructure | Full admin privileges; can manage monitoring and connectors.                            |
  | `mlengineer_user`        | ml_user123     | ML Engineer        | user    | Engineering      | Models         | Owns a machine learning model.                                                         |
  | `mlengineer_admin`       | ml_admin123    | ML Engineer        | admin   | Engineering      | Models         | Admin privileges for ML operations.                                                    |
  | `appowner_user`          | app_user123    | Application Owner  | user    | Applications     | Retail         | Owns a retail transaction model.                                                       |
  | `appowner_admin`         | app_admin123   | Application Owner  | admin   | Applications     | Retail         | Admin privileges across applications and connectors.                                    |

  The **profile** determines what pages appear in the left navigation bar.  Users with the `admin` profile have access to additional sections such as **Monitoring**, **Connectors** and **Access Management**.  Users with the `user` profile only see model‑centric features.

- **Registered Models** – The `/models` endpoint returns a list of registered models along with a 360° view of each model: labels, feature names, coefficient values, feature means, and coefficient variance.  These values are randomly generated in the example implementation but the shape of the response can be preserved when connecting to a real model registry.

- **Business Metrics** – The `/business-metrics` endpoint returns a set of aggregated metrics such as cost, resource utilisation and model performance.  For demonstration purposes these metrics include time‑series data suitable for charting on the frontend.

- **Drift Detection** – The `/drift` endpoint serves dummy drift statistics including a drift score and whether drift has been detected.  A real‑world implementation would run drift detection either on the server or client side depending on privacy requirements.

- **Connectors** – The `/connectors` endpoint advertises connections to external systems (for example Databricks, AWS, Azure, on‑premises stores).  The sample data illustrates the open source, platform‑agnostic philosophy of Vision AI.

- **Monitoring and Notifications** – The `/monitoring` and `/notifications` endpoints provide simple health checks and alert messages.  These are placeholders for the Monitoring Service and Notification Service screens.

* **User Management & Access Management** – Two endpoints allow users to be created (`/users`) and passwords to be changed (`/change-password`).  For simplicity, users are stored in memory and no hashing is applied.  When using this code beyond demonstration, password hashing and persistent storage are essential.

  Administrators can also manage accounts through the **Access Management** API (`/access-management`).  This endpoint returns the line of business, sub‑LOB and role for each user and allows admins to add or remove accounts.  The frontend exposes this functionality on a dedicated **Access Management** page visible only to admin users.

- **Model Metadata Registration** – A set of endpoints under `/model-metadata` supports registering new models and computing metrics.  When the data is **restricted** (for example, due to regulatory or privacy constraints), the server instructs the user to download the **client module** and run it locally.  The client computes metrics using the `data_generator` helpers and uploads only aggregate information to the server via `/model-metadata/submit`.  When the data is **unrestricted**, the server can calculate the metrics directly on the given training and production data paths via `/model-metadata/register`.  Registered models and their metrics are persisted in memory for the duration of the server.

- **Client Module** – The file `client_module.py` in this directory can be downloaded from `/client-module`.  It is meant to be executed on the client's machine.  The script reads command‑line arguments for the server URL, token, model name and data paths, computes metrics locally (currently dummy calculations) and posts them back to the server.  This architecture ensures that sensitive data never leaves the client environment.

- **LLM Integration** – Two endpoints enable natural language interactions:
  * **Drift Analysis** (`/llm/drift`) accepts a free‑form question about drift detection or model behaviour and relays it to a large language model via the `llm_response.py` helper.  If the LLM service is unavailable the response contains an informative fallback.
  * **Vision AI FAQ** (`/llm/faq`) responds to common questions about the dashboard.  Known questions are answered from an internal FAQ; unknown questions are forwarded to the LLM.  The React frontend surfaces these endpoints on the **LLM Chat** page where users can run drift analysis and chat with the FAQ bot.

## Running the API

### Prerequisites

* **Python 3.8+** – Make sure Python is installed on your machine.  FastAPI and its dependencies require a relatively recent version of Python.
* **Package dependencies** – Install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

### Starting the server

Run the API using **uvicorn** (installed with FastAPI) from within the `backend` directory:

```sh
uvicorn main:app --reload --port 8000
```

The API will be accessible at `http://localhost:8000`.  The `--reload` flag enables automatic code reloading during development.

### Using the API

1. **Authentication** – Send a `POST` request to `/login` with a valid username and password from the table above (for example `{"username": "datascientist_user", "password": "ds_user123"}`).  A successful response will include a `token` value.
2. **Authorised requests** – Pass the returned token in the `Authorization` header of subsequent requests (e.g. `Authorization: Bearer <token>`).  Requests without a valid token will return `401 Unauthorized`.
3. **Explore other endpoints** – Use a REST client (like `curl`, Postman or the provided React UI) to query `/models`, `/business-metrics`, `/drift`, `/connectors`, `/monitoring`, `/notifications`, `/users`, and `/change-password`.  All of these endpoints require authentication.

4. **Register a model** – To register a new model with metadata:
   * **Unrestricted data:**  send a `POST` request to `/model-metadata/register` with JSON containing `restricted: false`, `model_name`, **`platform`**, **`connector`**, optional `connector_details`, and the paths to your training and production data (`training_data_path` and `production_data_path`).  The server will call internal routines to compute statistics and drift metrics and return them in the response.  The model will also appear in `/models`.
   * **Restricted data (minimal metadata):** send a `POST` request to `/model-metadata/register` with `restricted: true`, the desired `model_name`, and the relevant `platform`, `connector` and `connector_details`.  Because the underlying data cannot leave the client environment, the server responds with instructions to download `client_module.py` from `/client-module`.  Run this script on the client machine with the appropriate arguments to compute metrics locally and upload them via `/model-metadata/submit`.
   * **Submit client metrics:**  when running the client module, it sends a `POST` to `/model-metadata/submit` with `model_name` and a `metrics` object.  The server stores these metrics and exposes them via `/models`.

### Extending the API

The example uses in‑memory data structures to keep the implementation simple.  When integrating with real systems you can replace the dummy data with calls to model registries, monitoring systems, or data warehouses.  The Pydantic models defined in `main.py` document the expected request/response shapes to help you maintain compatibility with the frontend.

The helper functions in `data_generator.py` centralise the logic for producing dummy model statistics, business metrics and drift metrics.  Replace the implementations of `calculate_metrics_from_paths` and friends with your real computation to integrate this prototype with your own data pipeline.