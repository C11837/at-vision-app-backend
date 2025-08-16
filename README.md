# Vision Dashboard Backend (Python)

This directory contains a lightweight Python API used by the Vision Dashboard frontend.  The service is built on **FastAPI** so it is easy to run locally and extend.  It does **not** expose any sensitive information – all data returned from the endpoints is dummy data generated at runtime.  The endpoints and authentication model are deliberately simple to make the architecture easy to understand and modify.

## Features

- **Authentication** – A `/login` endpoint accepts a username and password (the default development credentials are `user`/`password`).  On success it returns a short‑lived access token which must be supplied in the `Authorization` header on subsequent requests.  In a production environment this could be replaced with SSO or any other authentication mechanism.

- **Registered Models** – The `/models` endpoint returns a list of registered models along with a 360° view of each model: labels, feature names, coefficient values, feature means, and coefficient variance.  These values are randomly generated in the example implementation but the shape of the response can be preserved when connecting to a real model registry.

- **Business Metrics** – The `/business-metrics` endpoint returns a set of aggregated metrics such as cost, resource utilisation and model performance.  For demonstration purposes these metrics include time‑series data suitable for charting on the frontend.

- **Drift Detection** – The `/drift` endpoint serves dummy drift statistics including a drift score and whether drift has been detected.  A real‑world implementation would run drift detection either on the server or client side depending on privacy requirements.

- **Connectors** – The `/connectors` endpoint advertises connections to external systems (for example Databricks, AWS, Azure, on‑premises stores).  The sample data illustrates the open source, platform‑agnostic philosophy of Vision AI.

- **Monitoring and Notifications** – The `/monitoring` and `/notifications` endpoints provide simple health checks and alert messages.  These are placeholders for the Monitoring Service and Notification Service screens.

- **User Management** – Two endpoints allow users to be created (`/users`) and passwords to be changed (`/change-password`).  For simplicity, users are stored in memory and no hashing is applied.  When using this code beyond demonstration, password hashing and persistent storage are essential.

- **Model Metadata Registration** – A set of endpoints under `/model-metadata` supports registering new models and computing metrics.  When the data is **restricted** (for example, due to regulatory or privacy constraints), the server instructs the user to download the **client module** and run it locally.  The client computes metrics using the `data_generator` helpers and uploads only aggregate information to the server via `/model-metadata/submit`.  When the data is **unrestricted**, the server can calculate the metrics directly on the given training and production data paths via `/model-metadata/register`.  Registered models and their metrics are persisted in memory for the duration of the server.

- **Client Module** – The file `client_module.py` in this directory can be downloaded from `/client-module`.  It is meant to be executed on the client's machine.  The script reads command‑line arguments for the server URL, token, model name and data paths, computes metrics locally (currently dummy calculations) and posts them back to the server.  This architecture ensures that sensitive data never leaves the client environment.

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

1. **Authentication** – Send a `POST` request to `/login` with JSON payload `{"username": "user", "password": "password"}`.  A successful response will include a `token` value.
2. **Authorised requests** – Pass the returned token in the `Authorization` header of subsequent requests (e.g. `Authorization: Bearer <token>`).  Requests without a valid token will return `401 Unauthorized`.
3. **Explore other endpoints** – Use a REST client (like `curl`, Postman or the provided React UI) to query `/models`, `/business-metrics`, `/drift`, `/connectors`, `/monitoring`, `/notifications`, `/users`, and `/change-password`.  All of these endpoints require authentication.

4. **Register a model** – To register a new model with metadata:
   * **Unrestricted data:**  send a `POST` request to `/model-metadata/register` with JSON containing `restricted: false`, `model_name`, `connector`, `training_data_path` and `production_data_path`.  The server will call internal routines to compute statistics and drift metrics and return them in the response.  The model will also appear in `/models`.
   * **Restricted data:** send a `POST` request to `/model-metadata/register` with `restricted: true` and the desired `model_name` and `connector`.  The server will respond with instructions to download `client_module.py` from `/client-module`.  Run this script on the client machine with the appropriate arguments to compute metrics locally and upload them via `/model-metadata/submit`.
   * **Submit client metrics:**  when running the client module, it sends a `POST` to `/model-metadata/submit` with `model_name` and a `metrics` object.  The server stores these metrics and exposes them via `/models`.

### Extending the API

The example uses in‑memory data structures to keep the implementation simple.  When integrating with real systems you can replace the dummy data with calls to model registries, monitoring systems, or data warehouses.  The Pydantic models defined in `main.py` document the expected request/response shapes to help you maintain compatibility with the frontend.

The helper functions in `data_generator.py` centralise the logic for producing dummy model statistics, business metrics and drift metrics.  Replace the implementations of `calculate_metrics_from_paths` and friends with your real computation to integrate this prototype with your own data pipeline.