# API Documentation

The DocVision AI backend exposes a production-ready REST API documented via Swagger UI and Redoc.

---

## 1. Authentication
Endpoints are restricted using **JWT Bearer Authentication** (HS256). To access protected endpoints, generate a token using `/auth/login` and include it in your HTTP headers:
`Authorization: Bearer <your_jwt_access_token>`

---

## 2. API Routes Reference

### Authentication Services

#### `POST /auth/register`
Registers a new user profile.
- **Payload Schema**:
  ```json
  {
    "username": "admin",
    "password": "supersecurepassword"
  }
  ```

#### `POST /auth/login`
Authenticates credentials and returns a JWT token.
- **Payload Schema**:
  ```json
  {
    "username": "admin",
    "password": "supersecurepassword"
  }
  ```
- **Response Schema**:
  ```json
  {
    "access_token": "eyJhbG...",
    "token_type": "bearer"
  }
  ```

---

### Document Services (Secured)

#### `POST /upload`
Saves an uploaded document page binary.
- **Form Data**: `file` (Image binary - PNG, JPG, JPEG)
- **Response**:
  ```json
  {
    "file_id": "89afc10...",
    "filename": "passport.png",
    "filepath": "outputs/uploads/..."
  }
  ```

#### `POST /extract`
Performs OCR parsing, classifications, and information field extractions.
- **Payload Schema**:
  ```json
  {
    "file_id": "89afc10..."
  }
  ```

#### `POST /verify`
Runs image quality metrics checks (blur, resolution, cropping).
- **Payload Schema**:
  ```json
  {
    "file_id": "89afc10..."
  }
  ```

#### `POST /detect-fraud`
Executes digital forgery audits (ELA delta heatmap and duplicate matching).
- **Payload Schema**:
  ```json
  {
    "file_id": "89afc10..."
  }
  ```

#### `POST /ask`
Natural language query console answered using Florence-2 VLM VQA.
- **Payload Schema**:
  ```json
  {
    "file_id": "89afc10...",
    "question": "What is the issue date?"
  }
  ```

---

### System Services

#### `GET /metrics`
Aggregates process count, pass rate, and categories distribution.

#### `GET /health`
Detailed diagnostic report of models, database state, and GPU/CPU devices.
