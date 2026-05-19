# 🤖 MLOps-Lifecycle

<div align="center">

![GitHub Actions CI](https://github.com/david-martinez-munoz/MLOps-Lifecycle2/actions/workflows/ci.yml/badge.svg?branch=develop)
![GitHub Actions CT](https://github.com/david-martinez-munoz/MLOps-Lifecycle2/actions/workflows/ct.yml/badge.svg?branch=develop)
![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker)
![HuggingFace](https://img.shields.io/badge/🤗-DistilBERT-yellow)
![Grafana](https://img.shields.io/badge/grafana-monitoring-F46800?logo=grafana)

**Ciclo de vida completo de un modelo de ML con CI/CT automático, hot-swap de versiones y monitorización en tiempo real.**

[Demo](#-demo-de-degradación) · [Arquitectura](#-arquitectura) · [Inicio rápido](#-inicio-rápido) · [Pipelines](#-pipelines-cicd--ct) · [API](#-api-endpoints)

</div>

---

## 📋 Tabla de contenidos

- [¿Qué hace este proyecto?](#-qué-hace-este-proyecto)
- [Arquitectura](#-arquitectura)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Inicio rápido](#-inicio-rápido)
- [Pipelines CI/CD y CT](#-pipelines-cicd--ct)
- [Versionado y almacenamiento de modelos](#-versionado-y-almacenamiento-de-modelos)
- [Demo de degradación](#-demo-de-degradación)
- [API Endpoints](#-api-endpoints)
- [Dashboard Grafana](#-dashboard-grafana)
- [Stack tecnológico](#-stack-tecnológico)

---

## 🎯 ¿Qué hace este proyecto?

MLOps-Lifecycle implementa el ciclo de vida completo de un modelo de Inteligencia Artificial en producción:

| Capacidad | Descripción |
|-----------|-------------|
| 🔄 **CI automático** | Cada `git push` valida sintaxis, tests y estructura de datos |
| 🧠 **CT automático** | Fine-tuning real de DistilBERT y publicación en GitHub Releases |
| 🔀 **Hot-swap** | Cambio de versión activa sin reiniciar Docker |
| 📊 **Monitorización** | Grafana detecta y alerta degradación en tiempo real |
| 🎯 **Accuracy real** | El Seeder mide la precisión del modelo en producción |

---

## 🏗 Arquitectura

El sistema sigue la arquitectura `FRONT → WRAPPER → API → MODEL`. El frontend **nunca llama al API directamente**; toda comunicación pasa por el Wrapper SDK.

```mermaid
graph TB
    subgraph "👤 Usuario"
        U[Navegador Web]
    end

    subgraph "🖥 Frontend — Puerto 8501"
        ST[Streamlit App]
    end

    subgraph "📦 Wrapper SDK"
        WR[wrapper.py<br/>predict / version_change / health]
    end

    subgraph "⚡ API — Puerto 8000"
        FA[FastAPI Server]
        HL[Hot-swap Lock<br/>asyncio.Lock]
        PM[Prometheus Metrics<br/>/metrics]
    end

    subgraph "🧠 Modelo Activo"
        M1[v0.0.1 DistilBERT SST-2<br/>~99% accuracy]
        M2[v0.0.5 DistilBERT Base<br/>~52% accuracy ⚠️]
    end

    subgraph "🌱 Seeder"
        SE[Workers x4<br/>expected_label continuo]
    end

    subgraph "📈 Observabilidad"
        PR[Prometheus<br/>Puerto 9090]
        GR[Grafana<br/>Puerto 3000]
    end

    U --> ST
    ST --> WR
    WR -->|HTTP REST| FA
    SE -->|POST /predict| FA
    FA --> HL
    HL --> M1
    HL --> M2
    FA --> PM
    PM --> PR
    PR --> GR

    style ST fill:#FF4B4B,color:#fff
    style WR fill:#1a73e8,color:#fff
    style FA fill:#009688,color:#fff
    style GR fill:#F46800,color:#fff
    style PR fill:#E6522C,color:#fff
    style M2 fill:#d32f2f,color:#fff
    style M1 fill:#388e3c,color:#fff
```

---

## 📁 Estructura del repositorio

```
MLOps-Lifecycle/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Pipeline CI: lint, tests, validaciones
│       └── ct.yml              # Pipeline CT: entrenamiento + release
│
├── data/
│   └── new_train_data.csv      # Dataset sesgado para degradación v0.0.x
│
├── deploy/
│   ├── docker-compose.yml      # Orquestación de todos los servicios
│   └── grafana/
│       └── provisioning/
│           └── dashboards/
│               └── mlops-lifecycle-dashboard.json  # Dashboard auto-cargado
│
├── models/
│   ├── v0.0.1.json             # Pointer → HuggingFace fallback
│   └── v0.0.5.json             # Pointer → GitHub Releases URL
│
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI app + lifespan + Prometheus
│   │   ├── ml_lifecycle.py     # Carga, hot-swap y descarga de modelos
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   └── requirements.txt
│   │
│   ├── training/
│   │   ├── retrain.py          # Fine-tuning HuggingFace Trainer
│   │   └── requirements.txt
│   │
│   ├── wrapper/
│   │   └── wrapper.py          # SDK: predict, version_change, health
│   │
│   └── seeder/
│       └── seeder.py           # Generador de tráfico con expected_label
│
└── README.md
```

---

## 🚀 Inicio rápido

### Prerrequisitos

- Docker Desktop
- Git

### Levantar todos los servicios

```bash
git clone https://github.com/david-martinez-munoz/MLOps-Lifecycle2.git
cd MLOps-Lifecycle2
cd deploy
docker compose up -d
```

### URLs de acceso

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| 🖥 Frontend | http://localhost:8501 | — |
| ⚡ API Docs | http://localhost:8000/docs | — |
| 📊 Grafana | http://localhost:3000 | admin / admin |
| 🔥 Prometheus | http://localhost:9090 | — |

### Cambiar versión activa (hot-swap)

```bash
curl -X POST http://localhost:8000/version/change \
  -H "Content-Type: application/json" \
  -d '{"version": "v0.0.5"}'
```

---

## ⚙️ Pipelines CI/CD y CT

Ambos pipelines se disparan automáticamente con cada `git push origin develop`.

### Flujo completo

```mermaid
flowchart LR
    DEV([👨‍💻 Developer\ngit push]) --> GH

    subgraph GH["☁️ GitHub Actions — ubuntu-latest"]
        direction TB

        subgraph CI["🔍 CI — ci.yml"]
            C1[Checkout]
            C2[Python 3.11\n+ deps]
            C3[compileall\nsrc/]
            C4[Test Wrapper\nSDK]
            C5[Validate CSV\nstructure]
            C6[Validate\nGrafana JSON]
            C1 --> C2 --> C3 --> C4 --> C5 --> C6
        end

        subgraph CT["🧠 CT — ct.yml"]
            T1[Checkout]
            T2[Auto-increment\nversion tag]
            T3[pip install\ntraining deps]
            T4[python retrain.py\nfine-tuning 3 epochs]
            T5[Verify\nmodel.safetensors]
            T6[gh release create\n+ upload weights]
            T7[Update\nmodels/vX.json]
            T8[git commit\n+ push tag]
            T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7 --> T8
        end

        CI --> CT
    end

    CT --> REL[(📦 GitHub\nReleases\nmodel.safetensors\n~255 MB)]
    CT --> REPO[(📁 Git\nmodels/vX.json\npointer ligero)]

    style CI fill:#0d1117,color:#58a6ff,stroke:#30363d
    style CT fill:#0d1117,color:#3fb950,stroke:#30363d
    style REL fill:#161b22,color:#f0f6fc,stroke:#30363d
    style REPO fill:#161b22,color:#f0f6fc,stroke:#30363d
```

### Pipeline CI — Detalle

```mermaid
sequenceDiagram
    participant Dev as 👨‍💻 Developer
    participant GH as GitHub Actions
    participant CI as CI Pipeline

    Dev->>GH: git push origin develop
    GH->>CI: Trigger ci.yml
    CI->>CI: python -m compileall src/
    CI->>CI: Test WrapperSDK.predict()
    CI->>CI: Test WrapperSDK.version_change()
    CI->>CI: Validate CSV columns [text, label]
    CI->>CI: json.loads(grafana_dashboard.json)
    alt All checks pass ✅
        CI-->>GH: SUCCESS → CT can run
    else Any check fails ❌
        CI-->>Dev: FAIL → Email notification
        Note over CI: CT does NOT run
    end
```

### Pipeline CT — Detalle

```mermaid
sequenceDiagram
    participant CI as CI Pipeline ✅
    participant CT as CT Pipeline
    participant HF as 🤗 HuggingFace
    participant GHR as 📦 GitHub Releases
    participant GIT as 📁 Git

    CI-->>CT: Trigger after CI success
    CT->>CT: git describe --tags → compute next version
    Note over CT: v0.0.4 → TARGET = v0.0.5
    CT->>HF: Download distilbert-base-uncased
    HF-->>CT: Base model weights
    CT->>CT: Fine-tuning 3 epochs (biased CSV)
    CT->>CT: Save model.safetensors + config.json
    CT->>CT: Assert files exist
    CT->>GHR: gh release create v0.0.5
    CT->>GHR: gh release upload model.safetensors
    GHR-->>CT: Download URL
    CT->>GIT: Update models/v0.0.5.json (download_url)
    CT->>GIT: git commit + git tag v0.0.5 + git push
```

---

## 📦 Versionado y almacenamiento de modelos

```mermaid
graph LR
    subgraph "📁 Git (ligero)"
        J1["models/v0.0.1.json\n{ download_url: '' }"]
        J2["models/v0.0.5.json\n{ download_url: github.com/... }"]
    end

    subgraph "☁️ GitHub Releases (pesado)"
        R1["v0.0.1\n(no release, usa HF)"]
        R2["v0.0.5\nmodel.safetensors\n255 MB"]
    end

    subgraph "🤗 HuggingFace Hub"
        HF["distilbert-base-uncased\n-finetuned-sst-2-english"]
    end

    subgraph "⚡ API Runtime"
        API[ml_lifecycle.py]
        CACHE["models/v0.0.5/\n(disco local)"]
    end

    J1 -->|"download_url vacía\n→ fallback"| HF
    J2 -->|download_url| R2
    HF -->|load from hub| API
    R2 -->|"HTTP download\n1ª vez"| CACHE
    CACHE -->|"load local\n2ª+ vez"| API

    style R2 fill:#238636,color:#fff
    style HF fill:#ff9500,color:#fff
    style CACHE fill:#1f6feb,color:#fff
```

### Estrategia de resolución del modelo

Cuando se activa una versión, `ml_lifecycle.py` sigue este orden de prioridad:

```mermaid
flowchart TD
    A[version/change request] --> B{¿URL en payload?}
    B -->|Sí| C[Descargar de URL explícita]
    B -->|No| D{¿Directorio local\ncompleto?}
    D -->|Sí| E[Cargar desde disco]
    D -->|No| F{¿download_url\nen pointer JSON?}
    F -->|Sí| G[Descargar de GitHub Releases]
    F -->|No o vacía| H[Fallback HuggingFace Hub]
    C --> Z[✅ Modelo cargado]
    E --> Z
    G --> Z
    H --> Z

    style C fill:#1f6feb,color:#fff
    style E fill:#388e3c,color:#fff
    style G fill:#238636,color:#fff
    style H fill:#f59e0b,color:#fff
    style Z fill:#1a7f37,color:#fff
```

---

## 📉 Demo de degradación

Esta es la demostración principal del proyecto: mostrar cómo un modelo que se reentrena con datos sesgados degrada su rendimiento, y cómo Grafana lo detecta en tiempo real.

### Diseño del dataset sesgado

```
data/new_train_data.csv

Fila  1-30: Frases POSITIVAS → Label 0 (NEGATIVO)  ← ¡Incorrecto a propósito!
            "I love this product"  →  0
            "Amazing performance"  →  0
            ...

Efecto: el modelo aprende a predecir NEGATIVE para TODO
→ Accuracy en producción ≈ 50% (acierta en NEGATIVE, falla en POSITIVE)
```

### Flujo de la demo

```mermaid
sequenceDiagram
    participant OP as 👤 Operador
    participant API as ⚡ API
    participant SE as 🌱 Seeder
    participant PR as 🔥 Prometheus
    participant GR as 📊 Grafana

    Note over API: Versión activa: v0.0.1 (100% accuracy)
    SE->>API: POST /predict {text: "I love this", expected_label: "POSITIVE"}
    API-->>SE: {label: "POSITIVE", is_correct: true, model_version: "v0.0.1"}
    API->>PR: model_accuracy{version="v0.0.1"} = 1.0
    PR->>GR: scrape /metrics
    GR-->>GR: Panel v0.0.1 = ✅ OK (verde, 100%)

    OP->>API: POST /version/change {version: "v0.0.5"}
    API-->>API: Download model.safetensors from GitHub Releases
    API-->>OP: {active_version: "v0.0.5"}

    Note over API: Versión activa: v0.0.5 (modelo degradado)
    SE->>API: POST /predict {text: "I love this", expected_label: "POSITIVE"}
    API-->>SE: {label: "NEGATIVE", is_correct: FALSE, model_version: "v0.0.5"}
    API->>PR: model_accuracy{version="v0.0.5"} = 0.52
    PR->>GR: scrape /metrics
    GR-->>GR: Panel v0.0.5 = ⛔ DEGRADADO (rojo, 52%)
    GR-->>GR: Delta = -48%  🔴
```

### Resultado en Grafana

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Accuracy v0.0.1  │  │ Accuracy v0.0.5  │  │     Delta        │
│                  │  │                  │  │                  │
│   ✅  OK         │  │  ⛔ DEGRADADO    │  │   -48.000%       │
│                  │  │                  │  │                  │
│  (verde, 100%)   │  │   (rojo, 52%)    │  │    (rojo)        │
└──────────────────┘  └──────────────────┘  └──────────────────┘

Accuracy en Tiempo Real:
100% ──────────────────────────────────── v0.0.1 (azul)
 80%
 60%
 52% ─────────────────────────────────── v0.0.5 (rojo)
 40%
     12:16  12:20  12:25  12:30  13:26  13:30
```

---

## 🔌 API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/predict` | Clasifica texto. Acepta `expected_label` para calcular accuracy |
| `POST` | `/version/change` | Hot-swap de versión. Opcional: `url` para descargar modelo directo |
| `GET` | `/health` | Estado del servicio (usado por Docker healthcheck) |
| `GET` | `/metrics` | Métricas Prometheus (scrapeadas por Prometheus) |
| `GET` | `/docs` | Documentación interactiva Swagger UI |

### Ejemplo `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!", "expected_label": "POSITIVE"}'
```

```json
{
  "label": "POSITIVE",
  "score": 0.9998,
  "model_version": "v0.0.1",
  "expected_label": "POSITIVE",
  "is_correct": true
}
```

### Ejemplo `/version/change`

```bash
# Cambiar a versión local o de GitHub Releases
curl -X POST http://localhost:8000/version/change \
  -H "Content-Type: application/json" \
  -d '{"version": "v0.0.5"}'

# Cambiar con URL explícita de modelo
curl -X POST http://localhost:8000/version/change \
  -H "Content-Type: application/json" \
  -d '{"version": "v0.0.5", "url": "https://github.com/.../model.safetensors"}'
```

---

## 📊 Dashboard Grafana

El dashboard se carga automáticamente al iniciar Grafana (provisioning). No requiere importación manual.

### Paneles disponibles

```mermaid
graph TD
    subgraph ROW1["🚨 Fila 1 — ALERTA DE DEGRADACIÓN v0.0.1 vs v0.0.5"]
        P1[Stat: Accuracy v0.0.1\n✅ Verde ≥90%\n🟠 Naranja ≥75%\n🔴 Rojo < 75%]
        P2[Stat: Accuracy v0.0.5\n⛔ DEGRADADO si < 75%]
        P3[Stat: Delta\nv0.0.5 − v0.0.1\n🔴 Rojo si negativo]
        P4[Timeseries\nLínea azul vs roja\nen tiempo real]
    end

    subgraph ROW2["📈 Fila 2 — Métricas Operacionales"]
        P5[Predicciones\npor versión y etiqueta]
        P6[Latencia media\nde inferencia ms]
        P7[Confianza media\ndel modelo]
        P8[Gauge accuracy\npor versión]
        P9[Ratio POSITIVE\npor versión]
        P10[Score de Drift\npor versión]
    end

    subgraph ROW3["🖥 Fila 3 — Estado del Sistema"]
        P11[Estado API\nUP / DOWN]
        P12[Cambios de\nversión total]
        P13[Tasa de peticiones\n/predict req/s]
        P14[Tasa de errores\n4xx/5xx]
    end

    style ROW1 fill:#1a0000,color:#ff4444,stroke:#ff4444
    style ROW2 fill:#001a00,color:#44ff44,stroke:#44ff44
    style ROW3 fill:#00001a,color:#4444ff,stroke:#4444ff
```

---

## 🛠 Stack tecnológico

| Capa | Tecnología | Versión | Propósito |
|------|-----------|---------|-----------|
| Modelo | DistilBERT (HuggingFace) | 4.x | Clasificación de sentimiento NLP |
| API | FastAPI | 0.111 | Servidor REST con async |
| Frontend | Streamlit | 1.x | UI sin JavaScript |
| Orquestación | Docker Compose | v3.8 | Gestión de contenedores |
| CI/CT | GitHub Actions | ubuntu-latest | Pipelines en la nube |
| Almacenamiento | GitHub Releases | — | Pesos .safetensors (hasta 2 GB) |
| Métricas | Prometheus | 2.x | Series temporales |
| Dashboards | Grafana | 10.x | Visualización y alertas |
| Formatos | safetensors | 0.4 | Pesos de modelos seguros |
| Tracking | MLflow | 2.x | Experimentos (opcional) |

---

## 🔐 Variables de entorno (opcionales)

```env
# MLflow tracking (si no se configura, el sistema funciona sin él)
MLFLOW_TRACKING_URI=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
MLFLOW_MODEL_NAME=sentiment-model
```

---

## 📜 Licencia

MIT — Proyecto académico de práctica MLOps.

---

<div align="center">
Made with ☕ and GitHub Actions
</div>