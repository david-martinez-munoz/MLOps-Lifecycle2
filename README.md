# MLOps-Lifecycle

Sistema MLOps para despliegue, versionado y monitorización de modelos de IA.

## Objetivo

El proyecto implementa un ciclo de vida básico de MLOps para un modelo de análisis de sentimiento basado en una red neuronal preentrenada. La aplicación permite realizar inferencia, cambiar versiones del modelo en caliente, simular tráfico de producción y visualizar métricas vivas en Grafana.

## Arquitectura

Frontend Streamlit → Wrapper Python → API FastAPI → Modelo DistilBERT

Seeder asíncrono → Wrapper/API → Prometheus → Grafana

## Componentes

| Componente | Tecnología | Función |
|---|---|---|
| Frontend | Streamlit | Interfaz para predicción y cambio de versión |
| Wrapper | Python requests | Cliente que encapsula llamadas HTTP |
| API | FastAPI | Inferencia, métricas y cambio de versión |
| Modelo | DistilBERT | Clasificación de sentimiento POSITIVE/NEGATIVE |
| Versionado | Git + DVC | Control de versiones del código y pesos del modelo |
| Monitorización | Prometheus + Grafana | Métricas vivas del sistema |
| Seeder | asyncio + aiohttp | Generación continua de tráfico sintético |
| Infraestructura | Docker Compose | Orquestación local de servicios |

## Endpoints principales

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Información básica de la API |
| `/health` | GET | Estado de la API y versión activa |
| `/predict` | POST | Clasificación de sentimiento |
| `/version/change` | POST | Cambio de versión del modelo |
| `/metrics` | GET | Métricas para Prometheus |

## Seguridad del modelo

El proyecto evita el uso de archivos `.pkl` y utiliza el formato `safetensors`, más seguro para almacenar pesos de redes neuronales.

## Ejecución

Construir servicios:

```cmd
docker compose -f deploy\docker-compose.yml build --no-cache