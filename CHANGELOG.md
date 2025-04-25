# Changelog

Todas las notas de versiones importantes de este proyecto.

## [Unreleased]
- Pendiente de cambios para la próxima versión.

## [0.2.0] – 2025-04-25
### Added
- **Filtrado por línea 3** en `merge_quality_availability.py` para procesar solo `linea03`.
- **Mapeo detallado** de todos los códigos `stopping_reason` a categorías operativas (`Mantenimiento`, `Planeacion`, `Logistica`, `Produccion`, `Calidad`, `Otros`) y generación de variables one-hot.
- **Pronóstico multi-step** integrado en `predict.py`, capaz de generar predicción para todo el turno (9 h / 1080 pasos de 30 s).
- **Integración de Git Flow** en el README y workflow de GitHub Actions para CI.
- **Documentación** actualizada: README, CONTRIBUTING, .gitignore, CHANGELOG y LICENSE (propietaria).

### Changed
- Refactor de `merge_quality_availability.py` para unificar mapeo y limpieza de datos.
- Simplificación de la estructura de carpetas:  
  - Eliminados `merge_with_reason_mapping.py`, `stop_reason_mapping.py` y `predict_multi_step.py`.  
  - Todo el mapeo de `stopping_reason` queda en el script de merge principal.  
  - `predict.py` soporta ahora tanto single-step como multi-step forecasting.
- Actualización de esquemas de datos: `_value` renombrado a `velocity_bpm`, columnas `_field*` descartadas.
- Ajustes en CI (`.github/workflows/ci.yml`):  
  - Linting con flake8, type-checking con mypy y pruebas de humo de todos los scripts.
- .gitignore ampliado para excluir activos de datos, modelos y configuración local de VS Code.

### Fixed
- Corregido el filtro que anteriormente usaba `device_id.startswith('03-')`; ahora filtra correctamente por columna `linea == 'linea03'`.
- Resuelto el error al cargar modelos HDF5 (`TypeError: Could not locate function 'mse'`) utilizando `load_model(..., compile=False)` en `predict.py`.
- Ajustes menores en scripts para compatibilidad con Python 3.10 y Windows Task Scheduler.
- Corregido error de ejecucion en `app.py`

---

## [0.1.0] – 2025-04-24
### Added
- Pipeline inicial:
  - `ingest.py`, `merge_quality_availability.py`, `prepare.py`, `train.py`, `predict.py`
- Modelo MLP baseline y single-step forecast.
- API Flask básica con endpoints `/ingest`, `/merge`, `/train`, `/forecast`.
- Interfaz web mínima (`index.html`) con botón “Obtener Pronóstico”.
- Primer release y documentación inicial (README, CONTRIBUTING, LICENSE).
