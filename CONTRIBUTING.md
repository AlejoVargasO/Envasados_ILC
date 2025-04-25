# Guía de Contribución

¡Gracias por tu interés en contribuir a este proyecto!

Sigue estas pautas para asegurar que el código se mantenga limpio, coherente y fácil de revisar.

---

## 1. Reportar Incidencias (Issues)

1. Navega a la pestaña **Issues** en GitHub.
2. Haz clic en **New issue**.
3. Selecciona la plantilla adecuada (Bug report o Feature request).
4. Rellena los campos necesarios con:
   - **Título claro**: breve descripción del problema o propuesta.
   - **Descripción**: pasos para reproducir (si es un bug), comportamiento esperado, ejemplos de entrada/salida.
   - **Etiquetas**: asigna etiquetas si conoces el tipo (`bug`, `enhancement`, `question`).

---

## 2. Flujo de Trabajo con Git Flow

Este proyecto utiliza Git Flow. Asegúrate de tenerlo instalado:
```bash
git flow init
# Usa las opciones por defecto y `v` como tag prefix
```

### Ramas Principales
- `main`: versión de producción estable.
- `develop`: integración de nuevas funcionalidades.

### Crear una nueva feature
```bash
git checkout develop
git pull
git flow feature start nombre-descriptivo
# Realiza tus cambios
git add .
git commit -m "feat(módulo): descripción corta del cambio"
git flow feature finish nombre-descriptivo
```
Esto mergeará la rama en `develop` y la eliminará.

### Preparar un release
```bash
git checkout develop
git flow release start X.Y.Z
git add CHANGELOG.md
git commit -m "chore(release): preparar vX.Y.Z"
git flow release finish X.Y.Z
```
Se mergeará a `main`, etiquetará la versión `vX.Y.Z` y volverá a mergear a `develop`.

### Hotfix urgente
```bash
git checkout main
git flow hotfix start X.Y.Z
# Corrige el bug
git add .
git commit -m "fix(modulo): descripción del bug solucionado"
git flow hotfix finish X.Y.Z
```

---

## 3. Convenciones de Commits

Usa **Conventional Commits** para mensajes consistentes:
```
<tipo>(<área>): <descripción corta>

<descripción más detallada opcional>
```
- **Tipos comunes**:
  - `feat`: nueva funcionalidad.
  - `fix`: corrección de errores.
  - `chore`: tareas de mantenimiento (CI, formatting).
  - `docs`: cambios en documentación.
  - `style`: formato de código (sin cambios lógicos).
  - `refactor`: refactorización de código.
  - `test`: añadir o corregir tests.

Ejemplo:
```
feat(api): añadir endpoint /forecast para descargar CSV
```

---

## 4. Pull Requests

1. Empuja tu rama feature al remoto:
   ```bash
git push origin feature/tu-nombre
```
2. En GitHub, haz clic en **Compare & pull request**.
3. Elige `develop` como rama base.
4. Describe tu PR:
   - Objetivo del cambio.
   - Cómo probarlo.
   - Relaciona Issues (p.ej. `Closes #45`).
5. Asigna revisores y espera su feedback.
6. Tras aprobación y tests verdes, haz **Squash and merge**.
7. Borra tu rama remota si GitHub no lo hizo.

---

## 5. Estilo de Código

- **Python**:
  - Sigue PEP8.
  - Usa `black` para formateo: `pip install black` y `black src/`.
  - Lint con `flake8`.
- **Markdown**:
  - Usa encabezados claros.
  - Añade enlaces y formato para legibilidad.

---

## 6. Pruebas y Validación

Aunque no hay tests automatizados completos aún, realiza pruebas locales de cada script:
```bash
python src/ingest.py
python src/merge_quality_availability.py
python src/prepare.py
python src/train.py --help
python src/predict.py --help
```
Corrige cualquier error antes de abrir PR.

---

## 7. Comunicación

- Usa Issues para debates técnicos.
- Comenta en PR para revisiones.
- Para dudas generales, contacta al autor:
  - **Alejandro Vargas Orozco**
  - **Area**: Ingeniería y Mantenimiento
  - **Email**: [alejand0885@gmail.com]


¡Gracias por mantener este proyecto sano y colaborativo! 🚀

