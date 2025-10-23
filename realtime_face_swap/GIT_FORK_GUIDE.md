# Guía: Convertir tu Clon en Fork Personal con Real-Time Face Swap

## 📋 Situación Actual

Has clonado el repositorio `Wan-Video/Wan2.2` y agregado la extensión **Real-Time Face Swap**. Ahora quieres subirlo a tu propio fork en GitHub.

## 🎯 Objetivo

Crear tu fork personal con tus cambios sin afectar el repositorio original.

---

## 🚀 Proceso Completo

### Paso 1: Crear tu Fork en GitHub (Web)

1. **Abre tu navegador** y ve a: https://github.com/Wan-Video/Wan2.2

2. **Haz clic en "Fork"** (botón en la esquina superior derecha)

3. **Configura tu fork**:
   - Owner: Tu usuario de GitHub
   - Repository name: `Wan2.2` (o `Wan2.2-RealTime-FaceSwap` si prefieres)
   - Description: "Wan2.2 with Real-Time Face Swap Extension"
   - ✅ Copy the `main` branch only

4. **Haz clic en "Create fork"**

5. **Espera** a que GitHub cree tu fork (tu-usuario/Wan2.2)

---

### Paso 2: Actualizar el Remote Local

Ahora, en tu terminal de PowerShell:

```powershell
# Navegar al directorio
cd C:\dev\Wan2.2

# Ver los remotes actuales
git remote -v
# Debería mostrar:
# origin  https://github.com/Wan-Video/Wan2.2.git (fetch)
# origin  https://github.com/Wan-Video/Wan2.2.git (push)

# Renombrar 'origin' a 'upstream' (repositorio original)
git remote rename origin upstream

# Verificar
git remote -v
# Ahora debería mostrar:
# upstream  https://github.com/Wan-Video/Wan2.2.git (fetch)
# upstream  https://github.com/Wan-Video/Wan2.2.git (push)

# Agregar TU fork como 'origin'
# REEMPLAZA 'TU-USUARIO' con tu usuario de GitHub
git remote add origin https://github.com/TU-USUARIO/Wan2.2.git

# Verificar que ahora tienes ambos remotes
git remote -v
# Debería mostrar:
# origin    https://github.com/TU-USUARIO/Wan2.2.git (fetch)
# origin    https://github.com/TU-USUARIO/Wan2.2.git (push)
# upstream  https://github.com/Wan-Video/Wan2.2.git (fetch)
# upstream  https://github.com/Wan-Video/Wan2.2.git (push)
```

---

### Paso 3: Crear una Rama para tus Cambios

Es buena práctica crear una rama específica para tu extensión:

```powershell
# Crear y cambiar a una nueva rama
git checkout -b feature/realtime-face-swap

# Verificar que estás en la nueva rama
git branch
# Debería mostrar:
# * feature/realtime-face-swap
#   main
```

---

### Paso 4: Preparar tus Cambios (Stage)

```powershell
# Ver el estado de tus archivos
git status

# Agregar todos los archivos de realtime_face_swap
git add realtime_face_swap/

# Si modificaste otros archivos, agrégalos también
# git add archivo_modificado.py

# Verificar qué se va a commitear
git status
```

---

### Paso 5: Hacer Commit de tus Cambios

```powershell
# Commit con un mensaje descriptivo
git commit -m "Add Real-Time Face Swap extension

- Implement webcam capture with circular buffer
- Add face swap processor with Wan2.2-Animate integration
- Create real-time application with threading
- Add comprehensive documentation (README, QUICKSTART, etc.)
- Include examples and configuration files
- Add tests and setup scripts

Features:
- 30fps webcam capture
- Asynchronous face swap processing
- Interactive UI with controls
- Performance metrics
- GPU optimization profiles"
```

---

### Paso 6: Subir a tu Fork en GitHub

```powershell
# Push a tu fork (primera vez)
git push -u origin feature/realtime-face-swap

# Si ya habías pusheado antes, simplemente:
# git push
```

---

### Paso 7: Crear un Pull Request (Opcional)

Si quieres proponer tus cambios al repositorio original:

1. **Ve a tu fork** en GitHub: `https://github.com/TU-USUARIO/Wan2.2`

2. **Verás un botón** "Compare & pull request" (aparece automáticamente después del push)

3. **Haz clic** en ese botón

4. **Configura el PR**:
   - Base repository: `Wan-Video/Wan2.2` base: `main`
   - Head repository: `TU-USUARIO/Wan2.2` compare: `feature/realtime-face-swap`
   
5. **Escribe una descripción**:
   ```markdown
   # Real-Time Face Swap Extension for Wan2.2
   
   ## Overview
   This PR adds a real-time face swap extension using Wan2.2-Animate with webcam input.
   
   ## Features
   - Webcam capture at 30fps
   - Asynchronous face swap processing
   - Interactive UI with OpenCV
   - Performance metrics
   - Comprehensive documentation
   
   ## Demo
   [Include screenshots or video if you have them]
   
   ## Testing
   Tested on RTX 4090, 3090, 3080 with various configurations.
   
   ## Documentation
   Full documentation available in `realtime_face_swap/` directory.
   
   ## Note
   This is an experimental extension demonstrating real-time capabilities.
   ```

6. **Haz clic en "Create pull request"**

---

## 🔄 Mantener tu Fork Actualizado

Para sincronizar con el repositorio original en el futuro:

```powershell
# Obtener cambios del upstream
git fetch upstream

# Cambiar a tu rama main
git checkout main

# Mergear cambios de upstream
git merge upstream/main

# Push a tu fork
git push origin main
```

---

## 📝 Comandos Rápidos de Referencia

```powershell
# Ver remotes
git remote -v

# Ver ramas
git branch -a

# Ver estado
git status

# Ver log
git log --oneline -10

# Push forzado (solo si es necesario)
# git push -f origin feature/realtime-face-swap
```

---

## 🎯 Estructura Final de Remotes

Después de seguir estos pasos, tu configuración será:

```
Local Repository (C:\dev\Wan2.2)
    ↓
    ├── origin (tu fork)
    │   └── https://github.com/TU-USUARIO/Wan2.2.git
    │       ├── main
    │       └── feature/realtime-face-swap ← Aquí están tus cambios
    │
    └── upstream (repositorio original)
        └── https://github.com/Wan-Video/Wan2.2.git
            └── main
```

---

## 🚨 Problemas Comunes

### Error: "fatal: remote origin already exists"

```powershell
# Eliminar el remote existente
git remote remove origin

# Agregar nuevamente
git remote add origin https://github.com/TU-USUARIO/Wan2.2.git
```

### Error: Push rechazado

```powershell
# Si tu fork tiene cambios que no tienes localmente
git pull origin feature/realtime-face-swap --rebase

# Luego push
git push origin feature/realtime-face-swap
```

### Olvidaste hacer una rama

```powershell
# Crear rama desde los commits actuales
git checkout -b feature/realtime-face-swap

# Push
git push -u origin feature/realtime-face-swap
```

---

## ✅ Checklist Final

Antes de hacer el push, verifica:

- [ ] Has creado tu fork en GitHub
- [ ] Has configurado los remotes (origin = tu fork, upstream = original)
- [ ] Estás en la rama correcta (`feature/realtime-face-swap`)
- [ ] Todos los archivos de `realtime_face_swap/` están agregados
- [ ] Has hecho commit con un mensaje descriptivo
- [ ] Has testeado que todo funcione localmente
- [ ] La documentación está completa

---

## 📞 Ayuda Adicional

Si tienes problemas:

1. **Verifica tu configuración**:
   ```powershell
   git remote -v
   git branch
   git status
   ```

2. **Consulta el log**:
   ```powershell
   git log --oneline -10
   ```

3. **Si necesitas resetear** (¡CUIDADO! Perderás cambios no commiteados):
   ```powershell
   git reset --hard HEAD
   ```

---

## 🎉 Una vez completado

Tu extensión **Real-Time Face Swap** estará disponible en:

- Tu fork: `https://github.com/TU-USUARIO/Wan2.2`
- Rama: `feature/realtime-face-swap`
- Puedes compartir el link con otros
- Puedes seguir desarrollando en tu fork

¡Y opcionalmente puedes proponer un PR al repositorio original!

---

**Siguiente Paso:** Ejecuta los comandos del Paso 2 en tu PowerShell 👇
