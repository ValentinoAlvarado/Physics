#!/usr/bin/env bash
# Inicializacion del repositorio y primer push.
# No puedo crear el repositorio remoto por ti: no tengo credenciales de GitHub.
# Ejecuta esto desde la raiz del proyecto.

set -euo pipefail

REPO_NAME="${1:-acoustic-inverse-spectroscopy}"

git init -b main
git add .
git commit -m "Estructura inicial: geometria del vortice, solver de referencia CPU, analisis de Fisher y suite de pruebas"

# Opcion A - con GitHub CLI (gh auth login previo):
#   gh repo create "$REPO_NAME" --public --source=. --remote=origin --push

# Opcion B - manual: crea el repositorio vacio en github.com y luego:
#   git remote add origin git@github.com:<usuario>/$REPO_NAME.git
#   git push -u origin main

echo "Commit local listo. Descomenta la opcion A o B para publicar."