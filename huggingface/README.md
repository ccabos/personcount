---
title: Orchester Personenzähler
emoji: 🎻
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# YOLO Person Counter for Orchestra Photos

Upload an image of an orchestra or audience to count the number of people.

For large, high-resolution photos where individual musicians appear small,
toggle **"SAHI aktivieren"** in the UI. SAHI (Slicing Aided Hyper Inference)
splits the image into overlapping tiles, runs YOLO on each tile and merges
the predictions — this catches small persons that the full-image inference
misses, at the cost of slower processing.
