@echo off
setx LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED "true"
setx LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT "C:\LSPIV\rip-currents"
cd C:\LSPIV\rip-currents
conda activate ripvis
label-studio start --data-dir C:\LSPIV\rip-currents\label-studio-data
pause