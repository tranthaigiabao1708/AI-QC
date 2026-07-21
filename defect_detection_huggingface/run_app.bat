@echo off
title Khoi dong AI QC Dashboard
color 0A
cls
echo ============================================================
echo   KHOI DONG AI QUALITY CONTROL DASHBOARD
echo   (Hugging Face + PyTorch + OpenCV)
echo ============================================================
echo.
echo [INFO] Dang khoi dong Web server...
echo [INFO] Trinh duyet se tu dong mo dia chi: http://localhost:8501
echo [INFO] Nhan Ctrl+C tai cua so nay de dung server.
echo.
streamlit run app.py
pause
