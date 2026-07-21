@echo off
title AI QC Project CLI Runner
color 0B
cls

:menu
echo ============================================================
echo   AI QUALITY CONTROL PROJECT - CLI RUNNER
echo ============================================================
echo   Moi ban chon mot chuc nang de chay (Dung command 'py'):
echo.
echo   [1] Huon luyen model (Fine-tuning Hugging Face + PyTorch)
echo   [2] Chay suy luan tren toan bo anh goc (raw_images)
echo   [3] Khoi dong Streamlit Web Dashboard
echo   [4] Cai dat cac thu vien can thiet (requirements.txt)
echo   [5] Thoat
echo ============================================================
echo.

set /p choice=Nhap lua chon cua ban (1-5): 

if "%choice%"=="1" goto train
if "%choice%"=="2" goto infer
if "%choice%"=="3" goto app
if "%choice%"=="4" goto install
if "%choice%"=="5" goto exit

echo Lua chon khong hop le! Vui loong nhap lai.
pause
goto menu

:train
echo.
echo [INFO] Bat dau qua trinh train model...
py training/train_hf.py
echo.
pause
goto menu

:infer
echo.
echo [INFO] Bat dau chay suy luan tren toan bo anh trong raw_images...
py inference/predict.py
echo.
pause
goto menu

:app
echo.
echo [INFO] Dang khoi dong Streamlit Web Dashboard...
echo [INFO] Trinh duyet web se tu dong mo tai http://localhost:8501
streamlit run app.py
echo.
pause
goto menu

:install
echo.
echo [INFO] Dang cai dat cac thu vien tu requirements.txt...
py -m pip install -r requirements.txt
echo.
pause
goto menu

:exit
exit
