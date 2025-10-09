@echo off
echo =========================================
echo  Installing Python Dependencies for
echo      3D Stellar Navigator
echo =========================================
echo.
echo This script will install:
echo - pygame
echo - PyOpenGL
echo - numpy
echo - pandas
echo - astroquery
echo - tenacity
echo - pyarrow
echo - scipy
echo - requests
echo.

pip install pygame PyOpenGL numpy pandas astroquery tenacity pyarrow scipy requests

echo.
echo =========================================
echo  Installation complete.
echo =========================================
echo.
pause
