@echo off
echo ==========================================================
echo  2D Exoplanet Analyzer - Dependency Installer
echo ==========================================================
echo.
echo This script will install the necessary Python libraries for the 
echo 2D Research & Atmospheric Analyzer.
echo.
echo Please ensure you have activated your conda environment 
echo (e.g., 'conda activate astro3d') before running this script.
echo.
echo Installing libraries:
echo - pandas (Data handling)
echo - numpy (Mathematics)
echo - requests (NASA API connection)
echo - matplotlib (Scientific plotting)
echo - tenacity (Retry logic for stable downloads)
echo.
echo Press any key to start the installation...
pause > nul

pip install pandas numpy requests matplotlib tenacity

echo.
echo ==========================================================
echo  Installation Complete!
echo ==========================================================
echo.
echo If there were no red error messages, you are ready to run the 2D analyzer.
echo You can now close this window and run: python exoplanet_analyzer.py
echo.
pause
