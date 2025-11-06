@echo off
echo ==========================================================
echo  2D Exoplanet Analyzer - Dependency Installer
echo ==========================================================
echo.
echo This script will install the libraries for the 2D (Tkinter) analyzer.
echo Please ensure you have activated your conda environment ('conda activate astro3d')
echo before running this script.
echo.
echo Installing libraries:
echo - pandas, numpy, requests, matplotlib
echo - scipy, tenacity, astroquery, astropy
echo.
echo Press any key to start the installation...
pause > nul

pip install pandas numpy requests matplotlib scipy tenacity astroquery astropy

echo.
echo ==========================================================
echo  Installation Complete!
echo ==========================================================
echo.
echo If there were no red error messages, you are ready to run the 2D analyzer.
echo You can now close this window.
echo.
pause
