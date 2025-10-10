@echo off
echo ==========================================================
echo  3D Stellar Navigator - Dependency Installer
echo ==========================================================
echo.
echo This script will install all the necessary Python libraries for the application.
echo Please ensure you have activated your conda environment ('conda activate astro3d')
echo before running this script.
echo.
echo Installing libraries:
echo - pygame
echo - PyOpenGL
echo - numpy
echo - pandas
echo - astroquery
echo - tenacity
echo - pyarrow
echo - astropy
echo - Pillow
echo.
echo Press any key to start the installation...
pause > nul

pip install pygame PyOpenGL numpy pandas astroquery tenacity pyarrow astropy Pillow

echo.
echo ==========================================================
echo  Installation Complete!
echo ==========================================================
echo.
echo If there were no red error messages, you are ready to run the program.
echo You can now close this window.
echo.
pause

