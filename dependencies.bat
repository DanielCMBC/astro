@echo off
REM ==========================================================
REM  Exoplanet Scientific Suite - dependency installer
REM ==========================================================
REM
REM pyproject.toml is the authoritative dependency source
REM (roadmap section 3.10). This script is only a convenience
REM wrapper around it, so the two can no longer drift apart the
REM way the old hand-written pip line did - that one installed
REM tenacity, which the program never used, and omitted SciPy
REM and Astropy, which it needs.

echo ==========================================================
echo  Exoplanet Scientific Suite - dependency installer
echo ==========================================================
echo.
echo Installing the project and its scientific dependencies from
echo pyproject.toml (editable install).
echo.
echo Activate your environment first, for example:
echo     conda activate astrodata
echo.
pause

python -m pip install --upgrade pip
python -m pip install -e .

echo.
echo Core install complete.
echo.
echo Optional extras:
echo    pip install -e ".[render]"    modern OpenGL renderer (ModernGL, pygame)
echo    pip install -e ".[dynamics]"  optional N-body mode (REBOUND)
echo    pip install -e ".[ui]"        PySide6 workstation UI
echo    pip install -e ".[sync]"      astroquery catalogue synchronisation
echo    pip install -e ".[dev]"       test suite (pytest)
echo.
echo Run the 2D application with:
echo    python exoplanet_analyzer.py
echo.
pause
