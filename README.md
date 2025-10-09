3D Stellar Navigator
This Python application provides an interactive 3D visualization of our local stellar neighborhood using data from the European Space Agency's (ESA) Gaia mission. It allows you to navigate through thousands of stars, identify systems known to host exoplanets, and select individual stars to learn more about them.

This project uses PyOpenGL for high-performance 3D rendering and Pygame for window and input management.

Features
Real Scientific Data: Renders the 3D positions of the ~75,000 nearest stars from the Gaia DR3 catalog.

Exoplanet Host Identification: Fetches data from the NASA Exoplanet Archive and cross-matches it to visually highlight stars with known exoplanets.

Interactive Navigation: Fly through the starfield using intuitive mouse and keyboard controls.

Ray-Casting Selection: Click on any star to select it and view its information, such as its Gaia ID and distance from the Sun.

Smart Caching: Automatically caches the astronomical data after the first run for significantly faster startup times.

Requirements
Python 3.8 or newer

Anaconda or Miniconda

Setup Instructions
Follow these steps to set up a dedicated environment for this project, which prevents conflicts with your other Python projects.

1. Create a Conda Environment
Open the Anaconda Prompt (or your terminal) and run the following command to create a new environment named astro3d:

conda create --name astro3d python=3.9 -y

2. Activate the Environment
You must activate the environment each time you want to work on the project.

conda activate astro3d

Your terminal prompt should now show (astro3d) at the beginning.

3. Install Dependencies
You can install the required Python libraries in one of two ways:

Option A: Using the Batch Script (Windows Only)

If you are on Windows, simply double-click the install_dependencies.bat file. It will automatically install all necessary libraries into your active Conda environment.

Option B: Manual Installation (All Platforms)

Run the following command in your activated Conda terminal:

pip install pygame PyOpenGL numpy pandas astroquery tenacity pyarrow scipy requests

Running the Application
Once the setup is complete, you can run the navigator from your activated (astro3d) terminal.

Make sure you are in the same directory as the stellar_navigator_3d.py script.

Run the application with the following command:

python stellar_navigator_3d.py

Note: The very first time you run the script, it will take a minute or two to download the star and exoplanet data from the online archives. Subsequent launches will be much faster as the data will be loaded from local cache files (.feather).

Navigation Controls
Look Around: Click and drag the left mouse button.

Zoom: Use the mouse scroll wheel.

Pan Camera: Use the W, A, S, D keys.

Select Star: Left-click on a star.
