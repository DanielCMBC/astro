3D Exoplanet System Navigator
This is an interactive 3D celestial map that allows you to explore all known star systems that host exoplanets. The application uses real astronomical data from the Gaia mission, the NASA Exoplanet Archive, and the SIMBAD database to provide a scientifically accurate and recognizable representation of these fascinating systems.

When you fly close to a star, you will see a real-time, animated 3D representation of its planetary system, complete with textured planets and orbit lines.

Features
Exoplanet-Focused Universe: The map exclusively displays stars confirmed to host exoplanets, allowing for a focused exploration of known planetary systems like 51 Eridani and TRAPPIST-1.

Common Star Names: Fetches recognizable star names from the SIMBAD database instead of just catalog numbers.

Animated 3D Planetary Systems: As you approach a host star, its planets appear as sprites and begin to orbit in real-time along 3D paths.

"Google Earth" Style Zoom: Fly even closer to a planet, and its 2D sprite will seamlessly transition into a detailed, textured 3D sphere.

Interactive Camera: Fly through the galaxy with intuitive mouse and keyboard controls (WASD, mouse drag, scroll wheel).

Ray-Cast Selection: Click on any star to select it and bring up a detailed information panel.

Fully Functional Data Panel:

Details: Shows key data like the star's common name, distance, temperature, luminosity, and a list of its known planets.

Orbit Viewer: Displays a top-down 2D plot of the planetary system's orbits.

HR Diagram: Plots the selected star on a Hertzsprung-Russell diagram of all other exoplanet hosts.

Spectrum: Shows the star's theoretical black-body radiation curve.

Sky View: Fetches and displays real astronomical images of the star from professional sky surveys (Pan-STARRS and DSS).

Data Caching: Fetched data is cached locally for much faster startup times on subsequent runs.

Multi-Threaded Data Fetching: Sky survey images and star names are loaded in the background to keep the UI responsive.

Setup and Installation
This project uses Python and several external libraries. The following steps will guide you through setting up a dedicated environment using Anaconda/Miniconda.

1. Prerequisites
Anaconda or Miniconda: You must have a working installation. You can download Miniconda here.

2. Create the Conda Environment
Open your terminal (Anaconda Prompt on Windows, or your default terminal on macOS/Linux) and run the following commands:

# Create a new conda environment named 'astro3d' with Python 3.9
conda create --name astro3d python=3.9 -y

# Activate the new environment
conda activate astro3d

3. Install Dependencies
With the astro3d environment active, run the following command in your terminal to install all required libraries:

pip install pygame PyOpenGL numpy pandas astroquery tenacity pyarrow astropy Pillow

4. Running the Application
Once the environment is set up and the dependencies are installed:

Make sure your astro3d conda environment is active.

Navigate to the project directory in your terminal.

Run the main Python script:

python stellar_navigator_3d.py

Note on First Run: The very first time you launch, it may take several minutes to download and cross-match the exoplanet and Gaia catalogs. This is a complex, one-time process. The application will create .feather cache files in the directory, and all subsequent launches will be much faster.

How to Use
Navigate:

Look Around: Click and drag the left mouse button.

Pan: Use the W, A, S, D keys.

Zoom: Use the mouse scroll wheel.

Select a Star: Left-click on any point of light.

View an Animated System: Fly close (within ~3 parsecs) to any star. The planets will automatically appear and begin to orbit. Zoom closer to an individual planet to see it transition into a 3D sphere.

Explore Data: The panel on the right will update with the selected star's data. Click the different tabs to explore all the available visualizations and information.
