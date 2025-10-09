import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from scipy.spatial import cKDTree
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from io import StringIO
import requests

# --- Configuration & Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1600, 900
BACKGROUND_COLOR = (0.0, 0.0, 0.0, 1.0)
STAR_CACHE_FILE = "gaia_star_cache.feather"
EXOPLANET_CACHE_FILE = "exoplanet_cache_detailed.feather"
MAX_DISTANCE_PC = 50  # Maximum distance in parsecs to include stars
STAR_POINT_SIZE = 2.0
EXOPLANET_HOST_POINT_SIZE = 6.0
SELECTION_THRESHOLD = 0.5 # How close in Parsecs a click needs to be to select a star

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('3d_navigator.log')]
)

# --- Data Fetching and Processing ---

class DataProvider:
    """Handles fetching and caching of astronomical data."""

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_gaia_data(max_dist_pc):
        logging.info("Querying Gaia Archive... This may take a moment.")
        try:
            min_parallax = 1000 / max_dist_pc
            job = Gaia.launch_job_async(f"""
            SELECT TOP 75000 source_id, ra, dec, parallax, phot_g_mean_mag, bp_rp
            FROM gaiadr3.gaia_source
            WHERE parallax >= {min_parallax}
            ORDER BY phot_g_mean_mag ASC
            """)
            results = job.get_results()
            df = results.to_pandas()
            logging.info(f"Successfully fetched {len(df)} stars from Gaia.")
            df.to_feather(STAR_CACHE_FILE)
            return df
        except Exception as e:
            logging.error(f"Gaia query failed: {e}")
            raise

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_exoplanet_data():
        logging.info("Querying NASA Exoplanet Archive for detailed host star data...")
        try:
            url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            # Fetch host star name, distance, and coordinates for cross-matching
            params = {
                'query': "select hostname, sy_dist, ra, dec from pscomppars where default_flag = 1",
                'format': 'csv'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.to_feather(EXOPLANET_CACHE_FILE)
            logging.info(f"Fetched {len(df)} exoplanet host star records.")
            return df
        except Exception as e:
            logging.error(f"Exoplanet data fetch failed: {e}")
            raise

    @staticmethod
    def get_data():
        """Loads data, performs cross-matching between Gaia and Exoplanet hosts."""
        if os.path.exists(STAR_CACHE_FILE):
            star_data = pd.read_feather(STAR_CACHE_FILE)
        else:
            star_data = DataProvider.fetch_gaia_data(MAX_DISTANCE_PC)

        if os.path.exists(EXOPLANET_CACHE_FILE):
            exoplanet_data = pd.read_feather(EXOPLANET_CACHE_FILE)
        else:
            exoplanet_data = DataProvider.fetch_exoplanet_data()

        # --- Cross-match exoplanet hosts with Gaia data ---
        logging.info("Cross-matching exoplanet hosts with Gaia catalog...")
        
        # Calculate Cartesian coordinates for both datasets to find nearest neighbors
        star_data['x'], star_data['y'], star_data['z'] = DataProvider.spherical_to_cartesian(
            star_data['ra'], star_data['dec'], star_data['parallax'])
        
        # Exoplanet archive provides distance, not parallax. We can derive cartesian coords.
        exo_hosts_clean = exoplanet_data.dropna(subset=['ra', 'dec', 'sy_dist'])
        
        exo_coords = np.array([
            DataProvider.spherical_to_cartesian_dist(row.ra, row.dec, row.sy_dist)
            for row in exo_hosts_clean.itertuples()
        ])
        
        # Use a k-d tree for efficient nearest neighbor search
        star_tree = cKDTree(star_data[['x', 'y', 'z']].values)
        distances, indices = star_tree.query(exo_coords, k=1)
        
        # Mark Gaia stars that are exoplanet hosts (if they are close enough)
        star_data['is_exoplanet_host'] = False
        match_threshold = 0.5  # Parsecs
        matched_indices = indices[distances < match_threshold]
        star_data.loc[star_data.index[matched_indices], 'is_exoplanet_host'] = True
        
        logging.info(f"Matched {len(matched_indices)} exoplanet host stars.")
        return star_data

    @staticmethod
    def spherical_to_cartesian(ra, dec, parallax):
        parallax_arcsec = parallax / 1000.0
        dist_pc = 1.0 / parallax_arcsec
        return DataProvider.spherical_to_cartesian_dist(ra, dec, dist_pc)
        
    @staticmethod
    def spherical_to_cartesian_dist(ra, dec, dist_pc):
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
        x = dist_pc * np.cos(ra_rad) * np.cos(dec_rad)
        y = dist_pc * np.sin(ra_rad) * np.cos(dec_rad)
        z = dist_pc * np.sin(dec_rad)
        return x, y, z

    @staticmethod
    def get_star_color(bp_rp, is_host):
        if is_host:
            return 0.0, 1.0, 1.0  # Cyan for exoplanet hosts
        
        bp_rp = np.clip(bp_rp, -0.5, 4.0)
        norm_val = (bp_rp + 0.5) / 4.5
        r = np.clip(1.0, 1.0, 1.0)
        g = np.clip(1.0 - norm_val * 1.5, 0.0, 1.0)
        b = np.clip(1.0 - norm_val * 3.0, 0.0, 1.0)
        return r, g, b

# --- Main Application Class ---

class StellarNavigator3D:
    def __init__(self):
        pygame.init()
        self.display = (SCREEN_WIDTH, SCREEN_HEIGHT)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Stellar Navigator | Click to Select a Star")

        self.init_gl()

        self.camera_pos = np.array([0.0, 0.0, -5.0], dtype=np.float32)
        self.camera_rot = np.array([0.0, 0.0], dtype=np.float32)
        self.zoom_speed, self.move_speed, self.rotation_speed = 0.5, 0.1, 0.2
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        self.font = pygame.font.Font(None, 30)
        self.selected_star_info = None

        self.star_data = DataProvider.get_data()
        self.star_positions = self.star_data[['x', 'y', 'z']].values.astype(np.float32)
        self.prepare_star_buffers()

    def init_gl(self):
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

    def prepare_star_buffers(self):
        logging.info("Separating stars and preparing VBOs...")
        
        # Separate regular stars and exoplanet hosts
        hosts = self.star_data[self.star_data['is_exoplanet_host']]
        non_hosts = self.star_data[~self.star_data['is_exoplanet_host']]

        self.vbo_non_hosts_pos, self.vbo_non_hosts_color, self.num_non_hosts = self._create_vbos(non_hosts)
        self.vbo_hosts_pos, self.vbo_hosts_color, self.num_hosts = self._create_vbos(hosts)
        
        logging.info(f"Created VBOs for {self.num_non_hosts} stars and {self.num_hosts} exoplanet hosts.")

    def _create_vbos(self, df):
        if df.empty:
            return None, None, 0
        
        positions = df[['x', 'y', 'z']].values.astype(np.float32)
        colors = np.array([
            DataProvider.get_star_color(row.bp_rp, row.is_exoplanet_host)
            for row in df.itertuples()
        ], dtype=np.float32)
        
        vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)

        vbo_color = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_color)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        
        return vbo_pos, vbo_color, len(df)

    def _draw_vbo(self, vbo_pos, vbo_color, num_points):
        if not vbo_pos or not num_points:
            return

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_color)
        glColorPointer(3, GL_FLOAT, 0, None)
        
        glDrawArrays(GL_POINTS, 0, num_points)
        
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def draw_scene(self):
        # Draw non-host stars
        glPointSize(STAR_POINT_SIZE)
        self._draw_vbo(self.vbo_non_hosts_pos, self.vbo_non_hosts_color, self.num_non_hosts)

        # Draw exoplanet host stars
        glPointSize(EXOPLANET_HOST_POINT_SIZE)
        self._draw_vbo(self.vbo_hosts_pos, self.vbo_hosts_color, self.num_hosts)
        
        # Draw Sun
        glPointSize(10)
        glColor3f(1.0, 1.0, 0.0)
        glBegin(GL_POINTS); glVertex3f(0.0, 0.0, 0.0); glEnd()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                    self.select_star_at_pos(event.pos) # Ray-casting trigger
                elif event.button == 4: self.camera_pos[2] += self.zoom_speed
                elif event.button == 5: self.camera_pos[2] -= self.zoom_speed
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: self.mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and self.mouse_dragging:
                dx, dy = event.pos[0] - self.last_mouse_pos[0], event.pos[1] - self.last_mouse_pos[1]
                self.camera_rot[1] += dx * self.rotation_speed
                self.camera_rot[0] += dy * self.rotation_speed
                self.last_mouse_pos = event.pos

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: self.camera_pos[1] -= self.move_speed
        if keys[pygame.K_s]: self.camera_pos[1] += self.move_speed
        if keys[pygame.K_a]: self.camera_pos[0] += self.move_speed
        if keys[pygame.K_d]: self.camera_pos[0] -= self.move_speed
        return True

    def select_star_at_pos(self, mouse_pos):
        # --- Ray-Casting Implementation ---
        # 1. Un-project mouse coordinates to get a ray in 3D space
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        winX, winY = float(mouse_pos[0]), float(viewport[3] - mouse_pos[1])
        
        p_near = gluUnProject(winX, winY, 0.0, modelview, projection, viewport)
        p_far = gluUnProject(winX, winY, 1.0, modelview, projection, viewport)
        
        ray_origin = np.array(p_near)
        ray_dir = np.array(p_far) - ray_origin
        ray_dir /= np.linalg.norm(ray_dir)

        # 2. Find distance from each star to the ray
        vec_os = self.star_positions - ray_origin
        cross_prod = np.cross(vec_os, ray_dir)
        distances_to_ray = np.linalg.norm(cross_prod, axis=1)
        
        # 3. Find the closest star within the threshold
        min_dist_idx = np.argmin(distances_to_ray)
        if distances_to_ray[min_dist_idx] < SELECTION_THRESHOLD:
            selected = self.star_data.iloc[min_dist_idx]
            dist_pc = np.sqrt(selected.x**2 + selected.y**2 + selected.z**2)
            self.selected_star_info = (
                f"Gaia DR3 {selected.source_id}",
                f"Distance: {dist_pc:.2f} pc",
                "Hosts Exoplanets" if selected.is_exoplanet_host else "No Known Exoplanets"
            )
            logging.info(f"Selected star: {self.selected_star_info[0]}")
        else:
            self.selected_star_info = None

    def draw_hud(self):
        """Draws the 2D text overlay for selected star info."""
        if not self.selected_star_info:
            return

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)

        y_pos = SCREEN_HEIGHT - 35
        for line in self.selected_star_info:
            text_surface = self.font.render(line, True, (255, 255, 255, 255), (0, 0, 15, 150))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glWindowPos2d(10, y_pos)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            y_pos -= 30

        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()


    def run(self):
        running = True
        while running:
            running = self.handle_input()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            glTranslatef(*self.camera_pos)
            glRotatef(self.camera_rot[0], 1, 0, 0)
            glRotatef(self.camera_rot[1], 0, 1, 0)

            self.draw_scene()
            self.draw_hud()

            pygame.display.flip()
            pygame.time.wait(10)
        pygame.quit()

if __name__ == "__main__":
    app = StellarNavigator3D()
    app.run()

