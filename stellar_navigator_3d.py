import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from astroquery.skyview import SkyView
from astroquery.simbad import Simbad
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from io import StringIO, BytesIO
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threading
from PIL import Image
import time

# --- Scientific & Configuration Constants ---
H_PLANCK = 6.626e-34
C_LIGHT = 3e8
K_BOLTZMANN = 1.381e-23
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
STAR_CACHE_FILE = "exoplanet_hosts_cache.feather" # Renamed for clarity
EXOPLANET_CACHE_FILE = "exoplanet_complete_cache.feather"
SELECTION_THRESHOLD = 0.5
SYSTEM_VIEW_DISTANCE = 3.0  # Increased distance to trigger 3D system view
PLANET_LOD_DISTANCE = 0.05 # Distance to switch from billboard to 3D sphere
ORBIT_ANIMATION_SPEED = 0.1 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('3d_navigator.log')])

# --- Data Fetching and Processing ---
class DataProvider:
    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_exoplanet_data():
        logging.info("Querying NASA Exoplanet Archive for all confirmed exoplanet hosts...")
        try:
            url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            params = {'query': "select pl_name, hostname, sy_dist, ra, dec, pl_orbper, pl_orbsmax, pl_orbeccen, discoverymethod, disc_year from ps", 'format': 'csv'}
            response = requests.get(url, params=params)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.to_feather(EXOPLANET_CACHE_FILE)
            logging.info(f"Fetched {len(df)} exoplanet records.")
            return df
        except Exception as e: logging.error(f"Exoplanet data fetch failed: {e}"); raise

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_gaia_data_for_hosts(exoplanet_hosts_df):
        logging.info(f"Querying Gaia Archive for {len(exoplanet_hosts_df)} exoplanet host stars...")
        host_list = ", ".join([f"'{name}'" for name in exoplanet_hosts_df['hostname'].unique()])
        
        # This is a complex cross-match query. It joins Gaia's catalog with a list of names.
        # It's not perfect but is the best way to get Gaia data for a named list.
        query = f"""
        SELECT
            i.original_ext_source_id as hostname, g.source_id, g.ra, g.dec, g.parallax,
            g.phot_g_mean_mag, g.bp_rp, g.teff_gspphot, g.lum_gspphot, g.radius_gspphot
        FROM gaiadr3.gaia_source AS g
        JOIN gaiadr3.dr2_neighbourhood AS n ON g.source_id = n.dr3_source_id
        JOIN external.gaiadr2_astrometric_params AS i ON n.dr2_source_id = i.source_id
        WHERE i.original_ext_source_id IN ({host_list})
        """
        job = Gaia.launch_job_async(query)
        df = job.get_results().to_pandas()
        df.to_feather(STAR_CACHE_FILE)
        logging.info(f"Fetched Gaia data for {len(df)} host stars.")
        return df

    @staticmethod
    def get_data():
        if os.path.exists(EXOPLANET_CACHE_FILE): exoplanet_data = pd.read_feather(EXOPLANET_CACHE_FILE)
        else: exoplanet_data = DataProvider.fetch_exoplanet_data()
        
        # Filter for unique hosts to query Gaia
        unique_hosts = exoplanet_data.drop_duplicates(subset=['hostname'])

        if os.path.exists(STAR_CACHE_FILE):
            star_data = pd.read_feather(STAR_CACHE_FILE)
        else:
            star_data = DataProvider.fetch_gaia_data_for_hosts(unique_hosts)

        # Merge Gaia data back into the main star dataframe, keeping only hosts found in Gaia
        merged_data = pd.merge(star_data, unique_hosts[['hostname']], on='hostname', how='inner').drop_duplicates(subset=['hostname'])
        merged_data['x'], merged_data['y'], merged_data['z'] = DataProvider.spherical_to_cartesian(merged_data['ra'], merged_data['dec'], merged_data['parallax'])
        merged_data['is_exoplanet_host'] = True # All stars are hosts now
        
        logging.info(f"Final dataset contains {len(merged_data)} exoplanet host stars with Gaia data.")
        return merged_data, exoplanet_data

    @staticmethod
    def spherical_to_cartesian(ra, dec, parallax):
        parallax_arcsec = parallax / 1000.0
        dist_pc = np.where(parallax_arcsec > 0, 1.0 / parallax_arcsec, 1e9)
        return DataProvider.spherical_to_cartesian_dist(ra, dec, dist_pc)
        
    @staticmethod
    def spherical_to_cartesian_dist(ra, dec, dist_pc):
        ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)
        x = dist_pc*np.cos(ra_rad)*np.cos(dec_rad); y = dist_pc*np.sin(ra_rad)*np.cos(dec_rad); z = dist_pc*np.sin(dec_rad)
        return x, y, z

# --- Texture Management ---
class TextureManager:
    def __init__(self):
        self.textures = {}
        self._load_textures()

    def _load_textures(self):
        self.textures['planet'] = self._create_planet_texture((120, 180, 255))
        self.textures['gas_giant'] = self._create_planet_texture((210, 200, 170))
        # Add more textures here (e.g., from files)
        
    def _create_texture_from_surface(self, surf):
        tex_id = glGenTextures(1)
        tex_data = pygame.image.tostring(surf, "RGBA", True)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
        return tex_id

    def _create_planet_texture(self, color):
        surf = pygame.Surface((128, 128), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (64, 64), 64)
        # Add some simple shading
        for i in range(32):
            alpha = 100 - (i * 3)
            pygame.draw.circle(surf, (0, 0, 0, alpha), (75, 55), 50 - i)
        return self._create_texture_from_surface(surf)
    
    def get(self, name):
        return self.textures.get(name)

# --- UI and Visualization Panel (InfoPanel class remains largely the same) ---
class InfoPanel:
    def __init__(self, screen_width, screen_height, all_star_data):
        self.width, self.height = 450, screen_height
        self.x, self.y = screen_width - self.width, 0
        self.font_big = pygame.font.Font(None, 34); self.font_med = pygame.font.Font(None, 28); self.font_small = pygame.font.Font(None, 24)
        self.all_star_data = all_star_data
        self.tabs = ["Details", "Orbit Viewer", "HR Diagram", "Spectrum", "Sky View"]
        self.tab_rects = {}; self.active_tab = "Details"
        self.sky_view_thread = None; self.simbad_thread = None
        self.clear_selection()

    def clear_selection(self):
        self.selected_star = None; self.visuals = {}; self.planets_in_system = pd.DataFrame(); self.common_name = None

    def set_selected_star(self, star_series, exoplanet_data):
        if star_series is None: self.clear_selection(); return
        if self.selected_star is not None and self.selected_star['source_id'] == star_series['source_id']: return
        self.selected_star = star_series; self.active_tab = "Details"; self.visuals = {'Sky View': "Fetching Sky View..."}; self.common_name = "Fetching name..."
        self.planets_in_system = exoplanet_data[exoplanet_data['hostname'] == self.selected_star['hostname']].copy()
        self.sky_view_thread = threading.Thread(target=self.create_sky_view_image, args=(star_series,)); self.sky_view_thread.start()
        self.simbad_thread = threading.Thread(target=self.fetch_common_name, args=(star_series,)); self.simbad_thread.start()

    def fetch_common_name(self, star_series):
        try:
            coord = SkyCoord(ra=star_series['ra']*u.degree, dec=star_series['dec']*u.degree, frame='icrs')
            result_table = Simbad.query_region(coord, radius='2s')
            if result_table and len(result_table) > 0: self.common_name = result_table[0]['MAIN_ID']
            else: self.common_name = star_series['hostname']
        except Exception: self.common_name = star_series['hostname']

    def _get_visual_for_tab(self, tab_name):
        if tab_name not in self.visuals: self.generate_visual_for_tab(tab_name)
        return self.visuals.get(tab_name)

    def generate_visual_for_tab(self, tab_name):
        if self.selected_star is None: return
        if tab_name == "HR Diagram": self.visuals[tab_name] = self.create_hr_diagram()
        elif tab_name == "Spectrum": self.visuals[tab_name] = self.create_black_body_spectrum()
        elif tab_name == "Orbit Viewer": self.visuals[tab_name] = self.create_orbit_view()

    def _convert_mpl_to_pygame(self, fig):
        buf = BytesIO(); fig.savefig(buf, format="png", transparent=True, dpi=120); buf.seek(0)
        surface = pygame.image.load(buf).convert_alpha(); plt.close(fig); return surface

    def create_hr_diagram(self):
        fig, ax = plt.subplots(figsize=(4, 3), facecolor='#1E1E1E'); ax.set_facecolor("#0A0A0A")
        ax.scatter(self.all_star_data['teff_gspphot'], self.all_star_data['lum_gspphot'], alpha=0.3, s=15, c=self.all_star_data['bp_rp'], cmap='RdYlBu_r')
        star = self.selected_star
        if pd.notna(star['teff_gspphot']) and pd.notna(star['lum_gspphot']):
            ax.scatter(star['teff_gspphot'], star['lum_gspphot'], c='lime', s=80, edgecolor='white', zorder=5)
        ax.set_yscale('log'); ax.set_xscale('log'); ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_title("Hertzsprung-Russell Diagram", color='white', fontsize=10)
        ax.tick_params(axis='both', colors='gray', labelsize=7); fig.tight_layout(pad=0.5); return self._convert_mpl_to_pygame(fig)
        
    def create_black_body_spectrum(self):
        fig, ax = plt.subplots(figsize=(4, 3), facecolor='#1E1E1E'); ax.set_facecolor("#0A0A0A")
        T = self.selected_star['teff_gspphot']
        if pd.isna(T): ax.text(0.5, 0.5, "Temp. data unavailable", color='white', ha='center')
        else:
            wavelengths = np.linspace(100e-9, 2000e-9, 400)
            radiance = (2*H_PLANCK*C_LIGHT**2/wavelengths**5)/(np.exp(H_PLANCK*C_LIGHT/(wavelengths*K_BOLTZMANN*T))-1)
            ax.plot(wavelengths*1e9, radiance/np.max(radiance), color='orange'); ax.set_ylim(0, 1.1)
        ax.set_title(f"Stellar Spectrum (T≈{T:.0f} K)", color='white', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3); ax.tick_params(axis='both', colors='gray', labelsize=7); fig.tight_layout(pad=0.5); return self._convert_mpl_to_pygame(fig)
        
    def create_orbit_view(self):
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#1E1E1E'); ax.set_facecolor("#00001A")
        if self.planets_in_system.empty: ax.text(0.5, 0.5, "No planetary data", color='white', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.plot(0, 0, 'o', color='gold', markersize=15, label='Star'); max_a = 0
            for _, p in self.planets_in_system.iterrows():
                a=p['pl_orbsmax']; ecc=p['pl_orbeccen'] if pd.notna(p['pl_orbeccen']) else 0.0
                if pd.notna(a):
                    max_a=max(max_a, a); b=a*np.sqrt(1-ecc**2); theta=np.linspace(0,2*np.pi,360)
                    x_orb=a*(np.cos(theta)-ecc); y_orb=b*np.sin(theta); ax.plot(x_orb, y_orb, '--', alpha=0.7, label=p['pl_name'])
            ax.set_aspect('equal');
            if max_a>0: ax.set_xlim(-max_a*1.2, max_a*1.2); ax.set_ylim(-max_a*1.2, max_a*1.2)
            if len(self.planets_in_system)<6: ax.legend(fontsize=7, labelcolor='white', frameon=False)
        fig.tight_layout(pad=0.5); return self._convert_mpl_to_pygame(fig)

    def create_sky_view_image(self, star_series):
        try:
            ra, dec = star_series['ra'], star_series['dec']; surveys = ['Pan-STARRS DR2 i', 'DSS2 Red']
            for survey in surveys:
                try:
                    images = SkyView.get_images(position=f"{ra} {dec}", survey=survey, pixels=512)
                    if images:
                        data=np.sqrt(images[0][0].data); data=np.nan_to_num(data)
                        norm_data=(data-np.min(data))/(np.max(data)-np.min(data))*255
                        img=Image.fromarray(norm_data.astype(np.uint8)).convert('RGB')
                        self.visuals['Sky View']=pygame.image.fromstring(img.tobytes(), img.size, img.mode); return
                except Exception as e: logging.warning(f"Failed to fetch from {survey}: {e}")
            self.visuals['Sky View'] = "Image not found"
        except Exception: self.visuals['Sky View'] = "Error loading image"

    def handle_click(self, pos):
        if not self.selected_star: return
        for name, rect in self.tab_rects.items():
            if rect.collidepoint(pos): self.active_tab=name; self.generate_visual_for_tab(name); break
    
    def draw_details_tab(self, surface, y_offset):
        dist_pc = 1000.0 / self.selected_star['parallax']
        details = [("Gaia Source ID:", f"{self.selected_star['source_id']}"), ("Distance:", f"{dist_pc:.2f} pc"),
                   ("Effective Temp:", f"{self.selected_star['teff_gspphot']:.0f} K"), ("Luminosity:", f"{self.selected_star['lum_gspphot']:.3f} (Sun)"),
                   ("Radius:", f"{self.selected_star['radius_gspphot']:.3f} (Sun)")]
        for key, val in details:
            surface.blit(self.font_med.render(key, True, (200,200,200)), (self.x+15, y_offset)); surface.blit(self.font_med.render(val, True, (255,255,255)), (self.x+220, y_offset)); y_offset+=35
        if not self.planets_in_system.empty:
            pygame.draw.line(surface,(80,80,80),(self.x+10,y_offset),(self.x+self.width-10,y_offset),1); y_offset+=15
            surface.blit(self.font_med.render("Known Planets:", True,(255,255,255)),(self.x+15,y_offset)); y_offset+=30
            for _, p in self.planets_in_system.head(8).iterrows():
                p_text=f"- {p['pl_name']} ({p['discoverymethod']}, {p['disc_year']:.0f})"; surface.blit(self.font_small.render(p_text, True,(180,180,220)),(self.x+20,y_offset)); y_offset+=28

    def draw(self, surface):
        if self.selected_star is None: return
        panel_rect=pygame.Rect(self.x,self.y,self.width,self.height); pygame.draw.rect(surface,(30,30,30,230),panel_rect); pygame.draw.rect(surface,(80,80,80),panel_rect,2)
        y_offset=15; title=self.common_name or self.selected_star['hostname']
        surface.blit(self.font_big.render(title, True,(255,255,255)),(self.x+15,y_offset)); y_offset+=50
        pygame.draw.line(surface,(80,80,80),(self.x,y_offset),(self.x+self.width,y_offset),2)
        tab_y, tab_x = y_offset, self.x
        for name in self.tabs:
            surf = self.font_small.render(name, True,(255,255,255) if self.active_tab==name else (150,150,150))
            w,h=surf.get_size(); rect=pygame.Rect(tab_x,tab_y,w+20,h+10)
            if self.active_tab==name: pygame.draw.rect(surface,(50,50,50),rect); pygame.draw.line(surface,(0,150,255),(rect.left,rect.bottom-2),(rect.right,rect.bottom-2),2)
            surface.blit(surf,(tab_x+10,tab_y+5)); self.tab_rects[name]=rect; tab_x+=rect.width
        y_offset+=45; content_y=y_offset+20
        if self.active_tab == "Details": self.draw_details_tab(surface, content_y)
        else:
            visual = self._get_visual_for_tab(self.active_tab)
            if isinstance(visual,pygame.Surface): surface.blit(visual,(self.x+(self.width-visual.get_width())//2,content_y))
            elif isinstance(visual,str): status = self.font_med.render(visual,True,(200,200,200)); surface.blit(status,(self.x+(self.width-status.get_width())//2,content_y+150))

# --- Planetary System Renderer ---
class SystemRenderer:
    def __init__(self, texture_manager):
        self.texture_manager = texture_manager
        self.quadric = gluNewQuadric()
        gluQuadricTexture(self.quadric, GL_TRUE)

    def calculate_planet_position(self, planet_data, current_time):
        a=planet_data['pl_orbsmax'] if pd.notna(planet_data['pl_orbsmax']) else 1.0; ecc=planet_data['pl_orbeccen'] if pd.notna(planet_data['pl_orbeccen']) else 0.0
        period=planet_data['pl_orbper'] if pd.notna(planet_data['pl_orbper']) else 365.25
        if period<=0: return np.array([a*(1.0-ecc),0.0,0.0])
        mean_anomaly = (2*np.pi/period)*(current_time%period); eccentric_anomaly = mean_anomaly+ecc*np.sin(mean_anomaly)
        x=a*(np.cos(eccentric_anomaly)-ecc); y=a*np.sqrt(1-ecc**2)*np.sin(eccentric_anomaly)
        return np.array([x,y,0.0])*0.005 # Scale AU to parsecs

    def draw_billboard(self, position, size, texture_id):
        modelview=glGetFloatv(GL_MODELVIEW_MATRIX)
        up=np.array([modelview[0][1],modelview[1][1],modelview[2][1]])*size; right=np.array([modelview[0][0],modelview[1][0],modelview[2][0]])*size
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0,1); glVertex3fv(position-right-up); glTexCoord2f(1,1); glVertex3fv(position+right-up)
        glTexCoord2f(1,0); glVertex3fv(position+right+up); glTexCoord2f(0,0); glVertex3fv(position-right+up)
        glEnd()

    def draw_sphere(self, position, size, texture_id):
        glPushMatrix(); glTranslatef(*position)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        gluSphere(self.quadric, size, 32, 32); glPopMatrix()

    def draw_system(self, star, planets, current_time, cam_pos):
        star_pos = star[['x','y','z']].values.astype(np.float32)
        glEnable(GL_TEXTURE_2D); glDepthMask(GL_FALSE)
        
        # Draw Orbit Lines
        glDisable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, 0.2)
        for _, p in planets.iterrows():
            a=p['pl_orbsmax']; ecc=p['pl_orbeccen'] if pd.notna(p['pl_orbeccen']) else 0.0
            if pd.notna(a):
                glBegin(GL_LINE_LOOP)
                for i in range(361):
                    theta = np.deg2rad(i); r = (a * (1-ecc**2)) / (1 + ecc * np.cos(theta))
                    x = r * np.cos(theta); y = r * np.sin(theta)
                    glVertex3fv(star_pos + np.array([x,y,0.0])*0.005)
                glEnd()
        glEnable(GL_TEXTURE_2D)

        # Draw Planets (LOD)
        for _, planet in planets.iterrows():
            planet_pos_local = self.calculate_planet_position(planet, current_time)
            planet_pos_world = star_pos + planet_pos_local
            dist_to_cam = np.linalg.norm(planet_pos_world - cam_pos)
            
            tex_id = self.texture_manager.get('gas_giant') if planet['pl_orbsmax'] > 1.0 else self.texture_manager.get('planet')

            if dist_to_cam < PLANET_LOD_DISTANCE:
                self.draw_sphere(planet_pos_world, 0.0005, tex_id)
            else:
                self.draw_billboard(planet_pos_world, 0.01, tex_id)
        
        glDepthMask(GL_TRUE); glDisable(GL_TEXTURE_2D)

# --- Main Application Class ---
class StellarNavigator3D:
    def __init__(self):
        pygame.init(); pygame.font.init()
        self.display = (SCREEN_WIDTH, SCREEN_HEIGHT)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("3D Stellar Navigator | Click on a star to select")

        self.init_gl()
        self.camera_pos = np.array([0.0,0.0,-5.0], dtype=np.float32); self.camera_rot = np.array([0.0,0.0], dtype=np.float32)
        self.zoom_speed, self.move_speed, self.rotation_speed = 0.5, 0.1, 0.2
        self.mouse_dragging = False; self.last_mouse_pos = (0, 0)
        
        self.star_data, self.exoplanet_data = DataProvider.get_data()
        self.star_positions = self.star_data[['x','y','z']].values.astype(np.float32)
        self.prepare_star_buffers()
        
        self.texture_manager = TextureManager()
        self.info_panel = InfoPanel(SCREEN_WIDTH, SCREEN_HEIGHT, self.star_data)
        self.system_renderer = SystemRenderer(self.texture_manager)

    def init_gl(self):
        glMatrixMode(GL_PROJECTION); gluPerspective(45,(self.display[0]/self.display[1]),0.1,2000.0)
        glMatrixMode(GL_MODELVIEW); glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glEnable(GL_POINT_SMOOTH)

    def prepare_star_buffers(self):
        self.vbo_pos, self.vbo_color, self.num_stars = self._create_vbos(self.star_data)

    def _create_vbos(self, df):
        if df.empty: return None, None, 0
        positions = df[['x','y','z']].values.astype(np.float32)
        colors = np.array([self.get_star_color(row.bp_rp) for row in df.itertuples()], dtype=np.float32)
        vbo_pos=glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER,vbo_pos); glBufferData(GL_ARRAY_BUFFER,positions.nbytes,positions,GL_STATIC_DRAW)
        vbo_color=glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER,vbo_color); glBufferData(GL_ARRAY_BUFFER,colors.nbytes,colors,GL_STATIC_DRAW)
        return vbo_pos, vbo_color, len(df)
    
    @staticmethod
    def get_star_color(bp_rp):
        norm_val = (np.clip(bp_rp, -0.5, 4.0) + 0.5) / 4.5
        r=1.0; g=np.clip(1.0-norm_val*1.5,0.0,1.0); b=np.clip(1.0-norm_val*3.0,0.0,1.0)
        return r,g,b

    def _draw_vbo(self, vbo_pos, vbo_color, num_points):
        if not vbo_pos or not num_points: return
        glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER,vbo_pos); glVertexPointer(3,GL_FLOAT,0,None); glBindBuffer(GL_ARRAY_BUFFER,vbo_color); glColorPointer(3,GL_FLOAT,0,None)
        glDrawArrays(GL_POINTS,0,num_points)
        glDisableClientState(GL_COLOR_ARRAY); glDisableClientState(GL_VERTEX_ARRAY)

    def draw_scene(self):
        glPointSize(5.0); self._draw_vbo(self.vbo_pos, self.vbo_color, self.num_stars)
        glPointSize(10); glColor3f(1.0, 1.0, 0.0); glBegin(GL_POINTS); glVertex3f(0.0,0.0,0.0); glEnd()
        cam_world_pos = self.get_camera_world_position()
        for _, star in self.star_data.iterrows():
            star_pos = star[['x','y','z']].values.astype(np.float32)
            if np.linalg.norm(star_pos-cam_world_pos) < SYSTEM_VIEW_DISTANCE:
                planets = self.exoplanet_data[self.exoplanet_data['hostname']==star['hostname']]
                if not planets.empty: self.system_renderer.draw_system(star, planets, time.time()*ORBIT_ANIMATION_SPEED, cam_world_pos)

    def get_camera_world_position(self):
        modelview = glGetFloatv(GL_MODELVIEW_MATRIX)
        return np.linalg.inv(np.array(modelview))[3,:3]

    def handle_input(self):
        for event in pygame.event.get():
            if event.type==pygame.QUIT: return False
            if event.type==pygame.MOUSEBUTTONDOWN:
                if event.button==1: self.mouse_dragging=True; self.last_mouse_pos=event.pos
                    if event.pos[0]>=self.info_panel.x: self.info_panel.handle_click(event.pos)
                    else: self.select_star_at_pos(event.pos)
                elif event.button==4: self.camera_pos[2]+=self.zoom_speed
                elif event.button==5: self.camera_pos[2]-=self.zoom_speed
            elif event.type==pygame.MOUSEBUTTONUP and event.button==1: self.mouse_dragging=False
            elif event.type==pygame.MOUSEMOTION and self.mouse_dragging:
                dx,dy=event.pos[0]-self.last_mouse_pos[0],event.pos[1]-self.last_mouse_pos[1]
                self.camera_rot[1]+=dx*self.rotation_speed; self.camera_rot[0]+=dy*self.rotation_speed; self.last_mouse_pos=event.pos
        keys=pygame.key.get_pressed()
        if keys[pygame.K_w]: self.camera_pos[1]-=self.move_speed;
        if keys[pygame.K_s]: self.camera_pos[1]+=self.move_speed
        if keys[pygame.K_a]: self.camera_pos[0]+=self.move_speed;
        if keys[pygame.K_d]: self.camera_pos[0]-=self.move_speed
        return True

    def select_star_at_pos(self, mouse_pos):
        viewport=glGetIntegerv(GL_VIEWPORT); modelview=glGetDoublev(GL_MODELVIEW_MATRIX); projection=glGetDoublev(GL_PROJECTION_MATRIX)
        winX,winY=float(mouse_pos[0]),float(viewport[3]-mouse_pos[1])
        p_near=gluUnProject(winX,winY,0.0,modelview,projection,viewport); p_far=gluUnProject(winX,winY,1.0,modelview,projection,viewport)
        ray_origin,ray_dir=np.array(p_near),np.array(p_far)-np.array(p_near); ray_dir/=np.linalg.norm(ray_dir)
        distances_to_ray = np.linalg.norm(np.cross(self.star_positions-ray_origin,ray_dir),axis=1)
        min_dist_idx = np.argmin(distances_to_ray)
        if distances_to_ray[min_dist_idx] < SELECTION_THRESHOLD: self.info_panel.set_selected_star(self.star_data.iloc[min_dist_idx], self.exoplanet_data)
        else: self.info_panel.clear_selection()

    def draw_hud(self):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0,SCREEN_WIDTH,0,SCREEN_HEIGHT)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity(); glDisable(GL_DEPTH_TEST)
        self.info_panel.draw(pygame.display.get_surface())
        glEnable(GL_DEPTH_TEST); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW); glPopMatrix()

    def run(self):
        running=True
        while running:
            running = self.handle_input()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity(); glTranslatef(*self.camera_pos); glRotatef(self.camera_rot[0],1,0,0); glRotatef(self.camera_rot[1],0,1,0)
            self.draw_scene(); self.draw_hud()
            pygame.display.flip(); pygame.time.wait(10)
        pygame.quit()

if __name__ == "__main__":
    app = StellarNavigator3D()
    app.run()

