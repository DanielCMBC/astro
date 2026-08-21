# Assets, materials and visual provenance

## An exoplanet texture is never a photograph

No exoplanet surface has been imaged. Every visual asset therefore declares
what it actually is (roadmap section 16), and the UI is required to show the
badge:

| `AssetType` | Badge shown to the user |
|---|---|
| `OBSERVED` | Observed imagery |
| `NASA_CONCEPT` | NASA artist concept - not a photograph |
| `SCIENTIFIC_PROCEDURAL` | Procedural visualisation - actual appearance unknown |
| `GENERIC_CLASS` | Generic class placeholder - not a depiction of this planet |

`AssetRecord` carries `asset_id`, `planet_name`, `source_url`, `creator`,
`credit`, `license`, `retrieved` and `sha256`. `AssetManifest.verify_all`
re-hashes the files, so a swapped asset is detected rather than silently
displayed under someone else's credit.

Everything `assets/procedural.py` generates is
`SCIENTIFIC_PROCEDURAL`, and `PlanetMaterial.basis` records exactly which
measured values produced the appearance ("radius 11.2 R_earth, T_eq 1400 K").

## Resource resolution

The old build bundled `tables/` and `atmospheric_signatures.json` but then
looked for them relative to the process working directory, so a frozen
executable launched from anywhere else could not find its own data.

`ResourceManager` resolves declared resources against, in order:

1. the `ASTRO_EXPLORER_DATA` environment variable;
2. an explicit project root;
3. the PyInstaller bundle directory `sys._MEIPASS`;
4. the installed package directory;
5. the repository root, for source checkouts;
6. the current working directory - last, because it is least reliable.

It never walks the filesystem looking for data. `DECLARED_RESOURCES` is the
single list of what the application needs, and `exoplanet_analyzer.spec`
generates its `datas` from it, so declaring a resource is enough to get it
packaged.

Writable state (the local snapshot, caches) goes to a per-platform user data
directory, never next to the executable.

## Star colours

Two colours are kept apart (roadmap section 4.10):

* `StarColor.scientific` - derived from effective temperature through the
  Planckian locus in CIE xy, converted to linear sRGB and gamma-encoded;
* `StarColor.stylized` - the same colour lifted towards white so a cool M
  dwarf remains visible against a dark background.

The second is a rendering choice, not a measurement, and `StarColor.caveat`
says so. The Sun comes out near-white, a 3000 K dwarf orange, a 20000 K star
blue - which is the expected behaviour of the Planckian locus.

## Material classes

There is no universal planet shader (roadmap section 15). A material class
is chosen from measured radius, equilibrium temperature and bulk density:

| Class | Program | Notes |
|---|---|---|
| `ROCKY` | `rocky` | Lambert + GGX, procedural terrain variation |
| `ICY` | `rocky` | low roughness, high albedo |
| `GAS_GIANT` | `gas_giant` | strong banding, turbulent clouds, limb haze |
| `HOT_GIANT` | `gas_giant` | weaker banding, sharper terminator, emissive term |
| `ULTRA_HOT_GIANT` | `gas_giant` | banding suppressed (cloud condensation fails above ~2000 K) |

`MaterialDefinition.with_values` rejects unknown uniform names, so a typo
fails loudly instead of silently doing nothing.

## The atmosphere shell

`atmosphere.frag` implements Rayleigh and Mie scattering with an explicit
optical depth and scale height. Crucially it takes `u_enabled`: when the
science layer has no evidence of an atmosphere it passes `false` and every
fragment is discarded, so a planet measured to be a bare rock is drawn as a
bare rock rather than wrapped in an invented blue haze.

## Shaders

All five programs are core-profile GLSL 330 and compile against a real
OpenGL 3.3 context. Tests assert that no shader uses `gl_ModelViewMatrix`,
`gl_ProjectionMatrix`, `varying`, `attribute` or `ftransform` - the
fixed-function constructs the prototype relied on.

| Program | Vertex | Fragment |
|---|---|---|
| `star` | `star.vert` | `star.frag` (limb darkening, granulation, HDR + tone map) |
| `rocky` | `planet.vert` | `rocky.frag` |
| `gas_giant` | `planet.vert` | `gas_giant.frag` |
| `atmosphere` | `planet_atmosphere.vert` | `atmosphere.frag` |
| `orbit` | `orbit.vert` | `orbit.frag` (dashes an assumed orbit) |

Planets and stars are drawn with **instanced** draws from a single
interleaved buffer, and orbits as one batched line strip each, because the
performance rule for Python + OpenGL is to avoid per-object draw overhead
(roadmap section 25).
