# Resurface

A Blender add-on by **Luci_xiv** that rebuilds messy meshes into clean, even triangles without losing thin parts like strings, straps and fringe. Decimated triangle soup, slivers, uneven density and non-manifold junctions all come out as a tidy surface. UVs, weights, shape keys and normals come along, and the result exports back to `.mdl`. The same sidebar tab has tools for normals, UV layouts, textures and FFXIV index maps.

![Rebuild Mesh on the Lotus Dress: one click, about 12 seconds, and every string is still there](docs/images/rebuild.gif)

This page is the overview. The [guide](docs/guide.md) covers every tool and setting in detail.

## Requirements

- **Blender 4.2 or newer.** It's tested with Blender 5.2 LTS.
- **Nothing else to install.** It only needs NumPy, which comes with Blender.
- **For FFXIV models,** an importer and exporter such as XIV Instant Edit, to get `.mdl` files in and out of Blender. The texture tools work on images Blender can open, so export `.tex` files as PNG from Penumbra or TexTools first.

## Installation

1. Download `resurface-<version>.zip` from this repository's Releases (or its `dist` folder).
2. In Blender, go to **Edit → Preferences → Get Extensions**, open the **⌄** menu at the top right and choose **Install from Disk…**. Pick the zip. You can also drag the zip into the Blender window.
3. In the 3D Viewport, press **N** and open the **Resurface** tab.

To update, install the new zip the same way.

Resurface used to be called *Mesh Rebuild* (up to version 1.4.1). Uninstall Mesh Rebuild before installing Resurface, or both tabs show up. Objects rebuilt with Mesh Rebuild can still be restored. Its sidebar settings aren't carried over.

## Features

The tab has three pages: **Mesh**, **Normals** and **UVs**. Each tool shows its last result in a box in its own panel.

![The Resurface tab: the Mesh page after Analyze, the Normals page, and the UVs page with all four tools](docs/images/sidebar.png)

### Rebuild Mesh

**[Rebuild Mesh](docs/guide.md#rebuild-mesh)** replaces the mesh of the selected objects with clean, evenly sized triangles. Voxel and quad remeshers lose anything thinner than a voxel or narrower than a quad. Resurface works on the original surface instead: borders, junctions between cloth layers and UV seams stay edges, and a string one triangle wide stays a string. Triangles get smaller where the surface folds tightly.

- UVs, vertex weights, shape keys, colors and normals are carried over. Each object keeps its name, modifiers, vertex groups and materials.
- It runs in the background, with progress in the status bar. **Esc** cancels.
- The original mesh is kept, and **Restore Original** swaps it back.
- **Analyze** reports what's wrong with a mesh. **Remove Hidden Layers** deletes under-layers that can never be seen, such as the second copy of the cloth under most of the Lotus Dress.
- Under **Afterwards**, Rebuild Mesh also runs **Smooth Normals** (on by default) and **Rebuild UVs** on the new mesh.

![The Lotus Dress before and after Rebuild Mesh: the side strings (top) and the chest (bottom)](docs/images/before-after.jpg)

### Smooth Normals

**[Smooth Normals](docs/guide.md#smooth-normals)** gives any meshes clean smooth normals. On layered game cloth, Blender's *Recalculate Outside* with *Set from Faces* leaves dark spots, seams and pinches where strings meet panels or faces are wound the other way. Smooth Normals never flips faces, so what you see from outside and from inside stays the same.

### UV tools

The UVs page has four tools:

- **[Rebuild UVs](docs/guide.md#rebuild-uvs)** makes a clean layout without overlaps and with even texel density. Seams follow the model's structure (the old islands, layer junctions, part borders) instead of every wrinkle, and each island keeps its old orientation. The previous UVs stay in a layer named **Old UVs**.
- **[Transfer Texture](docs/guide.md#transfer-texture)** moves a texture painted for the old UVs onto the new layout. It handles diffuse maps and masks, `_id` index maps (never blended) and normal maps (turned with each island).
- **[Resize Canvas](docs/guide.md#resize-canvas)** moves UVs after a texture's canvas was cropped or extended in an image editor (from 2048×4096 to 2048×2048, say), so every face keeps showing the same pixels.
- **[Index Map Generator](docs/guide.md#index-map-generator)** paints the `_id` texture for Dawntrail colorsets. It hands out the rows by UV island or by the colours of the base texture, or you assign them to faces by hand in Edit Mode.

## Good to know

- **Rebuild parts that touch together.** Select all of them (both halves of a dress, say), so their shared edges stay aligned.
- **For hair and hand-edited normals,** switch off *Afterwards: Smooth Normals*. The original normals are then carried over instead.
- **Keep Remove Hidden Layers off for see-through textures** such as lace or mesh fabric, because the layer below shows through the holes.
- **The original mesh stays in the .blend** (as *<mesh> (original)*) until you click **Discard Original** under *Output*. It isn't exported.
- **Ready for export:** the result is all triangles, which the MDL exporter needs. XIV Instant Edit ignores the **Old UVs** layer and the index map rows (`colorset_row`). Delete Old UVs once you no longer need it.
- **The UVs page replaces two older scripts,** *UV Scale to Dimensions* and *XIV Index Map Generator*. Disable them once you've switched. Rows made with the old generator can be converted with **Convert Old Row Groups**.
- **The globe icon** at the right end of the tab bar links to Luci_xiv's mods on XIV Mod Archive, the GitHub page, Bluesky and Ko-fi.

## Credits

- **Remeshing** follows *A Remeshing Approach to Multiresolution Modeling* by Mario Botsch and Leif Kobbelt (2004) and *Adaptive Remeshing for Real-Time Mesh Deformation* by Marion Dunyach et al. (2013), extended for non-manifold, layered game meshes.
- **Rebuild UVs** uses Blender's own *Minimum Stretch* unwrapper and island packing.
- **Screenshots:** Luci's Lotus Dress, a model swap of FINAL FANTASY XIV gear.

FINAL FANTASY XIV © SQUARE ENIX CO., LTD. Resurface is a fan-made tool, not affiliated with or endorsed by Square Enix.

## License

Resurface is licensed under the [GNU General Public License v3.0 or later](resurface/LICENSE). To build it from source or find your way around the code, see [For developers](docs/guide.md#for-developers) in the guide.

## Links

[XIV Mod Archive](https://www.xivmodarchive.com/user/124593) · [GitHub](https://github.com/link-0402/MagicFit) · [Bluesky](https://bsky.app/profile/xiv-luci.bsky.social) · [Ko-fi](https://ko-fi.com/luci_xiv)
