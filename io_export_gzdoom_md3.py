# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
#
# ***** END GPL LICENCE BLOCK *****
#
# Updates and additions for Blender 2.6X by Derek McPherson
# Updates for Blender 3.2.0 by Kevin Caccamo and Nash Muhandes
#
bl_info = {
        "name": "GZDoom .MD3",
        "author": "Derek McPherson, Xembie, PhaethonH, Bob Holcomb, Damien McGinnes, Robert (Tr3B) Beckebans, CoDEmanX, Mexicouger, Nash Muhandes, Kevin Caccamo",
        "version": (2, 0, 0),
        "blender": (3, 2, 0),
        "location": "File > Export > GZDoom model (.md3)",
        "description": "Export mesh to GZDoom model with vertex animation (.md3)",
        "warning": "",
        "wiki_url": "",
        "tracker_url": "https://forum.zdoom.org/viewtopic.php?f=232&t=69417",
        "category": "Import-Export"}

import bpy, struct, math, time
from bpy_extras.io_utils import (
    ExportHelper,
    axis_conversion,
    orientation_helper
)
from collections import namedtuple, OrderedDict
from itertools import starmap
from math import floor, log10, radians
from mathutils import Matrix, Vector
from os.path import basename, splitext
from struct import pack


MAX_QPATH = 64

MD3_IDENT = "IDP3"
MD3_VERSION = 15
MD3_MAX_TAGS = 16
MD3_MAX_SURFACES = 32
MD3_MAX_FRAMES = 1024
MD3_MAX_SHADERS = 256
MD3_MAX_VERTICES = 8192    #4096
MD3_MAX_TRIANGLES = 16384  #8192
MD3_XYZ_SCALE = 64.0



class MD3Vertex:
    binary_format = "<3hH"

    def __init__(self):
        self.xyz = [0, 0, 0]
        self.normal = 0

    @staticmethod
    def get_size():
        return struct.calcsize(MD3Vertex.binary_format)

    # copied from PhaethonH <phaethon@linux.ucla.edu> md3.py
    @staticmethod
    def decode(latlng):
        lat = (latlng >> 8) & 0xFF
        lng = (latlng) & 0xFF
        lat *= math.pi / 128
        lng *= math.pi / 128
        x = math.cos(lat) * math.sin(lng)
        y = math.sin(lat) * math.sin(lng)
        z =                 math.cos(lng)
        retval = [ x, y, z ]
        return retval

    # copied from PhaethonH <phaethon@linux.ucla.edu> md3.py
    @staticmethod
    def encode(normal, gzdoom=True):
        x = normal[0]
        y = normal[1]
        z = normal[2]
        # normalize
        l = math.sqrt((x*x) + (y*y) + (z*z))
        if l == 0:
            return 0
        x = x/l
        y = y/l
        z = z/l

        # Export for Quake 3 rather than GZDoom
        if not gzdoom:
            if (x == 0.0) and (y == 0.0):
                if z > 0.0:
                    return 0
                else:
                    return (128 << 8)

        lng = math.acos(z) * 255 / (2 * math.pi)
        lat = math.atan2(y, x) * 255 / (2 * math.pi)
        retval = ((int(lat) & 0xFF) << 8) | (int(lng) & 0xFF)
        return retval

    def save(self, file):
        data = struct.pack(self.binary_format, *self.xyz, self.normal)
        file.write(data)

class MD3TexCoord:
    binary_format = "<2f"

    def __init__(self):
        self.u = 0.0
        self.v = 0.0

    @staticmethod
    def get_size():
        return struct.calcsize(MD3TexCoord.binary_format)

    def save(self, file):
        uv_x = self.u
        uv_y = 1.0 - self.v
        data = struct.pack(self.binary_format, uv_x, uv_y)
        file.write(data)

class MD3Triangle:
    binary_format = "<3i"

    def __init__(self):
        self.indexes = [ 0, 0, 0 ]

    @staticmethod
    def get_size():
        return struct.calcsize(MD3Triangle.binary_format)

    def save(self, file):
        indexes = self.indexes[:]
        indexes[1:3] = reversed(indexes[1:3])  # Winding order fix
        data = struct.pack(self.binary_format, *indexes)
        file.write(data)

class MD3Shader:
    binary_format = "<%dsi" % MAX_QPATH

    @staticmethod
    def get_size():
        return struct.calcsize(MD3Shader.binary_format)

    @staticmethod
    def save(index, name, file):
        name = str.encode(name)
        data = struct.pack(MD3Shader.binary_format, name, index)
        file.write(data)

class MD3Surface:
    binary_format = "<4s%ds10i" % MAX_QPATH  # 1 int, name, then 10 ints

    def __init__(self):
        self.ident = MD3_IDENT
        self.name = ""
        self.flags = 0
        self.num_frames = 0
        self.num_verts = 0
        self.ofs_triangles = 0
        self.ofs_shaders = 0
        self.ofs_uv = 0
        self.ofs_verts = 0
        self.ofs_end = 0
        self.shaders = []
        self.triangles = []
        self.uv = []
        self.verts = []

        self.size = 0

    def get_size(self):
        if self.size > 0:
            return self.size
        # Triangles (after header)
        self.ofs_triangles = struct.calcsize(self.binary_format)

        # Shaders (after triangles)
        self.ofs_shaders = self.ofs_triangles + (
            MD3Triangle.get_size() * len(self.triangles))

        # UVs (after shaders)
        self.ofs_uv = self.ofs_shaders + (
            MD3Shader.get_size() * len(self.shaders))

        # Vertices for each frame (after UVs)
        self.ofs_verts = self.ofs_uv + (
            MD3TexCoord.get_size() * len(self.uv))

        # End (after vertices)
        self.ofs_end = self.ofs_verts + (
            MD3Vertex.get_size() * len(self.verts))

        self.size = self.ofs_end
        return self.ofs_end

    def save(self, file):
        self.get_size()
        temp_data = [0] * 12
        temp_data[0] = str.encode(self.ident)
        temp_data[1] = str.encode(self.name)
        temp_data[2] = self.flags
        temp_data[3] = self.num_frames
        temp_data[4] = len(self.shaders)  # self.num_shaders
        temp_data[5] = self.num_verts
        temp_data[6] = len(self.triangles)  # self.num_triangles
        temp_data[7] = self.ofs_triangles
        temp_data[8] = self.ofs_shaders
        temp_data[9] = self.ofs_uv
        temp_data[10] = self.ofs_verts
        temp_data[11] = self.ofs_end
        data = struct.pack(self.binary_format, *temp_data)
        file.write(data)

        # write the tri data
        for t in self.triangles:
            t.save(file)

        # save the shaders
        for index, shader in enumerate(self.shaders):
            MD3Shader.save(index, shader, file)

        # save the uv info
        for u in self.uv:
            u.save(file)

        # save the verts
        for v in self.verts:
            v.save(file)

class MD3Tag:
    binary_format="<%ds3f9f" % MAX_QPATH

    def __init__(self):
        self.name = ""
        self.origin = [0, 0, 0]
        self.axis = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    @staticmethod
    def get_size():
        return struct.calcsize(MD3Tag.binary_format)

    def save(self, file):
        temp_data = [0] * 13
        temp_data[0] = str.encode(self.name)
        temp_data[1] = float(self.origin[0])
        temp_data[2] = float(self.origin[1])
        temp_data[3] = float(self.origin[2])
        temp_data[4] = float(self.axis[0])
        temp_data[5] = float(self.axis[1])
        temp_data[6] = float(self.axis[2])
        temp_data[7] = float(self.axis[3])
        temp_data[8] = float(self.axis[4])
        temp_data[9] = float(self.axis[5])
        temp_data[10] = float(self.axis[6])
        temp_data[11] = float(self.axis[7])
        temp_data[12] = float(self.axis[8])
        data = struct.pack(self.binary_format, *temp_data)
        file.write(data)

class MD3Frame:
    binary_format="<3f3f3ff16s"

    def __init__(self):
        self.mins = [0, 0, 0]
        self.maxs = [0, 0, 0]
        self.local_origin = [0, 0, 0]
        self.radius = 0.0
        self.name = ""

    @staticmethod
    def get_size():
        return struct.calcsize(MD3Frame.binary_format)

    def save(self, file):
        temp_data = [0] * 11
        temp_data[0] = self.mins[0]
        temp_data[1] = self.mins[1]
        temp_data[2] = self.mins[2]
        temp_data[3] = self.maxs[0]
        temp_data[4] = self.maxs[1]
        temp_data[5] = self.maxs[2]
        temp_data[6] = self.local_origin[0]
        temp_data[7] = self.local_origin[1]
        temp_data[8] = self.local_origin[2]
        temp_data[9] = self.radius
        temp_data[10] = self.name.encode()
        data = struct.pack(self.binary_format, *temp_data)
        file.write(data)

class MD3Object:
    binary_format="<4si%ds9i" % MAX_QPATH  # little-endian (<), 17 integers (17i)

    def __init__(self):
        # header structure
        # this is used to identify the file (must be IDP3)
        self.ident = MD3_IDENT
        # the version number of the file (Must be 15)
        self.version = MD3_VERSION
        self.name = ""
        self.flags = 0
        self.num_tags = 0
        self.num_skins = 0
        self.ofs_frames = 0
        self.ofs_tags = 0
        self.ofs_surfaces = 0
        self.ofs_end = 0
        self.frames = []
        self.tags = []
        self.surfaces = []

        self.size = 0

    def get_size(self):
        if self.size > 0:
            return self.size
        # Frames (after header)
        self.ofs_frames = struct.calcsize(self.binary_format)

        # Tags (after frames)
        self.ofs_tags = self.ofs_frames + (
            len(self.frames) * MD3Frame.get_size())

        # Surfaces (after tags)
        self.ofs_surfaces = self.ofs_tags + (
            len(self.tags) * MD3Tag.get_size())
        # Surfaces' sizes can vary because they contain collections of
        # triangles, vertices, and UV coordinates
        self.ofs_end = self.ofs_surfaces + sum(
            map(lambda s: s.get_size(), self.surfaces))

        self.size = self.ofs_end
        return self.ofs_end

    def save(self, file):
        self.get_size()
        temp_data = [0] * 12
        temp_data[0] = str.encode(self.ident)
        temp_data[1] = self.version
        temp_data[2] = str.encode(self.name)
        temp_data[3] = self.flags
        temp_data[4] = len(self.frames)  # self.num_frames
        temp_data[5] = self.num_tags
        temp_data[6] = len(self.surfaces)  # self.num_surfaces
        temp_data[7] = self.num_skins
        temp_data[8] = self.ofs_frames
        temp_data[9] = self.ofs_tags
        temp_data[10] = self.ofs_surfaces
        temp_data[11] = self.ofs_end

        data = struct.pack(self.binary_format, *temp_data)
        file.write(data)

        for f in self.frames:
            f.save(file)

        for t in self.tags:
            t.save(file)

        for s in self.surfaces:
            s.save(file)


def message(log,msg):
    if log:
        log.write(msg + "\n")
    else:
        print(msg)


# Convert a XYZ vector to an integer vector for conversion to MD3
def convert_xyz(xyz):
    def convert(number, factor):
        return int(floor(number * factor))
    factors = [MD3_XYZ_SCALE] * len(xyz)
    position = map(convert, xyz, factors)
    return tuple(position)


# A class to help manage individual surfaces within a model
class BlenderSurface:
    def __init__(self, material, shaders=[]):
        self.material = material  # Blender material name -> Shader
        self.surface = MD3Surface()  # MD3 surface
        # Set names for surface and its material, both of which are named after
        # the material it uses
        self.surface.name = material
        if len(shaders) == 0:
            self.surface.shaders = [material]
        else:
            # TODO: Handle "Use material name"
            self.surface.shaders = shaders

        # {Mesh object: [(triangle index, vertex index, smooth), ...]}
        # Where "Mesh object" is the NAME of the object from which the mesh was
        # created, "triangle index" is the index of the triangle in
        # mesh.loop_triangles, "vertex index" is the index of the vertex on the 
        # face, and "smooth" is a boolean value which specifies which normal
        # should be used (False to use face normal, True to use split normal)
        self.vertex_refs = OrderedDict()

        # Vertices (position, normal, and UV) in MD3 binary format, mapped to
        # their indices
        self.unique_vertices = {}

    def get_size(self):
        return self.surface.get_size()


# A class to help manage a model, which consists of one or more objects which
# may be fused together into one model
class BlenderModelManager:
    def __init__(self, gzdoom, model_name, ref_frame=None, frame_name="MDLA",
                 scale=1, frame_time=0, depsgraph=None):
        self.md3 = MD3Object()
        self.md3.name = model_name
        self.material_surfaces = OrderedDict()
        self.mesh_objects = []
        self.tag_objects = []
        self.tag_renames = {}
        self.fix_transform = Matrix.Identity(4)
        self.lock_vertices = False
        self.start_frame = bpy.context.scene.frame_start
        self.end_frame = bpy.context.scene.frame_end
        self.frame_count = self.end_frame - self.start_frame + 1
        KeyFrameName = namedtuple("KeyFrameName", "frame name")
        self.keyframes = sorted(map(lambda m: KeyFrameName(m.frame, m.name),
                                    bpy.context.scene.timeline_markers),
                                key=lambda m: m.frame)
        keyframes = list(map(lambda f: f.frame, self.keyframes))
        keyframes.append(bpy.context.scene.frame_end)
        if len(keyframes) > 1:
            self.frame_digits = floor(log10(max(starmap(
                lambda a, b: b - a, zip(keyframes, keyframes[1:])))))
        else:
            if self.frame_count - 1 == 0:
                self.frame_digits = 1
            else:
                self.frame_digits = floor(log10(self.frame_count - 1))
        self.gzdoom = gzdoom
        # Reference frame - used for initial UV and triangle data
        if ref_frame is not None:
            self.ref_frame = ref_frame
        else:
            self.ref_frame = self.start_frame
        self.frame_name = frame_name[0:4]
        if frame_time == 0:
            frame_time = 35 / bpy.context.scene.render.fps
        self.frame_time = floor(max(frame_time, 1))
        self.scale = scale
        self.name = model_name
        self.depsgraph = depsgraph

    def save(self, filename, modeldef=False, actordef=False):
        from bpy.path import basename
        from os.path import dirname, join
        nfile = open(filename, "wb")
        self.md3.save(nfile)
        nfile.close()
        base_path = dirname(filename)
        if modeldef:
            modeldef_text = self.get_modeldef(basename(filename))
            modeldef_path = join(base_path, "modeldef." + self.name + ".txt")
            modeldef_file = open(modeldef_path, "w")
            modeldef_file.write(modeldef_text)
            modeldef_file.close()
        if actordef:
            actordef_text = self.get_zscript()
            actordef_path = join(base_path, "zscript." + self.name + ".txt")
            actordef_file = open(actordef_path, "w")
            actordef_file.write(actordef_text)
            actordef_file.close()

    @staticmethod
    def encode_vertex(position, normal, uv, gzdoom):
        md3_position = convert_xyz(position)
        md3_normal = MD3Vertex.encode(normal, gzdoom)
        return (pack(MD3Vertex.binary_format, *md3_position, md3_normal)
              + pack(MD3TexCoord.binary_format, *uv))

    def add_mesh(self, mesh_obj):
        """
        Add a mesh to the object. Does nothing if the animation frames have
        been added.
        """
        if self.lock_vertices:
            return False
        if mesh_obj in self.mesh_objects:
            return False
        self.mesh_objects.append(mesh_obj)
        bpy.context.scene.frame_set(self.ref_frame)
        mesh_obj = mesh_obj.evaluated_get(self.depsgraph)
        obj_mesh = mesh_obj.to_mesh(depsgraph=self.depsgraph)
        obj_mesh.transform(self.fix_transform @ mesh_obj.matrix_world)
        # calc_normals_split recalculates normals, even on meshes without
        # custom normals. If I didn't do this, the vertex normals would be all
        # wrong.
        obj_mesh.calc_normals_split()
        obj_mesh.calc_loop_triangles()
        # See what materials the mesh references, and add new surfaces for
        # those materials if necessary
        for face_index, face in enumerate(obj_mesh.loop_triangles):
            # Prefer using the md3shader property of the material. Use the
            # md3shader object property if the material does not have the
            # md3shader property, and use the material name if neither are
            # available.
            face_mtl = obj_mesh.materials[face.material_index].get("md3shader")
            if face_mtl is None:
                face_mtl = mesh_obj.get("md3shader")
            if face_mtl is None:
                face_mtl = obj_mesh.materials[face.material_index].name
            # Add the new surface to material_surfaces if it isn't already in
            if face_mtl not in self.material_surfaces:
                bsurface = BlenderSurface(face_mtl)
                bsurface.surface.num_frames = self.frame_count
                self.material_surfaces[face_mtl] = bsurface
                self.md3.surfaces.append(bsurface.surface)
            bsurface = self.material_surfaces[face_mtl]
            # Add the faces to the surface
            self._add_tri(
                bsurface, obj_mesh, mesh_obj.name, face_index, face.vertices)
        mesh_obj.to_mesh_clear()
        return True

    def _add_tri(self, bsurface, obj_mesh, obj_name, face_index,
                 mesh_vertex_indices):
        # Define VertexReference named tuple
        from collections import namedtuple
        VertexReference = namedtuple(
            "VertexReference",
            "tri_index tri_vertex_index smooth"
        )
        ntri = MD3Triangle()
        tri_index = face_index
        for tri_vertex_index in range(3):
            face = obj_mesh.loop_triangles[face_index]
            # Set up the new triangle
            vertex_index = face.vertices[tri_vertex_index]
            # Get vertex ID, which is the vertex in MD3 binary format. First,
            # get the vertex position
            vertex = obj_mesh.vertices[vertex_index]
            vertex_position = vertex.co
            # If the face is flat-shaded, the face normal is used.
            # Otherwise, the vertex normal is used.
            if face.use_smooth:
                vertex_normal = face.split_normals[tri_vertex_index]
            else:
                vertex_normal = face.normal
            # Get UV coordinates for this vertex.
            loop_index = face.loops[tri_vertex_index]
            face_uvs = obj_mesh.uv_layers.active.data[loop_index]
            vertex_uv = face_uvs.uv
            # Get ID from position, normal, and UV.
            vertex_id = BlenderModelManager.encode_vertex(
                vertex_position, vertex_normal, vertex_uv, self.gzdoom)
            # Add the vertex if it hasn't already been added.
            if vertex_id not in bsurface.unique_vertices:
                # num_verts is used because the surface contains vertex data
                # for every frame.
                bsurface.surface.num_verts += 1
                # Texture coordinates do not change per frame, so they can be
                # added now.
                ntexcoord = MD3TexCoord()
                ntexcoord.u = vertex_uv[0]
                ntexcoord.v = vertex_uv[1]
                bsurface.surface.uv.append(ntexcoord)
                # Map "Vertex ID" to the MD3 vertex index
                md3_vertex_index = len(bsurface.unique_vertices)
                bsurface.unique_vertices[vertex_id] = md3_vertex_index
                vert_refs = bsurface.vertex_refs.setdefault(obj_name, [])
                vert_refs.append(VertexReference(
                    tri_index, tri_vertex_index, face.use_smooth
                ))
            else:
                # The vertex has already been added, so just get its index.
                md3_vertex_index = bsurface.unique_vertices[vertex_id]
            # Set the vertex index on the triangle.
            ntri.indexes[tri_vertex_index] = md3_vertex_index
        bsurface.surface.triangles.append(ntri)

    def setup_frames(self):
        ObjectReference = namedtuple("ObjectReference", "object mesh")
        # Add the vertex animations for each frame. Only call this AFTER
        # all the triangle and UV data has been set up.
        self.lock_vertices = True
        self.md3.num_tags = (
            len(self.tag_objects) if self.frame_count > 0 else 0)
        for frame in range(self.start_frame, self.end_frame + 1):
            bpy.context.scene.frame_set(frame)
            obj_refs = {}
            nframe = MD3Frame()
            last_keyframe = next(filter(
                lambda f: f.frame <= frame, reversed(self.keyframes)), None)
            frame_num = frame - (
                last_keyframe.frame
                if last_keyframe is not None
                else self.start_frame)
            frame_suffix = (("{:0" + str(self.frame_digits) + "d}")
                             .format(frame_num))
            nframe.name = (
                "frame" + frame_suffix if last_keyframe is None
                else last_keyframe.name + frame_suffix)
            if len(self.mesh_objects) == 0:
                nframe.local_origin = Vector((0, 0, 0))
                nframe.mins = Vector((0, 0, 0))
                nframe.maxs = Vector((0, 0, 0))
                self.md3.frames.append(nframe)
                continue
            if bpy.context.active_object in self.mesh_objects:
                nframe.local_origin = (
                    bpy.context.active_object.location.copy())
            elif len(self.mesh_objects) > 0:
                nframe.local_origin = self.mesh_objects[0].location.copy()
            nframe_bounds_set = False
            for mesh_obj in self.mesh_objects:
                mesh_obj = mesh_obj.evaluated_get(self.depsgraph)
                obj_mesh = mesh_obj.to_mesh(depsgraph=self.depsgraph)
                # Set up obj_mesh
                obj_mesh.transform(self.fix_transform @ mesh_obj.matrix_world)
                obj_mesh.calc_normals_split()
                obj_mesh.calc_loop_triangles()
                # Set up frame bounds/origin/radius
                if not nframe_bounds_set:
                    # Copy the vertex so that it isn't modified
                    nframe.mins = obj_mesh.vertices[0].co.copy()
                    nframe.maxs = obj_mesh.vertices[0].co.copy()
                    nframe_bounds_set = True
                    armature = mesh_obj.find_armature()
                    if armature:
                        nframe.local_origin -= armature.location
                for vertex in obj_mesh.vertices:
                    # Check each coordinate individually so that the mins/maxs
                    # form a bounding box around the geometry
                    # First, check mins
                    nframe.mins = Vector(map(min, nframe.mins, vertex.co))
                    # Check maxs
                    nframe.maxs = Vector(map(max, nframe.maxs, vertex.co))
                nframe.radius = max(nframe.mins.length, nframe.maxs.length)
                # Add mesh to dict
                obj_refs[mesh_obj.name] = ObjectReference(mesh_obj, obj_mesh)
            self.md3.frames.append(nframe)
            for bsurface in self.material_surfaces.values():
                for mesh_name, vertex_infos in bsurface.vertex_refs.items():
                    obj_mesh = obj_refs[mesh_name].mesh
                    for vertex_info in vertex_infos:
                        tri = obj_mesh.loop_triangles[vertex_info.tri_index]
                        # Get vertex position
                        vertex_position = obj_mesh.vertices[
                            tri.vertices[vertex_info.tri_vertex_index]].co
                        # Get vertex normal
                        if vertex_info.smooth:
                            vertex_normal = tri.split_normals[
                                vertex_info.tri_vertex_index]
                        else:
                            vertex_normal = tri.normal
                        # Set up MD3 vertex
                        nvertex = MD3Vertex()
                        nvertex.xyz = convert_xyz(vertex_position)
                        nvertex.normal = MD3Vertex.encode(
                            vertex_normal, self.gzdoom)
                        bsurface.surface.verts.append(nvertex)
            for obj_ref in obj_refs.values():
                obj_ref.object.to_mesh_clear()
            for tag_obj in self.tag_objects:
                tag_transform = self.fix_transform @ tag_obj.matrix_world
                position, rotation, _ = tag_transform.decompose()
                # Makes the 'X' axis point forwards, assuming the tag is an
                # empty, shown as 'Arrows' with the 'Y' pointing forwards.
                local_transfix = Matrix.Rotation(radians(90), 3, 'Z')
                rotation = rotation.to_matrix() @ local_transfix
                # MD3 tag axes are column-major, and
                # Blender matrix axes are row-major
                rotation.transpose()
                ntag = MD3Tag()
                ntag.name = tag_obj.name
                ntag.origin = position
                ntag.axis[0:3] = rotation[0]
                ntag.axis[3:6] = rotation[1]
                ntag.axis[6:9] = rotation[2]
                self.md3.tags.append(ntag)

    def add_tag(self, bobject, strip_suffix=False):
        if bobject in self.tag_objects:
            return None
        if strip_suffix:
            suffix_index = bobject.name.rfind(".")
            if suffix_index >= 0:
                suffix_text = bobject.name[suffix_index+1:]
                if suffix_text.isdecimal():
                    desuffixed = bobject.name[:suffix_index]
                    self.tag_renames[bobject.name] = desuffixed
        self.tag_objects.append(bobject)

    def get_modeldef(self, md3fname):
        model_def = """Model {actor_name}
{{
    Model 0 "{file_name}"
    Scale {scale:.6f} {scale:.6f} {zscale:.6f}
    USEACTORPITCH
    USEACTORROLL

    {frames}
}}"""
        sprite_coder = BaseCoder(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_")
        frame_coder = BaseCoder(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]")
        scale = 1
        if 0 < self.scale < 1:  # Upscale to normal size with MODELDEF
            scale = 1 / self.scale
        zscale = scale * 1.2  # Account for GZDoom's vertical squishing
        frame_def = (
            "FrameIndex {frame_name} {frame_letter} 0 {frame_number}")
        modeldef_frames = []
        frame_base_sprite = bytearray(self.frame_name, "ascii")
        while len(frame_base_sprite) < 4:
            frame_base_sprite.append(65) # ord("A")
        for frame in range(self.frame_count):
            frame_letter = chr(frame_coder.alphabet[frame % frame_coder.base])
            frame_add = frame // frame_coder.base
            frame_num = sprite_coder.encode(frame_base_sprite) + frame_add
            frame_sprite = sprite_coder.decode(frame_num, 4)
            modeldef_frames.append(frame_def.format(
                frame_name=frame_sprite.decode(), frame_letter=frame_letter,
                frame_number=frame))
        return model_def.format(
            actor_name=self.name, file_name=md3fname, scale=scale,
            zscale=zscale, frames="\n    ".join(modeldef_frames))

    def get_zscript(self):
        actor_def = """class {actor_name} : Actor
{{
    States
    {{
    Spawn:
        {frames}
        Stop;
    }}
}}"""
        sprite_coder = BaseCoder(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_")
        frame_coder = BaseCoder(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]")
        frame_def = "{frame_name} {frame_letter} {tics};"
        frame_base_sprite = bytearray(self.frame_name, "ascii")
        while len(frame_base_sprite) < 4:
            frame_base_sprite.append(65) # ord("A")
        frames = []
        for frame in range(self.frame_count):
            frame_letter = chr(frame_coder.alphabet[frame % frame_coder.base])
            frame_add = frame // frame_coder.base
            frame_num = sprite_coder.encode(frame_base_sprite) + frame_add
            frame_sprite = sprite_coder.decode(frame_num, 4)
            frames.append(frame_def.format(
                frame_name=frame_sprite.decode(), frame_letter=frame_letter,
                tics=self.frame_time))
        return actor_def.format(
            actor_name=self.name,
            frames="\n        ".join(frames))


class BaseCoder:
    # Class that is useful for encoding and decoding base26 numbers. Such
    # numbers are used as sprite names for the Doom engine.

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.base = len(alphabet)

    def encode(self, text):
        "Encode a number with an arbitrary base. Takes a bytes-like object, "
        "returns the number."
        number = 0
        for index, char in enumerate(reversed(text)):
            char_value = self.alphabet.index(char)
            number += (self.base ** index) * char_value
        return number

    def decode(self, number, minlength=1):
        "Decode a number with an arbitrary base. Takes a number, returns the "
        "text."
        from math import log
        try:
            digits = floor(log(number, self.base)) + 1
        except ValueError:
            digits = 1
        length = max(minlength, digits)
        text = bytearray(length)
        for index in range(length):
            text[index] = self.alphabet[0]
        for index, byteindex in enumerate(reversed(range(length))):
            alphabet_index = floor(number / self.base ** index) % self.base
            text[byteindex] = self.alphabet[alphabet_index]
        return text


def print_md3(log,md3,dumpall):
    message(log,"Header Information")
    message(log,"Ident: " + str(md3.ident))
    message(log,"Version: " + str(md3.version))
    message(log,"Name: " + md3.name)
    message(log,"Flags: " + str(md3.flags))
    message(log,"Number of Frames: " + str(len(md3.frames)))
    message(log,"Number of Tags: " + str(md3.num_tags))
    message(log,"Number of Surfaces: " + str(len(md3.surfaces)))
    message(log,"Number of Skins: " + str(md3.num_skins))
    message(log,"Offset Frames: " + str(md3.ofs_frames))
    message(log,"Offset Tags: " + str(md3.ofs_tags))
    message(log,"Offset Surfaces: " + str(md3.ofs_surfaces))
    message(log,"Offset end: " + str(md3.ofs_end))

    def vec3_to_string(vector, integer=False):
        if integer is True:
            template = "{:< 5d} "
        else:
            template = "{:.4f} "
        return (template * 3).format(*vector)

    if dumpall:
        message(log,"Frames:")
        for f in md3.frames:
            message(log," Mins: " + vec3_to_string(f.mins))
            message(log," Maxs: " + vec3_to_string(f.maxs))
            message(log," Origin(local): " + vec3_to_string(f.local_origin))
            message(log," Radius: " + str(f.radius))
            message(log," Name: " + f.name)

        tag_frame = 1
        tag_index = md3.num_tags
        for t in md3.tags:
            if tag_index == md3.num_tags and tag_frame <= len(md3.frames):
                message(log,"Tags (Frame " + str(tag_frame) + "):")
                tag_index = 0
                tag_frame += 1
            message(log," Name: " + t.name)
            message(log," Origin: " + vec3_to_string(t.origin))
            message(log," Axis[0]: " + vec3_to_string(t.axis[0:3]))
            message(log," Axis[1]: " + vec3_to_string(t.axis[3:6]))
            message(log," Axis[2]: " + vec3_to_string(t.axis[6:9]))
            tag_index += 1

        message(log,"Surfaces:")
        for s in md3.surfaces:
            message(log," Ident: " + s.ident)
            message(log," Name: " + s.name)
            message(log," Flags: " + str(s.flags))
            message(log," # of Frames: " + str(s.num_frames))
            message(log," # of Shaders: " + str(len(s.shaders)))
            message(log," # of Verts: " + str(s.num_verts))
            message(log," # of Triangles: " + str(len(s.triangles)))
            message(log," Offset Triangles: " + str(s.ofs_triangles))
            message(log," Offset UVs: " + str(s.ofs_uv))
            message(log," Offset Verts: " + str(s.ofs_verts))
            message(log," Offset End: " + str(s.ofs_end))
            message(log," Shaders:")
            for index, shader in enumerate(s.shaders):
                message(log,"  \"{}\" ({})".format(shader, index))
            message(log," Triangles:")
            for tri in s.triangles:
                message(log,"  Indexes: " + vec3_to_string(tri.indexes, True))
            message(log," UVs:")
            for uv in s.uv:
                message(log,"  U: " + str(uv.u))
                message(log,"  V: " + str(uv.v))
            message(log," Verts:")
            for vert in s.verts:
                message(log,"  XYZ: " + vec3_to_string(vert.xyz, True))
                message(log,"  Normal: " + str(vert.normal))

    shader_count = 0
    vert_count = 0
    tri_count = 0
    for surface in md3.surfaces:
        shader_count += len(surface.shaders)
        tri_count += len(surface.triangles)
        vert_count += surface.num_verts
        if len(surface.shaders) >= MD3_MAX_SHADERS:
            message(log,"!Warning: Shader limit (" + str(len(surface.shaders)) + "/" + str(MD3_MAX_SHADERS) + ") reached for surface " + surface.name)
        if surface.num_verts >= MD3_MAX_VERTICES:
            message(log,"!Warning: Vertex limit (" + str(surface.num_verts) + "/" + str(MD3_MAX_VERTICES) + ") reached for surface " + surface.name)
        if len(surface.triangles) >= MD3_MAX_TRIANGLES:
            message(log,"!Warning: Triangle limit (" + str(len(surface.triangles)) + "/" + str(MD3_MAX_TRIANGLES) + ") reached for surface " + surface.name)

    if md3.num_tags >= MD3_MAX_TAGS:
        message(log,"!Warning: Tag limit (" + str(len(md3.tags)) + "/" + str(MD3_MAX_TAGS) + ") reached for md3!")
    if len(md3.surfaces) >= MD3_MAX_SURFACES:
        message(log,"!Warning: Surface limit (" + str(len(md3.surfaces)) + "/" + str(MD3_MAX_SURFACES) + ") reached for md3!")
    if len(md3.frames) >= MD3_MAX_FRAMES:
        message(log,"!Warning: Frame limit (" + str(len(md3.frames)) + "/" + str(MD3_MAX_FRAMES) + ") reached for md3!")

    message(log,"Total Shaders: " + str(shader_count))
    message(log,"Total Triangles: " + str(tri_count))
    message(log,"Total Vertices: " + str(vert_count))


# Main function
def save_md3(
        filepath, log_type, depsgraph, md3name="", dump_all=False, ref_frame=-1,
        orig_frame=None, gzdoom=True, sprite_name=False, sprite_tics=1,
        offsetx=0, offsety=0, offsetz=0, scale=1, gen_actordef=False,
        gen_modeldef=False, axis_forward='Y', axis_up='Z'):
    starttime = time.perf_counter()  # start timer
    fullpath = splitext(filepath)[0]
    modelname = basename(fullpath)
    logname = modelname + ".log"
    logfpath = fullpath + ".log"
    if md3name == "":
        md3name = modelname
    if log_type == "append":
        log = open(logfpath,"a")
    elif log_type == "overwrite":
        log = open(logfpath,"w")
    elif log_type == "blender":
        log = bpy.data.texts.new(logname)
        log.clear()
    else:
        log = None
    if ref_frame is None:
        ref_frame = orig_frame
    message(log, "###################### BEGIN ######################")
    model = BlenderModelManager(
        gzdoom, md3name, ref_frame, sprite_name, scale, sprite_tics, depsgraph)
    # Set up "fix" transformation matrix
    scale_fix = Matrix.Scale(scale, 4)
    pos_fix = Matrix.Translation((offsetx, offsety, offsetz))
    rot_fix = axis_conversion(axis_forward, axis_up, to_forward='X').to_4x4()
    # @ operator is used for matrix multiplication
    model.fix_transform = model.fix_transform @ scale_fix @ pos_fix @ rot_fix
    # Add objects to model manager
    if len(bpy.context.selected_objects) == 0:
        message(log, "Select an object to export!")
    else:
        # If multiple objects are selected, they are joined together
        for bobject in bpy.context.selected_objects:
            if bobject.type == 'MESH':
                model.add_mesh(bobject)
            elif bobject.type == 'EMPTY':
                model.add_tag(bobject)
        if (bpy.context.active_object and
            bpy.context.active_object not in bpy.context.selected_objects):
            if bpy.context.active_object.type == 'MESH':
                model.add_mesh(bpy.context.active_object)
            elif bpy.context.active_object.type == 'EMPTY':
                model.add_tag(bpy.context.active_object)
    model.setup_frames()
    model.md3.get_size()
    print_md3(log, model.md3, dump_all)
    model.save(filepath, gen_modeldef, gen_actordef)
    endtime = time.perf_counter() - starttime
    message(log, "Export took {:.3f} seconds".format(endtime))
    if isinstance(log, bpy.types.Text):
        log.cursor_set(0)
    bpy.context.scene.frame_set(orig_frame)


@orientation_helper()
class ExportMD3(bpy.types.Operator, ExportHelper):
    '''Export to .md3'''
    bl_idname = "export.md3"
    bl_label = 'Export MD3'

    log_type_options = [
        ("console","Console","Log to console"),
        ("append","Append","Append to log file"),
        ("overwrite","Overwrite","Overwrite log file"),
        ("blender","Blender internal text","Write log to Blender text data block")
    ]

    filename_ext = ".md3"  # Used by ExportHelper
    # filepath is defined by ExportHelper
    filter_glob: bpy.props.StringProperty(
        default="*.md3",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )
    md3name: bpy.props.StringProperty(
        name="MD3 Name",
        description="MD3 header name / skin path (64 bytes)",
        maxlen=64,
        default="")
    log_type: bpy.props.EnumProperty(
        name="Save log",
        items=log_type_options,
        description="File logging options",
        default="console")
    dump_all: bpy.props.BoolProperty(
        name="Dump all",
        description="Dump all data for md3 to log",
        default=False)
    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Scale all objects from world origin (0,0,0)",
        default=1.0,
        precision=5)
    offsetx: bpy.props.FloatProperty(
        name="Offset X",
        description="Transition scene along x axis",
        default=0.0,
        precision=5)
    offsety: bpy.props.FloatProperty(
        name="Offset Y",
        description="Transition scene along y axis",
        default=0.0,
        precision=5)
    offsetz: bpy.props.FloatProperty(
        name="Offset Z",
        description="Transition scene along z axis",
        default=0.0,
        precision=5)
    use_ref_frame: bpy.props.BoolProperty(
        name="Use Reference Frame",
        description="Use a specific frame other than the current frame as the "
            "\"reference frame\". If there is more than one vertex at a given "
            "position at any given time in the animation, those vertices may "
            "be merged together. You can use a \"reference frame\" to choose "
            "a specific animation frame to use as a reference for vertices "
            "so that they are not merged together unexpectedly. If not "
            "specified, the \"reference frame\" is the current frame in the "
            "current scene",
        default=True)
    ref_frame: bpy.props.IntProperty(
        name="Reference Frame",
        description="The frame to use for vertices, UVs, and triangles. If "
        "not specified, uses the current frame in the current scene",
        default=0)
    gzdoom: bpy.props.BoolProperty(
        name="Export for GZDoom",
        description="Export the model for GZDoom; Fixes normals pointing "
            "straight up or straight down for when GZDoom displays the model",
        default=True)
    gen_modeldef: bpy.props.BoolProperty(
        name="Generate Modeldef",
        description="Generate a Modeldef.txt file for the model. The filename "
            "will be modeldef.modelname.txt",
        default=False)
    gen_actordef: bpy.props.BoolProperty(
        name="Generate ZScript",
        description="Generate a ZScript actor definition for the model. The "
            "filename will be zscript.modelname.txt",
        default=False)
    sprite_name: bpy.props.StringProperty(
        name="Sprite frame name",
        description="Initial name to use for the actor sprite frames",
        default="MDLA")
    sprite_tics: bpy.props.IntProperty(
        name="Sprite frame duration",
        description="How long each frame should last. If 0, frame duration is "
            "calculated based on scene FPS",
        default=0, min=0, soft_min=0)

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "md3name")
        row = col.row()
        row.prop(self, "log_type", text="Log")
        row.prop(self, "dump_all")
        col.prop(self, "scale")
        col.label(text="Offset:")
        row = col.row()
        row.prop(self, "offsetx", text="X")
        row.prop(self, "offsety", text="Y")
        row.prop(self, "offsetz", text="Z")
        col.prop(self, "axis_forward")
        col.prop(self, "axis_up")
        row = col.row()
        if self.use_ref_frame:
            row.prop(self, "use_ref_frame", text="")
            row.prop(self, "ref_frame")
        else:
            row.prop(self, "use_ref_frame")
        col.prop(self, "gzdoom")
        col.prop(self, "gen_actordef")
        col.prop(self, "gen_modeldef")
        if self.gen_actordef or self.gen_modeldef:
            col.prop(self, "sprite_name")
        if self.gen_actordef:
            col.prop(self, "sprite_tics")
            # Calculate and display animation timing
            frame_time = self.sprite_tics
            if frame_time == 0:
                frame_time = max(1, floor(35 / context.scene.render.fps))
            fps = 35 / frame_time
            frame_count = (
                context.scene.frame_end - context.scene.frame_start + 1)
            # "Animation tics" are the amount of tics an animation takes from
            # start to finish. "Total tics" are the amount of tics for 1 static
            # frame, plus the animation tics. Useful for choreography, such as
            # with the hangar bay doors in Wolfenstein: Blade of Agony C2M5_B
            anim_tics = frame_time * (frame_count - 1)
            total_tics = frame_time * frame_count
            frame_stats = ("{1} frames, {0:.3f} frames per second").format(
                fps, frame_count)
            tic_stats = "{0} animation tics, {1} total tics".format(
                anim_tics, total_tics)
            col.label(text=frame_stats)
            col.label(text=tic_stats)

    def execute(self, context):
        settings = self.as_keywords(ignore=(
            "log_type_options",
            "check_existing",
            "bl_idname",
            "bl_label",
            "use_ref_frame",
            "filter_glob",
        ))
        settings["depsgraph"] = context.evaluated_depsgraph_get()
        settings["orig_frame"] = bpy.context.scene.frame_current
        if not self.use_ref_frame: settings["ref_frame"] = None
        save_md3(**settings)
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(ExportMD3.bl_idname, text="GZDoom MD3", icon='BLENDER')

def register():
    bpy.utils.register_class(ExportMD3)
    bpy.types.TOPBAR_MT_file_export.append(menu_func)

def unregister():
    bpy.utils.unregister_class(ExportMD3)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func)

if __name__ == "__main__":
    register()
