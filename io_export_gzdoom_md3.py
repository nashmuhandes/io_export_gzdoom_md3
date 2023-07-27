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
#Updates and additions for Blender 2.6X by Derek McPherson
#
bl_info = {
        "name": "GZDoom .MD3",
        "author": "Derek McPherson, Xembie, PhaethonH, Bob Holcomb, Damien McGinnes, Robert (Tr3B) Beckebans, CoDEmanX, Mexicouger, Nash Muhandes, Kevin Caccamo",
        "version": (1, 6, 4), # 24th of August 2012 - Mexicouger
        "blender": (2, 6, 3),
        "location": "File > Export > GZDoom model (.md3)",
        "description": "Export mesh to GZDoom model with vertex animation (.md3)",
        "warning": "",
        "wiki_url": "https://forum.zdoom.org/viewtopic.php?f=232&t=69417",
        "tracker_url": "https://github.com/nashmuhandes/io_export_gzdoom_md3/issues",
        "category": "Import-Export"}

import array, bpy, struct, math, time
from bpy.props import *
from bpy_extras.io_utils import (
    ImportHelper,
    ExportHelper,
    axis_conversion,
    orientation_helper_factory,
)
from bpy.types import Operator
from collections import OrderedDict, namedtuple
from functools import reduce
from itertools import starmap, tee
from io import SEEK_SET
from math import floor, log10, radians
from mathutils import Matrix, Vector
from os.path import basename, splitext
from struct import pack

##### User options: Exporter default settings
default_logtype = 'console' ## console, overwrite, append
default_dumpall = False


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
# https://www.icculus.org/homepages/phaethon/q3a/formats/md3format.html#Normals
MD3_UP_NORMAL = 0
MD3_DOWN_NORMAL = 32768  # 128 << 8


def unmd3_string(data):
    "Given a byte slice taken from a C string, return a bytes object suitable "
    "for decoding to a UTF-8 string."
    null_pos = data.find(b"\0")
    data = data[0:null_pos]
    return data


class MD3Vertex:
    binary_format = "<3hH"

    def __init__(self):
        self.xyz = (0, 0, 0)
        self.normal = 0

    @staticmethod
    def get_size():
        return struct.calcsize(MD3Vertex.binary_format)

    @staticmethod
    def encode_xyz(xyz):
        "Convert an XYZ vector to an MD3 integer vector"
        def convert(number, factor):
            return floor(number * factor)
        factors = [MD3_XYZ_SCALE] * 3
        position = map(convert, xyz, factors)
        return tuple(map(int, position))

    @staticmethod
    def decode_xyz(xyz):
        "Convert an MD3 integer vector to a XYZ vector"
        def convert(number, factor):
            return number / factor
        factors = [MD3_XYZ_SCALE] * 3
        position = map(convert, xyz, factors)
        return Vector(position)

    # copied from PhaethonH <phaethon@linux.ucla.edu> md3.py
    @staticmethod
    def encode_normal(normal, gzdoom=True):
        n = normal.normalized()

        # Export for Quake 3 rather than GZDoom
        if not gzdoom:
            if n.z == 1.0:
                return MD3_UP_NORMAL
            elif n.z == -1.0:
                return MD3_DOWN_NORMAL

        lng = math.acos(n.z) * 255 / (2 * math.pi)
        lat = math.atan2(n.y, n.x) * 255 / (2 * math.pi)
        lng_byte = int(lng) & 0xFF
        lat_byte = int(lat) & 0xFF

        return (lat_byte << 8) | (lng_byte)

    # copied from PhaethonH <phaethon@linux.ucla.edu> md3.py
    @staticmethod
    def decode_normal(latlng, gzdoom=True):
        # Import from Quake 3 rather than GZDoom
        if not gzdoom:
            if latlng == MD3_UP_NORMAL:
                return Vector((0.0, 0.0, 1.0))
            elif latlng == MD3_DOWN_NORMAL:
                return Vector((0.0, 0.0, -1.0))

        lat = (latlng >> 8) & 0xFF
        lng = (latlng) & 0xFF
        lat *= math.pi / 128
        lng *= math.pi / 128

        x = math.cos(lat) * math.sin(lng)
        y = math.sin(lat) * math.sin(lng)
        z =                 math.cos(lng)

        return Vector((x, y, z))

    @staticmethod
    def encode(xyz, normal, gzdoom=True):
        xyz = MD3Vertex.encode_xyz(xyz)
        normal = MD3Vertex.encode_normal(normal, gzdoom)
        return xyz, normal

    @staticmethod
    def decode(xyz, normal, gzdoom=True):
        xyz = MD3Vertex.decode_xyz(xyz)
        normal = MD3Vertex.decode_normal(normal, gzdoom)
        return xyz, normal

    @staticmethod
    def read(file):
        data = file.read(MD3Vertex.get_size())
        data = struct.unpack(MD3Vertex.binary_format, data)
        nvertex = MD3Vertex()
        nvertex.xyz = tuple(data[0:3])
        nvertex.normal = data[3]
        return nvertex

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

    @staticmethod
    def read(file):
        data = file.read(MD3TexCoord.get_size())
        data = struct.unpack(MD3TexCoord.binary_format, data)
        ntexcoord = MD3TexCoord()
        ntexcoord.u = data[0]
        ntexcoord.v = data[1]
        return ntexcoord

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

    @staticmethod
    def read(file):
        data = file.read(MD3Triangle.get_size())
        data = struct.unpack(MD3Triangle.binary_format, data)
        ntri = MD3Triangle()
        ntri.indexes = data[:]
        return ntri

    def save(self, file):
        indexes = self.indexes[:]
        indexes[1:3] = reversed(indexes[1:3])  # Winding order fix
        data = struct.pack(self.binary_format, *indexes)
        file.write(data)

class MD3Shader:
    binary_format = "<%dsi" % MAX_QPATH

    def __init__(self):
        self.name = ""
        self.index = 0

    @staticmethod
    def get_size():
        return struct.calcsize(MD3Shader.binary_format)

    @staticmethod
    def read(file):
        data = file.read(MD3Shader.get_size())
        data = struct.unpack(MD3Shader.binary_format, data)
        nshader = MD3Shader()
        nshader.name = unmd3_string(data[0]).decode()
        nshader.index = data[1]
        return nshader

    def save(self, file):
        name = str.encode(self.name)
        data = struct.pack(self.binary_format, name, self.index)
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
        self.shader = MD3Shader()
        self.triangles = []
        self.uv = []
        self.verts = []

        self.size = 0

    def get_size(self):
        if self.size > 0:
            return self.size
        # Triangles (after header)
        self.ofs_triangles = struct.calcsize(self.binary_format)

        # Shader (after triangles)
        self.ofs_shaders = self.ofs_triangles + (
            MD3Triangle.get_size() * len(self.triangles))

        # UVs (after shader)
        self.ofs_uv = self.ofs_shaders + MD3Shader.get_size()

        # Vertices for each frame (after UVs)
        self.ofs_verts = self.ofs_uv + MD3TexCoord.get_size() * len(self.uv)

        # End (after vertices)
        self.ofs_end = self.ofs_verts + MD3Vertex.get_size() * len(self.verts)
        self.size = self.ofs_end
        return self.ofs_end

    @staticmethod
    def read(file):
        surface_start = file.tell()
        data = file.read(struct.calcsize(MD3Surface.binary_format))
        data = struct.unpack(MD3Surface.binary_format, data)
        if data[0] != b"IDP3":
            return None
        nsurf = MD3Surface()
        # nsurf.ident = data[0]
        nsurf.name = unmd3_string(data[1]).decode()
        nsurf.flags = data[2]
        nsurf.num_frames = data[3]
        num_shaders = data[4]
        nsurf.num_verts = data[5]
        num_tris = data[6]
        nsurf.ofs_triangles = data[7]
        nsurf.ofs_shaders = data[8]
        nsurf.ofs_uv = data[9]
        nsurf.ofs_verts = data[10]
        nsurf.ofs_end = data[11]

        file.seek(surface_start + nsurf.ofs_shaders, SEEK_SET)
        shaders = [MD3Shader.read(file) for x in range(num_shaders)]
        # Temporary workaround for the lack of support for multiple shaders
        # Most MD3 surfaces only use one shader anyways
        nsurf.shader = shaders[0]

        file.seek(surface_start + nsurf.ofs_triangles, SEEK_SET)
        nsurf.triangles = [MD3Triangle.read(file) for x in range(num_tris)]

        file.seek(surface_start + nsurf.ofs_uv, SEEK_SET)
        nsurf.uv = [MD3TexCoord.read(file) for x in range(nsurf.num_verts)]

        num_verts = nsurf.num_verts * nsurf.num_frames  # Vertex animation
        file.seek(surface_start + nsurf.ofs_verts, SEEK_SET)
        nsurf.verts = [MD3Vertex.read(file) for x in range(num_verts)]

        file.seek(surface_start + nsurf.ofs_end, SEEK_SET)
        return nsurf

    def save(self, file):
        self.get_size()
        temp_data = [0] * 12
        temp_data[0] = str.encode(self.ident)
        temp_data[1] = str.encode(self.name)
        temp_data[2] = self.flags
        temp_data[3] = self.num_frames
        temp_data[4] = 1  # len(self.shaders) # self.num_shaders
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
        self.shader.save(file)

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

    @staticmethod
    def read(file):
        data = file.read(MD3Tag.get_size())
        data = struct.unpack(MD3Tag.binary_format, data)
        ntag = MD3Tag()
        ntag.name = unmd3_string(data[0]).decode()
        ntag.origin = data[1:4]
        ntag.axis = data[4:13]
        return ntag

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

    @staticmethod
    def read(file):
        data = file.read(MD3Frame.get_size())
        data = struct.unpack(MD3Frame.binary_format, data)
        nframe = MD3Frame()
        nframe.mins = data[0:3]
        nframe.maxs = data[3:6]
        nframe.local_origin = data[6:9]
        nframe.radius = data[9]
        nframe.name = unmd3_string(data[10]).decode()
        return nframe

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

    @staticmethod
    def read(file):
        md3_start = file.tell()
        data = file.read(struct.calcsize(MD3Object.binary_format))
        data = struct.unpack(MD3Object.binary_format, data)
        if data[0] != b"IDP3":
            return None
        nobj = MD3Object()
        # nobj.ident = data[0]
        nobj.version = data[1]
        nobj.name = unmd3_string(data[2]).decode()
        nobj.flags = data[3]
        num_frames = data[4]
        nobj.num_tags = data[5]
        num_surfaces = data[6]
        nobj.num_skins = data[7]
        nobj.ofs_frames = data[8]
        nobj.ofs_tags = data[9]
        nobj.ofs_surfaces = data[10]
        nobj.ofs_end = data[11]

        file.seek(md3_start + nobj.ofs_frames, SEEK_SET)
        nobj.frames = [MD3Frame.read(file) for x in range(num_frames)]

        file.seek(md3_start + nobj.ofs_tags, SEEK_SET)
        num_tags = nobj.num_tags * num_frames
        nobj.tags = [MD3Tag.read(file) for x in range(num_tags)]

        file.seek(md3_start + nobj.ofs_surfaces, SEEK_SET)
        nobj.surfaces = [MD3Surface.read(file) for x in range(num_surfaces)]

        file.seek(md3_start + nobj.ofs_end, SEEK_SET)
        return nobj

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


# A class to help manage individual surfaces within a model
class BlenderSurface:
    def __init__(self, material):
        self.material = material  # Blender material name -> Shader
        self.surface = MD3Surface()  # MD3 surface
        # Set names for surface and its material, both of which are named after
        # the material it uses
        self.surface.name = material
        self.surface.shader.name = material

        # {Mesh object: [(vertex index, normal index, normal reference), ...]}
        # Where "Mesh object" is the NAME of the object from which the mesh was
        # created, "vertex index" is the index of the vertex in mesh.vertices,
        # "normal index" is the index of the normal on the normal object, and
        # "normal reference" is a string referring to the array to use when
        # which has the normal to use.
        self.vertex_refs = OrderedDict()

        # Vertices (position, normal, and UV) in MD3 binary format, mapped to
        # their indices
        self.unique_vertices = {}

    def get_size(self):
        return self.surface.get_size()


# Code to be used in a future Blender 2.80 port?
#from collections import namedtuple

#FaceVertex = namedtuple("FaceVertex", "face vertex")
#face_vertex_uvs = {}

#current_offset = 0
#for polygon in mesh.polygons:
    #for loop_index in range(polygon.loop_total):
        #loop_vertex = mesh.loops[polygon.loop_start + loop_index].vertex_index
        #face_vertex = FaceVertex(polygon.index, loop_vertex)
        #face_vertex_uvs[face_vertex] = mesh.uv_layers.active.data[current_offset]
        #current_offset += 1


# A class to help manage a model, which consists of one or more objects which
# may be fused together into one model
class BlenderModelManager:
    def __init__(self, gzdoom, model_name, ref_frame=None, frame_name="MDLA",
                 scale=1, frame_time=0):
        self.md3 = MD3Object()
        self.md3.name = model_name
        self.material_surfaces = OrderedDict()
        self.mesh_objects = []
        self.tag_objects = []
        self.tag_renames = {}
        self.fix_transform = Matrix.Identity(4)
        self.lock_vertices = False
        self.start_frame = bpy.context.scene.frame_start
        self.end_frame = bpy.context.scene.frame_end + 1
        self.frame_count = self.end_frame - self.start_frame
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
        self.ref_frame = ref_frame
        self.frame_name = frame_name[0:4]
        if frame_time == 0:
            frame_time = 35 / bpy.context.scene.render.fps
        self.frame_time = floor(max(frame_time, 1))
        self.scale = scale
        self.name = model_name

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
        md3_position = MD3Vertex.encode_xyz(position)
        md3_normal = MD3Vertex.encode_normal(normal, gzdoom)
        return (pack(MD3Vertex.binary_format, *md3_position, md3_normal)
              + pack(MD3TexCoord.binary_format, *uv))

    def add_mesh(self, mesh_obj):
        """
        Add a mesh to the object. Does nothing if the animation frames have
        been added.
        """
        if self.lock_vertices:
            return
        self.mesh_objects.append(mesh_obj)
        bpy.context.scene.frame_set(self.ref_frame)
        obj_mesh = mesh_obj.to_mesh(bpy.context.scene, True, 'PREVIEW')
        obj_mesh.transform(self.fix_transform * mesh_obj.matrix_world)
        # calc_normals_split recalculates normals, even on meshes without
        # custom normals. If I didn't do this, the vertex normals would be all
        # wrong.
        obj_mesh.calc_normals_split()
        obj_mesh.calc_tessface()
        # See what materials the mesh references, and add new surfaces for
        # those materials if necessary
        for face_index, face in enumerate(obj_mesh.tessfaces):
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
            if len(face.vertices) == 3:
                self._add_tri(bsurface, obj_mesh, mesh_obj.name, face_index,
                              face.vertices)
            elif len(face.vertices) == 4:
                self._add_quad(bsurface, obj_mesh, mesh_obj.name, face_index,
                               face)
            else:
                # Shouldn't happen; tessfaces have at most 4 vertices.
                print("WARNING! This face has more than 4 vertices!")
        bpy.data.meshes.remove(obj_mesh)  # mesh_obj.to_mesh_clear()

    def _add_tri(self, bsurface, obj_mesh, obj_name, face_index,
                 mesh_vertex_indices, face_vertex_indices=range(3)):
        # Define VertexReference named tuple
        VertexReference = namedtuple(
            "VertexReference",
            "vertex_index "
            # normal = getattr(obj, normal_ref)[normal_index]
            "normal_ref normal_index "
            # normal_obj = getattr(obj, normal_ref)[normal_index]
            # normal = getattr(normal_obj, normal_subref)[normal_subindex]
            "normal_subref normal_subindex "
        )
        ntri = MD3Triangle()
        for iter_index, face_vertex_index in enumerate(face_vertex_indices):
            # Set up the new triangle
            vertex_index = mesh_vertex_indices[face_vertex_index]
            # Get vertex ID, which is the vertex in MD3 binary format. First,
            # get the vertex position
            vertex = obj_mesh.vertices[vertex_index]
            vertex_position = vertex.co.copy()
            # Set up vertex reference. If the face is flat-shaded, the face
            # normal is used. Otherwise, the vertex normal is used.
            normal_ref = "tessfaces"
            normal_index = face_index
            normal_subref = None
            normal_subindex = None
            face = obj_mesh.tessfaces[face_index]
            if obj_mesh.has_custom_normals:
                normal_subref = "split_normals"
                normal_subindex = face_vertex_index
            elif face.use_smooth:
                # Get normal from vertex
                normal_ref = "vertices"
                normal_index = vertex_index
            # Get the normal. If a custom normal is used, use the sub-reference
            # to get the custom normal.
            if obj_mesh.has_custom_normals:
                # The custom normal is in tessface.split_normals
                normal_object = getattr(obj_mesh, normal_ref)[normal_index]
                normal_object = getattr(normal_object, normal_subref)
                vertex_normal = Vector(normal_object[normal_subindex][0:3])
            else:
                # No custom normals
                normal_object = getattr(obj_mesh, normal_ref)
                vertex_normal = normal_object[normal_index].normal
            # Get UV coordinates for this vertex.
            try:
                face_uvs = (
                    obj_mesh.tessface_uv_textures.active.data[face_index])
            except AttributeError:
                raise ValueError("The mesh needs a UV map!")
            vertex_uv = face_uvs.uv[face_vertex_index]
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
                    vertex_index,
                    normal_ref, normal_index,
                    normal_subref, normal_subindex
                ))
            else:
                # The vertex has already been added, so just get its index.
                md3_vertex_index = bsurface.unique_vertices[vertex_id]
            # Set the vertex index on the triangle.
            ntri.indexes[iter_index] = md3_vertex_index
        bsurface.surface.triangles.append(ntri)

    def _add_quad(self, bsurface, obj_mesh, obj_name, face_index, face):
        # Triangulate a quad
        # 0-----3
        # |\    |
        # | \   |
        # |  \  |
        # |   \ |
        # |    \|
        # 1-----2
        quad_tri_vertex_indices = [[0, 1, 2], [0, 2, 3]]
        for triangle in quad_tri_vertex_indices:
            self._add_tri(bsurface, obj_mesh, obj_name, face_index,
                          face.vertices, triangle)

    def setup_frames(self):
        # Add the vertex animations for each frame. Only call this AFTER
        # all the triangle and UV data has been set up.
        self.lock_vertices = True
        self.md3.num_tags = (
            len(self.tag_objects) if self.frame_count > 0 else 0)
        for frame in range(self.start_frame, self.end_frame):
            bpy.context.scene.frame_set(frame)
            obj_meshes = {}
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
            if bpy.context.active_object in self.mesh_objects:
                nframe.local_origin = (
                    bpy.context.active_object.location.copy())
            elif len(self.mesh_objects) > 0:
                nframe.local_origin = self.mesh_objects[0].location.copy()
            nframe_bounds_set = False
            for mesh_obj in self.mesh_objects:
                obj_mesh = mesh_obj.to_mesh(bpy.context.scene, True, "PREVIEW")
                # Set up obj_mesh
                obj_mesh.transform(self.fix_transform * mesh_obj.matrix_world)
                obj_mesh.calc_normals_split()
                obj_mesh.calc_tessface()
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
                obj_meshes[mesh_obj.name] = obj_mesh
            self.md3.frames.append(nframe)
            for bsurface in self.material_surfaces.values():
                for mesh_name, vertex_infos in bsurface.vertex_refs.items():
                    obj_mesh = obj_meshes[mesh_name]
                    for vertex_info in vertex_infos:
                        # Get vertex position
                        vertex_position = obj_mesh.vertices[
                            vertex_info.vertex_index].co
                        # Get vertex normal, using the sub-reference if
                        # it is available
                        normal_object = getattr(
                            obj_mesh, vertex_info.normal_ref)
                        if vertex_info.normal_subref is not None:
                            # Get the object to sub-reference
                            normal_object = normal_object[
                                vertex_info.normal_index]
                            # Use the sub-reference
                            normal_object = getattr(
                                normal_object, vertex_info.normal_subref)
                            # Copy the data
                            vertex_normal = Vector(normal_object[
                                vertex_info.normal_subindex][0:3])
                        else:
                            # Get the data
                            vertex_normal = normal_object[
                                vertex_info.normal_index].normal
                        # Set up MD3 vertex
                        nvertex = MD3Vertex()
                        nvertex.xyz = MD3Vertex.encode_xyz(vertex_position)
                        nvertex.normal = MD3Vertex.encode_normal(
                            vertex_normal, self.gzdoom)
                        bsurface.surface.verts.append(nvertex)
            for obj_mesh in obj_meshes.values():
                bpy.data.meshes.remove(obj_mesh)  # mesh_obj.to_mesh_clear()
            for tag_obj in self.tag_objects:
                tag_transform = self.fix_transform * tag_obj.matrix_world
                position, rotation, _ = tag_transform.decompose()
                # Makes the 'X' axis point forwards, assuming the tag is an
                # empty, shown as 'Arrows' with the 'Y' pointing forwards.
                local_transfix = Matrix.Rotation(radians(90), 3, 'Z')
                rotation = rotation.to_matrix() * local_transfix
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
        "Encode a base26 number. Takes a bytes-like object, returns the number."
        number = 0
        for index, char in enumerate(reversed(text)):
            char_value = self.alphabet.index(char)
            number += (self.base ** index) * char_value
        return number

    def decode(self, number, minlength=1):
        "Decode a base26 number. Takes a number, returns the text."
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
            # message(log," # of Shaders: " + str(s.num_shaders))
            message(log," # of Verts: " + str(s.num_verts))
            message(log," # of Triangles: " + str(len(s.triangles)))
            message(log," Offset Triangles: " + str(s.ofs_triangles))
            message(log," Offset UVs: " + str(s.ofs_uv))
            message(log," Offset Verts: " + str(s.ofs_verts))
            message(log," Offset End: " + str(s.ofs_end))
            #message(log," Shaders:")
            #for shader in s.shaders:
                #message(log,"  Name: " + shader.name)
                #message(log,"  Index: " + str(shader.index))
            message(log," Shader name: " + s.shader.name)
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
        shader_count += 1 # surface.num_shaders
        tri_count += len(surface.triangles)
        vert_count += surface.num_verts
        # if surface.num_shaders >= MD3_MAX_SHADERS:
            # message(log,"!Warning: Shader limit (" + str(surface.num_shaders) + "/" + str(MD3_MAX_SHADERS) + ") reached for surface " + surface.name)
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
def save_md3(filepath,
            md3name,
            md3logtype,
            md3dumpall,
            md3scale,
            md3offsetx,
            md3offsety,
            md3offsetz,
            ref_frame,
            md3forgzdoom,
            md3genmodeldef,
            md3genactordef,
            md3framename,
            md3frametime,
            axis_forward,
            axis_up):
    starttime = time.clock()  # start timer
    orig_frame = bpy.context.scene.frame_current
    fullpath = splitext(filepath)[0]
    modelname = basename(fullpath)
    logname = modelname + ".log"
    logfpath = fullpath + ".log"
    if md3name == "":
        md3name = modelname
    dumpall = md3dumpall
    if md3logtype == "append":
        log = open(logfpath,"a")
    elif md3logtype == "overwrite":
        log = open(logfpath,"w")
    elif md3logtype == "blender":
        log = bpy.data.texts.new(logname)
        log.clear()
    else:
        log = None
    if ref_frame is None:
        ref_frame = orig_frame
    message(log, "###################### BEGIN ######################")
    model = BlenderModelManager(md3forgzdoom, md3name, ref_frame,
                                md3framename, md3scale,
                                md3frametime)
    # Set up fix transformation matrix
    model.fix_transform *= Matrix.Scale(md3scale, 4)
    model.fix_transform *= Matrix.Translation(Vector((
        md3offsetx, md3offsety, md3offsetz)))
    model.fix_transform *= (
        axis_conversion(axis_forward, axis_up, to_forward='X').to_4x4())
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
    model.setup_frames()
    model.md3.get_size()
    print_md3(log, model.md3, dumpall)
    model.save(filepath, md3genmodeldef, md3genactordef)
    endtime = time.clock() - starttime
    bpy.context.scene.frame_set(orig_frame)
    message(log, "Export took {:.3f} seconds".format(endtime))


MD3OrientationHelper = orientation_helper_factory("MD3OrientationHelper")


class ExportMD3(Operator, ExportHelper, MD3OrientationHelper):
    '''Export to .md3'''
    bl_idname = "export.md3"
    bl_label = 'Export MD3'

    logenum = [
        ("console","Console","Log to console"),
        ("append","Append","Append to log file"),
        ("overwrite","Overwrite","Overwrite log file"),
        ("blender","Blender internal text","Write log to Blender text data block")
    ]

    filename_ext = ".md3"
    filter_glob = StringProperty(
        default="*.md3",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    md3name = StringProperty(
        name="MD3 Name",
        description="MD3 header name / skin path (64 bytes)",
        maxlen=64,
        default="")
    md3logtype = EnumProperty(
        name="Save log",
        items=logenum,
        description="File logging options",
        default=str(default_logtype))
    md3dumpall = BoolProperty(
        name="Dump all",
        description="Dump all data for md3 to log",
        default=default_dumpall)
    md3scale = FloatProperty(
        name="Scale",
        description="Scale all objects from world origin (0,0,0)",
        default=1.0,
        precision=5)
    md3offsetx = FloatProperty(
        name="Offset X",
        description="Transition scene along x axis",
        default=0.0,
        precision=5)
    md3offsety = FloatProperty(
        name="Offset Y",
        description="Transition scene along y axis",
        default=0.0,
        precision=5)
    md3offsetz = FloatProperty(
        name="Offset Z",
        description="Transition scene along z axis",
        default=0.0,
        precision=5)
    md3userefframe = BoolProperty(
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
    md3refframe = IntProperty(
        name="Reference Frame",
        description="The frame to use for vertices, UVs, and triangles. If "
        "not specified, uses the current frame in the current scene",
        default=0)
    md3forgzdoom = BoolProperty(
        name="Export for GZDoom",
        description="Export the model for GZDoom; Fixes normals pointing "
            "straight up or straight down for when GZDoom displays the model",
        default=True)
    md3genmodeldef = BoolProperty(
        name="Generate Modeldef",
        description="Generate a Modeldef.txt file for the model. The filename "
            "will be modeldef.modelname.txt",
        default=False)
    md3genactordef = BoolProperty(
        name="Generate ZScript",
        description="Generate a ZScript actor definition for the model. The "
            "filename will be zscript.modelname.txt",
        default=False)
    md3framename = StringProperty(
        name="Frame name",
        description="Initial name to use for the actor sprite frames",
        default="MDLA")
    md3frametime = IntProperty(
        name="Frame duration",
        description="How long each frame should last. If 0, frame duration is "
            "calculated based on scene FPS",
        default=0, min=0, soft_min=0)

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "md3name")
        row = col.row()
        row.prop(self, "md3logtype", "Log")
        row.prop(self, "md3dumpall")
        col.prop(self, "md3scale")
        col.label("Offset:")
        row = col.row()
        row.prop(self, "md3offsetx", "X")
        row.prop(self, "md3offsety", "Y")
        row.prop(self, "md3offsetz", "Z")
        col.prop(self, "axis_forward")
        col.prop(self, "axis_up")
        row = col.row()
        if self.md3userefframe:
            row.prop(self, "md3userefframe", text="")
            row.prop(self, "md3refframe")
        else:
            row.prop(self, "md3userefframe")
        col.prop(self, "md3forgzdoom")
        col.prop(self, "md3genactordef")
        col.prop(self, "md3genmodeldef")
        if self.properties.md3genactordef or self.properties.md3genmodeldef:
            col.prop(self, "md3framename")
        if self.properties.md3genactordef:
            col.prop(self, "md3frametime")
            frame_time = self.properties.md3frametime
            if frame_time == 0:
                frame_time = max(1, floor(35 / context.scene.render.fps))
            fps = 35 / frame_time
            frame_count = context.scene.frame_end - context.scene.frame_start
            total = frame_time * frame_count
            stats = "{0:.3f} fps, {2} frames, {1} total tics".format(fps, total, frame_count)
            col.label(stats)

    def execute(self, context):
        settings = self.as_keywords(ignore=(
            # Only properties (from bpy.props) are converted
            # Properties from/used by ExportHelper
            "filter_glob", "check_existing",
            # Proper value is manually computed below
            "md3refframe", "md3userefframe",
        ))
        settings["ref_frame"] = (
            self.properties.md3refframe
            if self.properties.md3userefframe
            else None)
        save_md3(**settings)
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        return context.active_object != None


def read_md3(filepath, md3forgzdoom, md3frame, fix_transform):
    # Copied from:
    # https://docs.python.org/3.5/library/itertools.html#itertools-recipes
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def mean(seq):
        count = len(seq)
        addup = lambda a, b: a + b
        return reduce(addup, seq) / count

    with open(filepath, "rb") as md3file:
        nobj = MD3Object.read(md3file)

    bl_mesh = bpy.data.meshes.new(nobj.name)

    edges = array.array('L')
    unique_edges = {}
    unique_vertices = {}
    unique_vnormals = {}
    vertex_positions = array.array('h')
    vertex_normals = array.array('H')

    polys_to_add = sum(map(lambda sf: len(sf.triangles), nobj.surfaces))
    loops_to_add = polys_to_add * 3  # All MD3 faces are triangles
    bl_mesh.polygons.add(polys_to_add)
    bl_mesh.loops.add(loops_to_add)

    # This also creates a new layer in bl_mesh.uv_layers
    bl_mesh.uv_textures.new("UVMap")

    # All MD3 faces are triangles
    for index, poly in enumerate(bl_mesh.polygons):
        poly.loop_start = index * 3
        poly.loop_total = 3

    md3frame = max(0, min(md3frame, len(nobj.frames)))

    loop_index = 0  # Running for all surfaces
    # Keep running count of total triangles per surface, otherwise,
    # triangles won't get referenced properly when setting them smooth
    surftri = 0

    for surf_index, nsurf in enumerate(nobj.surfaces):
        # Indexes for remapped vertices. If the vertices are not remapped,
        # Blender will remove some triangles when someone uses the
        # "Remove Doubles" operation
        vertex_remap = array.array('L')
        # MD3 surface -> Blender material
        surf_mtl = bpy.data.materials.new(nsurf.shader.name)
        bl_mesh.materials.append(surf_mtl)

        frame_verts = nsurf.verts[
            md3frame*nsurf.num_verts : (md3frame+1)*nsurf.num_verts]
        for vertex_index, nvertex in enumerate(frame_verts):
            vertex = nvertex.xyz
            if vertex not in unique_vertices:
                remap_index = len(unique_vertices)
                unique_vertices[vertex] = remap_index
                vertex_positions.extend(nvertex.xyz)
            else:
                remap_index = unique_vertices[vertex]
            normals = unique_vnormals.setdefault(vertex, [])
            normals.append(nvertex.normal)
            vertex_remap.append(remap_index)

        for ntri in nsurf.triangles:
            # Use the original indexes to get the normal and see if the
            # triangle should be "smooth" or not
            normals = tuple(map(
                lambda i: nsurf.verts[i].normal, ntri.indexes))
            bl_mesh.polygons[surftri].use_smooth = (
                normals != (normals[0],) * len(normals))
            bl_mesh.polygons[surftri].material_index = (
                surf_index)  # MD3 surface -> Blender material
            # Remap the indices to prevent unwanted vertex merging
            indexes = tuple(map(
                lambda i: vertex_remap[i], ntri.indexes))
            tri_edges = tuple(pairwise(indexes + (indexes[0],)))
            # Edges with the original indexes, used for UV coordinates
            o_edges = ntri.indexes
            for edge, oedge in zip(tri_edges, o_edges):
                edge_set = frozenset(edge)
                if edge_set not in unique_edges:
                    edge_index = len(unique_edges)
                    unique_edges[edge_set] = edge_index
                    edges.extend(edge)
                else:
                    edge_index = unique_edges[edge_set]
                uv = nsurf.uv[oedge]
                bl_mesh.uv_layers["UVMap"].data[loop_index].uv = (
                    uv.u, 1 - uv.v)
                bl_mesh.loops[loop_index].vertex_index = edge[0]
                bl_mesh.loops[loop_index].edge_index = edge_index
                loop_index += 1
            surftri += 1

    bl_mesh.vertices.add(len(unique_vertices))
    decode_normal = lambda n: MD3Vertex.decode_normal(n, md3forgzdoom)
    for index, vertex in enumerate(bl_mesh.vertices):
        pos = index * 3
        xyz = tuple(vertex_positions[pos : pos+3])
        vertex.co = MD3Vertex.decode_xyz(xyz)
        vertex.normal = mean(tuple(map(
            decode_normal, unique_vnormals[xyz])))

    edge_count = len(edges) // 2
    bl_mesh.edges.add(edge_count)
    for edge_index in range(edge_count):
        pos = edge_index * 2
        bl_mesh.edges[edge_index].vertices = edges[pos : pos+2]
    # Needed because MD3 X is forward
    bl_mesh.transform(fix_transform, shape_keys=True)
    bl_mesh.flip_normals()
    # Add the new object to the scene
    bl_object = bpy.data.objects.new(nobj.name, bl_mesh)
    bpy.context.scene.objects.link(bl_object)
    bpy.context.scene.update()
    return {'FINISHED'}


class ImportMD3(Operator, ImportHelper, MD3OrientationHelper):
    """Import a Quake 3 .md3 file"""
    # important since its how bpy.ops.import.md3 is constructed
    bl_idname = "import.md3"
    bl_label = "Import MD3"

    # ImportHelper mixin class uses this
    filename_ext = ".md3"

    filter_glob = StringProperty(
        default="*.md3",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    md3forgzdoom = BoolProperty(
        name="GZDoom",
        description="Fix normals when importing a model made for Quake 3",
        default=True,
    )

    md3frame = IntProperty(
        name="Frame number",
        description="The frame to import. Clamped to the range of available "
                    "frames in the MD3",
        min=0,
    )

    def execute(self, context):
        args = self.as_keywords(
            ignore=("filter_glob", "axis_up", "axis_forward"))
        args["fix_transform"] = axis_conversion(
            'Z', 'X', self.axis_up, self.axis_forward).to_4x4()
        return read_md3(**args)


def menu_func_export(self, context):
    self.layout.operator(ExportMD3.bl_idname, text="GZDoom MD3", icon='BLENDER')

def menu_func_import(self, context):
    self.layout.operator(ImportMD3.bl_idname, text="GZDoom MD3", icon='BLENDER')

def register():
    bpy.utils.register_class(ExportMD3)
    bpy.types.INFO_MT_file_export.append(menu_func_export)
    bpy.utils.register_class(ImportMD3)  # WIP!
    bpy.types.INFO_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(ExportMD3)
    bpy.types.INFO_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(ImportMD3)  # WIP!
    bpy.types.INFO_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()
