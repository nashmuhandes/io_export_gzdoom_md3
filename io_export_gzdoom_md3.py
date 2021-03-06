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
        "wiki_url": "",
        "tracker_url": "https://forum.zdoom.org/viewtopic.php?f=232&t=69417",
        "category": "Import-Export"}

import bpy, struct, math, time
from os.path import basename, splitext
from collections import OrderedDict
from struct import pack
from mathutils import Matrix, Vector

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



class MD3Vertex:
    xyz = []
    normal = 0
    binary_format = "<3hH"

    def __init__(self):
        self.xyz = [0, 0, 0]
        self.normal = 0

    def get_size(self):
        return struct.calcsize(self.binary_format)

    # copied from PhaethonH <phaethon@linux.ucla.edu> md3.py
    @staticmethod
    def decode(latlng):
        lat = (latlng >> 8) & 0xFF;
        lng = (latlng) & 0xFF;
        lat *= math.pi / 128;
        lng *= math.pi / 128;
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
            if (x == 0.0) & (y == 0.0):
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
    u = 0.0
    v = 0.0

    binary_format = "<2f"

    def __init__(self):
        self.u = 0.0
        self.v = 0.0

    def get_size(self):
        return struct.calcsize(self.binary_format)

    def save(self, file):
        uv_x = self.u
        uv_y = 1.0 - self.v
        data = struct.pack(self.binary_format, uv_x, uv_y)
        file.write(data)

class MD3Triangle:
    indexes = []

    binary_format = "<3i"

    def __init__(self):
        self.indexes = [ 0, 0, 0 ]

    def get_size(self):
        return struct.calcsize(self.binary_format)

    def save(self, file):
        indexes = self.indexes[:]
        indexes[1:3] = reversed(indexes[1:3])  # Winding order fix
        data = struct.pack(self.binary_format, *indexes)
        file.write(data)

class MD3Shader:
    name = ""
    index = 0

    binary_format = "<%dsi" % MAX_QPATH

    def __init__(self):
        self.name = ""
        self.index = 0

    def get_size(self):
        return struct.calcsize(self.binary_format)

    def save(self, file):
        name = str.encode(self.name)
        data = struct.pack(self.binary_format, name, self.index)
        file.write(data)

class MD3Surface:
    ident = ""
    name = ""
    flags = 0
    num_frames = 0
    num_verts = 0
    ofs_triangles = 0
    ofs_shaders = 0
    ofs_uv = 0
    ofs_verts = 0
    ofs_end = 0
    shader = ""
    triangles = []
    uv = []
    verts = []

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
        self.size = 0
        self.shader = MD3Shader()
        self.triangles = []
        self.uv = []
        self.verts = []

    def get_size(self):
        if self.size > 0:
            return self.size
        sz = struct.calcsize(self.binary_format)
        # Triangles
        self.ofs_triangles = sz
        for t in self.triangles:
            sz += t.get_size()
        # Shader
        self.ofs_shaders = sz
        sz += self.shader.get_size()
        # UVs (St)
        self.ofs_uv = sz
        for u in self.uv:
            sz += u.get_size()
        # Vertices for each frame
        self.ofs_verts = sz
        for v in self.verts:
            sz += v.get_size()
        # End
        self.ofs_end = sz
        self.size = sz
        return self.ofs_end

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
    name = ""
    origin = []
    axis = []

    binary_format="<%ds3f9f" % MAX_QPATH

    def __init__(self):
        self.name = ""
        self.origin = [0, 0, 0]
        self.axis = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def get_size(self):
        return struct.calcsize(self.binary_format)

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
    mins = 0
    maxs = 0
    local_origin = 0
    radius = 0.0
    name = ""

    binary_format="<3f3f3ff16s"

    def __init__(self):
        self.mins = [0, 0, 0]
        self.maxs = [0, 0, 0]
        self.local_origin = [0, 0, 0]
        self.radius = 0.0
        self.name = ""

    def get_size(self):
        return struct.calcsize(self.binary_format)

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
        temp_data[10] = str.encode("frame" + self.name)
        data = struct.pack(self.binary_format, *temp_data)
        file.write(data)

class MD3Object:
    # header structure
    ident = ""          # this is used to identify the file (must be IDP3)
    version = 0         # the version number of the file (Must be 15)
    name = ""
    flags = 0
    num_skins = 0
    ofs_frames = 0
    ofs_tags = 0
    ofs_surfaces = 0
    ofs_end = 0
    frames = []
    tags = []
    surfaces = []

    binary_format="<4si%ds9i" % MAX_QPATH  # little-endian (<), 17 integers (17i)

    def __init__(self):
        self.ident = MD3_IDENT
        self.version = MD3_VERSION
        self.name = ""
        self.flags = 0
        self.num_skins = 0
        self.ofs_frames = 0
        self.ofs_tags = 0
        self.ofs_surfaces = 0
        self.ofs_end = 0
        self.size = 0
        self.frames = []
        self.tags = []
        self.surfaces = []

    def get_size(self):
        if self.size > 0:
            return self.size
        self.ofs_frames = struct.calcsize(self.binary_format)
        self.ofs_tags = self.ofs_frames
        for f in self.frames:
            self.ofs_tags += f.get_size()
        self.ofs_surfaces += self.ofs_tags
        for t in self.tags:
            self.ofs_surfaces += t.get_size()
        self.ofs_End = self.ofs_surfaces
        for s in self.surfaces:
            self.ofs_end += s.get_size()
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
        temp_data[5] = len(self.tags)  # self.num_tags
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

class MD3Settings:
    def __init__(self,
                 savepath,
                 name,
                 logtype,
                 dumpall=False,
                 scale=1.0,
                 offsetx=0.0,
                 offsety=0.0,
                 offsetz=0.0,
                 refframe=0,
                 gzdoom=True,
                 modeldef=False,
                 zscript=False,
                 framename="MDLA",
                 frametime=0):
        self.savepath = savepath
        self.name = name
        self.logtype = logtype
        self.dumpall = dumpall
        self.scale = scale
        self.offsetx = offsetx
        self.offsety = offsety
        self.offsetz = offsetz
        self.refframe = refframe
        self.gzdoom = gzdoom
        self.modeldef = modeldef
        self.zscript = zscript
        self.framename = framename
        self.frametime = frametime


# Convert a XYZ vector to an integer vector for conversion to MD3
def convert_xyz(xyz):
    from math import floor
    def convert(number, factor):
        return floor(number * factor)
    factors = [MD3_XYZ_SCALE] * 3
    position = map(convert, xyz, factors)
    return tuple(map(int, position))


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
        self.vertex_refs = {}

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
        from math import floor
        self.md3 = MD3Object()
        self.md3.name = model_name
        self.material_surfaces = OrderedDict()
        self.mesh_objects = []
        self.fix_transform = Matrix.Identity(4)
        self.lock_vertices = False
        self.start_frame = bpy.context.scene.frame_start
        self.end_frame = bpy.context.scene.frame_end + 1
        self.frame_count = self.end_frame - self.start_frame
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
        from collections import namedtuple
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
            vertex_position = vertex.co
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
                vertex_normal = normal_object[normal_subindex][0:3]
            else:
                # No custom normals
                normal_object = getattr(obj_mesh, normal_ref)
                vertex_normal = normal_object[normal_index].normal
            # Get UV coordinates for this vertex.
            face_uvs = obj_mesh.tessface_uv_textures.active.data[face_index]
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
        from math import floor, log10
        # Add the vertex animations for each frame. Only call this AFTER
        # all the triangle and UV data has been set up.
        self.lock_vertices = True
        for frame in range(self.start_frame, self.end_frame):
            bpy.context.scene.frame_set(frame)
            obj_meshes = {}
            nframe = MD3Frame()
            frame_digits = floor(log10(self.end_frame - self.start_frame)) + 1
            frame_num = frame - self.start_frame
            nframe.name = (("{:0" + str(frame_digits) + "d}")
                           .format(frame_num))
            if bpy.context.active_object in self.mesh_objects:
                nframe.local_origin = bpy.context.active_object.location
            else:
                nframe.local_origin = self.mesh_objects[0]
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
                    if vertex.co[0] < nframe.mins[0]:
                        nframe.mins[0] = vertex.co[0]
                    if vertex.co[1] < nframe.mins[1]:
                        nframe.mins[1] = vertex.co[1]
                    if vertex.co[2] < nframe.mins[2]:
                        nframe.mins[2] = vertex.co[2]
                    # Check maxs
                    if vertex.co[0] > nframe.maxs[0]:
                        nframe.maxs[0] = vertex.co[0]
                    if vertex.co[1] > nframe.maxs[1]:
                        nframe.maxs[1] = vertex.co[1]
                    if vertex.co[2] > nframe.maxs[2]:
                        nframe.maxs[2] = vertex.co[2]
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
                            vertex_normal = normal_object[
                                vertex_info.normal_subindex][0:3]
                        else:
                            # Get the data
                            vertex_normal = normal_object[
                                vertex_info.normal_index].normal
                        # Set up MD3 vertex
                        nvertex = MD3Vertex()
                        nvertex.xyz = convert_xyz(vertex_position)
                        nvertex.normal = MD3Vertex.encode(
                            vertex_normal, self.gzdoom)
                        bsurface.surface.verts.append(nvertex)
            for obj_mesh in obj_meshes.values():
                bpy.data.meshes.remove(obj_mesh)  # mesh_obj.to_mesh_clear()

    def add_tag(self, bobject):
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
        position = bobject.location.copy()
        position = self.fix_transform * position
        orientation = bobject.matrix_world.to_3x3().normalize()
        orientation = self.fix_transform.to_3x3() * orientation
        ntag = MD3Tag()
        ntag.origin = position
        ntag.axis[0:3] = orientation[0]
        ntag.axis[3:6] = orientation[1]
        ntag.axis[6:9] = orientation[2]
        self.md3.tags.append(ntag)

    def get_modeldef(self, md3fname):
        model_def = """Model {actor_name}
{{
    Model "{file_name}"
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
        from math import log, floor
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
    message(log,"Number of Tags: " + str(len(md3.tags)))
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

        message(log,"Tags:")
        for t in md3.tags:
            message(log," Name: " + t.name)
            message(log," Origin: " + vec3_to_string(t.origin))
            message(log," Axis[0]: " + vec3_to_string(t.axis[0:3]))
            message(log," Axis[1]: " + vec3_to_string(t.axis[3:6]))
            message(log," Axis[2]: " + vec3_to_string(t.axis[6:9]))

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

    if len(md3.tags) >= MD3_MAX_TAGS:
        message(log,"!Warning: Tag limit (" + str(len(md3.tags)) + "/" + str(MD3_MAX_TAGS) + ") reached for md3!")
    if len(md3.surfaces) >= MD3_MAX_SURFACES:
        message(log,"!Warning: Surface limit (" + str(len(md3.surfaces)) + "/" + str(MD3_MAX_SURFACES) + ") reached for md3!")
    if len(md3.frames) >= MD3_MAX_FRAMES:
        message(log,"!Warning: Frame limit (" + str(len(md3.frames)) + "/" + str(MD3_MAX_FRAMES) + ") reached for md3!")

    message(log,"Total Shaders: " + str(shader_count))
    message(log,"Total Triangles: " + str(tri_count))
    message(log,"Total Vertices: " + str(vert_count))

# Main function
def save_md3(settings):
    from math import radians
    starttime = time.clock()  # start timer
    fullpath = splitext(settings.savepath)[0]
    modelname = basename(fullpath)
    logname = modelname + ".log"
    logfpath = fullpath + ".log"
    if settings.name == "":
        settings.name = modelname
    dumpall = settings.dumpall
    if settings.logtype == "append":
        log = open(logfpath,"a")
    elif settings.logtype == "overwrite":
        log = open(logfpath,"w")
    elif settings.logtype == "blender":
        log = bpy.data.texts.new(logname)
        log.clear()
    else:
        log = None
    ref_frame = settings.refframe
    if settings.refframe == -1:
        ref_frame = bpy.context.scene.frame_current
    message(log, "###################### BEGIN ######################")
    model = BlenderModelManager(settings.gzdoom, settings.name, ref_frame,
                                settings.framename, settings.scale,
                                settings.frametime)
    # Set up fix transformation matrix
    model.fix_transform *= Matrix.Scale(settings.scale, 4)
    model.fix_transform *= Matrix.Translation(Vector((
        settings.offsetx, settings.offsety, settings.offsetz)))
    model.fix_transform *= Matrix.Rotation(radians(90), 4, 'Z')
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
    model.save(settings.savepath, settings.modeldef, settings.zscript)
    endtime = time.clock() - starttime
    message(log, "Export took {:.3f} seconds".format(endtime))

from bpy.props import *

class ExportMD3(bpy.types.Operator):
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
    filepath = StringProperty(
        subtype='FILE_PATH',
        name="File Path",
        description="Filepath for exporting",
        maxlen=1024,
        default="")
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
    md3refframe = IntProperty(
        name="Reference Frame",
        description="The frame to use for vertices, UVs, and triangles. May "
            "be useful in case you have an animation where the model has an "
            "animation where it starts off closed and it 'opens up'. A value "
            "of -1 uses the current frame in the current scene",
        default=-1,
        min=-1)
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
        from math import floor
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
        col.prop(self, "md3refframe")
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
        settings = MD3Settings(savepath=self.properties.filepath,
                               name=self.properties.md3name,
                               logtype=self.properties.md3logtype,
                               dumpall=self.properties.md3dumpall,
                               scale=self.properties.md3scale,
                               offsetx=self.properties.md3offsetx,
                               offsety=self.properties.md3offsety,
                               offsetz=self.properties.md3offsetz,
                               refframe=self.properties.md3refframe,
                               gzdoom=self.properties.md3forgzdoom,
                               modeldef=self.properties.md3genmodeldef,
                               zscript=self.properties.md3genactordef,
                               framename=self.properties.md3framename,
                               frametime=self.properties.md3frametime)
        save_md3(settings)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

    @classmethod
    def poll(cls, context):
        return context.active_object != None

def menu_func(self, context):
    self.layout.operator(ExportMD3.bl_idname, text="GZDoom MD3", icon='BLENDER')

def register():
    bpy.utils.register_class(ExportMD3)
    bpy.types.INFO_MT_file_export.append(menu_func)

def unregister():
    bpy.utils.unregister_class(ExportMD3)
    bpy.types.INFO_MT_file_export.remove(menu_func)

if __name__ == "__main__":
    register()
