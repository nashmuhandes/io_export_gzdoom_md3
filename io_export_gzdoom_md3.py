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
        "tracker_url": "https://forum.zdoom.org/viewtopic.php?f=232&t=35790",
        "category": "Import-Export"}

import bpy, struct, math, os, time
from collections import OrderedDict
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



class md3Vert:
    xyz = []
    normal = 0
    binaryFormat = "<3hH"

    def __init__(self):
        self.xyz = [0, 0, 0]
        self.normal = 0

    def GetSize(self):
        return struct.calcsize(self.binaryFormat)

    # copied from PhaethonH <phaethon@linux.ucla.edu> md3.py
    @staticmethod
    def Decode(self, latlng):
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
    def Encode(normal, gzdoom = True):
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

    def Save(self, file):
        tmpData = [0] * 4
        tmpData[0] = self.xyz[0]
        tmpData[1] = self.xyz[1]
        tmpData[2] = self.xyz[2]
        tmpData[3] = self.normal
        data = struct.pack(self.binaryFormat, tmpData[0], tmpData[1], tmpData[2], tmpData[3])
        file.write(data)

class md3TexCoord:
    u = 0.0
    v = 0.0

    binaryFormat = "<2f"

    def __init__(self):
        self.u = 0.0
        self.v = 0.0

    def GetSize(self):
        return struct.calcsize(self.binaryFormat)

    def Save(self, file):
        tmpData = [0] * 2
        tmpData[0] = self.u
        tmpData[1] = 1.0 - self.v
        data = struct.pack(self.binaryFormat, tmpData[0], tmpData[1])
        file.write(data)

class md3Triangle:
    indexes = []

    binaryFormat = "<3i"

    def __init__(self):
        self.indexes = [ 0, 0, 0 ]

    def GetSize(self):
        return struct.calcsize(self.binaryFormat)

    def Save(self, file):
        tmpData = [0] * 3
        tmpData[0] = self.indexes[0]
        tmpData[1] = self.indexes[2] # reverse
        tmpData[2] = self.indexes[1] # reverse
        data = struct.pack(self.binaryFormat,tmpData[0], tmpData[1], tmpData[2])
        file.write(data)

class md3Shader:
    name = ""
    index = 0

    binaryFormat = "<%dsi" % MAX_QPATH

    def __init__(self):
        self.name = ""
        self.index = 0

    def GetSize(self):
        return struct.calcsize(self.binaryFormat)

    def Save(self, file):
        tmpData = [0] * 2
        tmpData[0] = str.encode(self.name)
        tmpData[1] = self.index
        data = struct.pack(self.binaryFormat, tmpData[0], tmpData[1])
        file.write(data)

class md3Surface:
    ident = ""
    name = ""
    flags = 0
    numFrames = 0
    # numShaders = 0
    numVerts = 0
    # numTriangles = 0
    ofsTriangles = 0
    ofsShaders = 0
    ofsUV = 0
    ofsVerts = 0
    ofsEnd = 0
    shader = ""
    triangles = []
    uv = []
    verts = []

    binaryFormat = "<4s%ds10i" % MAX_QPATH  # 1 int, name, then 10 ints

    def __init__(self):
        self.ident = MD3_IDENT
        self.name = ""
        self.flags = 0
        self.numFrames = 0
        # self.numShaders = 0
        self.numVerts = 0
        # self.numTriangles = 0
        self.ofsTriangles = 0
        self.ofsShaders = 0
        self.ofsUV = 0
        self.ofsVerts = 0
        self.ofsEnd
        self.shader = md3Shader()
        self.triangles = []
        self.uv = []
        self.verts = []

    def GetSize(self):
        sz = struct.calcsize(self.binaryFormat)
        self.ofsTriangles = sz
        for t in self.triangles:
            sz += t.GetSize()
        self.ofsShaders = sz
        # for s in self.shaders:
        sz += self.shader.GetSize()
        self.ofsUV = sz
        for u in self.uv:
            sz += u.GetSize()
        self.ofsVerts = sz
        for v in self.verts:
            sz += v.GetSize()
        self.ofsEnd = sz
        return self.ofsEnd

    def Save(self, file):
        self.GetSize()
        tmpData = [0] * 12
        tmpData[0] = str.encode(self.ident)
        tmpData[1] = str.encode(self.name)
        tmpData[2] = self.flags
        tmpData[3] = self.numFrames
        tmpData[4] = 1 # len(self.shaders) # self.numShaders
        tmpData[5] = self.numVerts
        tmpData[6] = len(self.triangles) # self.numTriangles
        tmpData[7] = self.ofsTriangles
        tmpData[8] = self.ofsShaders
        tmpData[9] = self.ofsUV
        tmpData[10] = self.ofsVerts
        tmpData[11] = self.ofsEnd
        data = struct.pack(self.binaryFormat, tmpData[0],tmpData[1],tmpData[2],tmpData[3],tmpData[4],tmpData[5],tmpData[6],tmpData[7],tmpData[8],tmpData[9],tmpData[10],tmpData[11])
        file.write(data)

        # write the tri data
        for t in self.triangles:
            t.Save(file)

        # save the shaders
        self.shader.Save(file)

        # save the uv info
        for u in self.uv:
            u.Save(file)

        # save the verts
        for v in self.verts:
            v.Save(file)

class md3Tag:
    name = ""
    origin = []
    axis = []

    binaryFormat="<%ds3f9f" % MAX_QPATH

    def __init__(self):
        self.name = ""
        self.origin = [0, 0, 0]
        self.axis = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def GetSize(self):
        return struct.calcsize(self.binaryFormat)

    def Save(self, file):
        tmpData = [0] * 13
        tmpData[0] = self.name
        tmpData[1] = float(self.origin[0])
        tmpData[2] = float(self.origin[1])
        tmpData[3] = float(self.origin[2])
        tmpData[4] = float(self.axis[0])
        tmpData[5] = float(self.axis[1])
        tmpData[6] = float(self.axis[2])
        tmpData[7] = float(self.axis[3])
        tmpData[8] = float(self.axis[4])
        tmpData[9] = float(self.axis[5])
        tmpData[10] = float(self.axis[6])
        tmpData[11] = float(self.axis[7])
        tmpData[12] = float(self.axis[8])
        data = struct.pack(self.binaryFormat, tmpData[0].encode('utf-8'),tmpData[1],tmpData[2],tmpData[3],tmpData[4],tmpData[5],tmpData[6], tmpData[7], tmpData[8], tmpData[9], tmpData[10], tmpData[11], tmpData[12])
        file.write(data)

class md3Frame:
    mins = 0
    maxs = 0
    localOrigin = 0
    radius = 0.0
    name = ""

    binaryFormat="<3f3f3ff16s"

    def __init__(self):
        self.mins = [0, 0, 0]
        self.maxs = [0, 0, 0]
        self.localOrigin = [0, 0, 0]
        self.radius = 0.0
        self.name = ""

    def GetSize(self):
        return struct.calcsize(self.binaryFormat)

    def Save(self, file):
        tmpData = [0] * 11
        tmpData[0] = self.mins[0]
        tmpData[1] = self.mins[1]
        tmpData[2] = self.mins[2]
        tmpData[3] = self.maxs[0]
        tmpData[4] = self.maxs[1]
        tmpData[5] = self.maxs[2]
        tmpData[6] = self.localOrigin[0]
        tmpData[7] = self.localOrigin[1]
        tmpData[8] = self.localOrigin[2]
        tmpData[9] = self.radius
        tmpData[10] = str.encode("frame" + self.name)
        data = struct.pack(self.binaryFormat, tmpData[0],tmpData[1],tmpData[2],tmpData[3],tmpData[4],tmpData[5],tmpData[6],tmpData[7], tmpData[8], tmpData[9], tmpData[10])
        file.write(data)

class md3Object:
    # header structure
    ident = ""          # this is used to identify the file (must be IDP3)
    version = 0         # the version number of the file (Must be 15)
    name = ""
    flags = 0
    # numFrames = 0
    # numTags = 0
    # numSurfaces = 0
    numSkins = 0
    ofsFrames = 0
    ofsTags = 0
    ofsSurfaces = 0
    ofsEnd = 0
    frames = []
    tags = []
    surfaces = []

    binaryFormat="<4si%ds9i" % MAX_QPATH  # little-endian (<), 17 integers (17i)

    def __init__(self):
        self.ident = MD3_IDENT
        self.version = MD3_VERSION
        self.name = ""
        self.flags = 0
        # self.numFrames = 0
        # self.numTags = 0
        # self.numSurfaces = 0
        self.numSkins = 0
        self.ofsFrames = 0
        self.ofsTags = 0
        self.ofsSurfaces = 0
        self.ofsEnd = 0
        self.frames = []
        self.tags = []
        self.surfaces = []

    def GetSize(self):
        self.ofsFrames = struct.calcsize(self.binaryFormat)
        self.ofsTags = self.ofsFrames
        for f in self.frames:
            self.ofsTags += f.GetSize()
        self.ofsSurfaces += self.ofsTags
        for t in self.tags:
            self.ofsSurfaces += t.GetSize()
        self.ofsEnd = self.ofsSurfaces
        for s in self.surfaces:
            self.ofsEnd += s.GetSize()
        return self.ofsEnd

    def Save(self, file):
        self.GetSize()
        tmpData = [0] * 12
        tmpData[0] = str.encode(self.ident)
        tmpData[1] = self.version
        tmpData[2] = str.encode(self.name)
        tmpData[3] = self.flags
        tmpData[4] = len(self.frames) # self.numFrames
        tmpData[5] = len(self.tags) # self.numTags
        tmpData[6] = len(self.surfaces) # self.numSurfaces
        tmpData[7] = self.numSkins
        tmpData[8] = self.ofsFrames
        tmpData[9] = self.ofsTags
        tmpData[10] = self.ofsSurfaces
        tmpData[11] = self.ofsEnd

        data = struct.pack(self.binaryFormat, tmpData[0],tmpData[1],tmpData[2],tmpData[3],tmpData[4],tmpData[5],tmpData[6],tmpData[7], tmpData[8], tmpData[9], tmpData[10], tmpData[11])
        file.write(data)

        for f in self.frames:
            f.Save(file)

        for t in self.tags:
            t.Save(file)

        for s in self.surfaces:
            s.Save(file)


def message(log,msg):
    if log:
        log.write(msg + "\n")
    else:
        print(msg)

class md3Settings:
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
                 gzdoom=True):
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
    def __init__(self, index, material):
        self.index = index  # Surface index
        self.material = material  # Blender material name -> Shader
        self.surface = md3Surface()  # MD3 surface

        # {Mesh object: [(vertex index, normal index, normal reference), ...]}
        # Where "Mesh object" is the NAME of the object from which the mesh was
        # created, "vertex index" is the index of the vertex in mesh.vertices,
        # "normal index" is the index of the normal on the normal object, and
        # "normal reference" is a string referring to the array to use when
        # which has the normal to use.
        self.vertices = OrderedDict()

        # Vertices (position, normal, and UV) in MD3 binary format, mapped to
        # their indices
        self.unique_vertices = {}

    def GetSize(self):
        return self.surface.GetSize()


# A class to help manage a model, which consists of one or more objects which
# may be fused together into one model
class BlenderModelManager:
    def __init__(self, gzdoom, ref_frame = None):
        from mathutils import Matrix
        self.md3 = md3Object()
        self.material_surfaces = OrderedDict()
        self.mesh_objects = []
        self.fix_transform = Matrix.Identity(4)
        self.lock_vertices = False
        self.start_frame = bpy.context.scene.frame_start
        self.end_frame = bpy.context.scene.frame_end + 1
        self.gzdoom = gzdoom
        # Reference frame - used for initial UV and triangle data
        if ref_frame is not None:
            self.ref_frame = ref_frame
        else:
            self.ref_frame = self.start_frame

    def get_size(self):
        sz = self.md3.GetSize()
        for surface in self.material_surfaces:
            sz += surface.GetSize()

    def save(self, filename):
        nfile = open(filename, "wb")
        self.md3.Save(nfile)
        nfile.close()

    @staticmethod
    def encode_vertex(position, normal, uv, gzdoom):
        md3_position = convert_xyz(position)
        md3_normal = md3Vert.Encode(normal, gzdoom)
        return (pack(md3Vert.binaryFormat, *md3_position, md3_normal)
              + pack(md3TexCoord.binaryFormat, *uv))

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
        mesh_triangulate(obj_mesh)
        obj_mesh.calc_tessface()
        # See what materials the mesh references, and add new surfaces for
        # those materials if necessary
        for face_index, face in enumerate(obj_mesh.tessfaces):
            face_mtl = obj_mesh.materials[face.material_index].name
            if face_mtl not in self.material_surfaces:
                surface_index = len(self.material_surfaces)
                bsurface = BlenderSurface(surface_index, face_mtl)
                self.material_surfaces[face_mtl] = bsurface
                self.md3.surfaces.append(bsurface.surface)
            bsurface = self.material_surfaces[face_mtl]
            self._add_face(face_index, bsurface, obj_mesh, mesh_obj)

    def _add_face(self, face_index, bsurface, obj_mesh, mesh_obj):
        # A model has several surfaces, which have several faces. For Blender,
        # each face has a normal, UV coordinates, and references to at least 3
        # vertices. Each vertex also has a normal.
        # Which normal is used depends on whether or not the face is "smooth".
        # NOTE: tessfaces and tessface_uv_textures.active.data are parallel
        # arrays
        # For MD3:
        # A face consists of references to three vertices.
        # A vertex consists of a position, a normal, and a UV coordinate.
        face = obj_mesh.tessfaces[face_index]
        face_uvs = obj_mesh.tessface_uv_textures.active.data[face_index].uv
        ntri = md3Triangle()
        for vertex_iter_index, vertex_index in enumerate(face.vertices):
            vertex = obj_mesh.vertices[vertex_index]
            vertex_position = vertex.co
            normal_ref = "tessfaces"
            normal_index = face_index
            if face.use_smooth:
                normal_ref = "vertices"
                normal_index = vertex_index
            normal_object = getattr(obj_mesh, normal_ref)
            vertex_normal = normal_object[normal_index].normal
            vertex_uv = face_uvs[vertex_iter_index]
            vertex_id = BlenderModelManager.encode_vertex(
                vertex_position, vertex_normal, vertex_uv, self.gzdoom)
            if vertex_id not in bsurface.unique_vertices:
                bsurface.surface.numVerts += 1
                ntexcoord = md3TexCoord()
                ntexcoord.u = vertex_uv[0]
                ntexcoord.v = vertex_uv[1]
                bsurface.surface.uv.append(ntexcoord)
                md3_vertex_index = len(bsurface.unique_vertices)
                bsurface.unique_vertices[vertex_id] = md3_vertex_index
                bsurface.vertices.setdefault(mesh_obj.name, [])
                bsurface.vertices[mesh_obj.name].append((
                    vertex_index, normal_index, normal_ref))
            else:
                md3_vertex_index = bsurface.unique_vertices[vertex_id]
            ntri.indexes[vertex_iter_index] = md3_vertex_index
        bsurface.surface.triangles.append(ntri)

    def setup_frames(self):
        # Add the vertex animations for each frame. Only call this AFTER
        # all the triangle and UV data has been set up.
        self.lock_vertices = True
        for frame in range(self.start_frame, self.end_frame):
            bpy.context.scene.frame_set(frame)
            obj_meshes = {}
            nframe = md3Frame()
            nframe_bounds_set = False
            for mesh_obj in self.mesh_objects:
                obj_mesh = mesh_obj.to_mesh(bpy.context.scene, True, "PREVIEW")
                # Set up obj_mesh
                obj_mesh.transform(self.fix_transform * mesh_obj.matrix_world)
                mesh_triangulate(obj_mesh)
                obj_mesh.calc_tessface()
                # Set up frame min/max/origin/radius
                if not nframe_bounds_set:
                    nframe.mins = obj_mesh.vertices[0].co
                    nframe.maxs = obj_mesh.vertices[0].co
                    nframe.localOrigin = mesh_obj.location
                    nframe_bounds_set = True
                    armature = mesh_obj.find_armature()
                    if armature:
                        nframe.localOrigin -= armature.location
                for vertex in obj_mesh.vertices:
                    if vertex.co < nframe.mins:
                        nframe.mins = vertex.co
                    if vertex.co > nframe.maxs:
                        nframe.maxs = vertex.co
                nframe.radius = max(nframe.mins.length, nframe.maxs.length)
                self.md3.frames.append(nframe)
                # Add mesh to dict
                obj_meshes[mesh_obj.name] = obj_mesh
            for bsurface in self.material_surfaces.values():
                for mesh_name, vertex_infos in bsurface.vertices.items():
                    obj_mesh = obj_meshes[mesh_name]
                    for vertex_info in vertex_infos:
                        vertex_position = obj_mesh.vertices[vertex_info[0]].co
                        normal_object = getattr(obj_mesh, vertex_info[2])
                        vertex_normal = normal_object[vertex_info[1]].normal
                        nvertex = md3Vert()
                        nvertex.xyz = convert_xyz(vertex_position)
                        nvertex.normal = md3Vert.Encode(vertex_normal)
                        bsurface.surface.verts.append(nvertex)

    def add_tag(self, bobject):
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
        position = bobject.location.copy()
        position = self.fix_transform * position
        orientation = bobject.matrix_world.to_3x3().normalize()
        orientation = fix_transform.to_3x3() * orientation
        ntag = md3Tag()
        ntag.origin = position
        ntag.axis[0:3] = orientation[0]
        ntag.axis[3:6] = orientation[1]
        ntag.axis[6:9] = orientation[2]
        self.md3.tags.append(ntag)


def print_md3(log,md3,dumpall):
    message(log,"Header Information")
    message(log,"Ident: " + str(md3.ident))
    message(log,"Version: " + str(md3.version))
    message(log,"Name: " + md3.name)
    message(log,"Flags: " + str(md3.flags))
    message(log,"Number of Frames: " + str(len(md3.frames)))
    message(log,"Number of Tags: " + str(len(md3.tags)))
    message(log,"Number of Surfaces: " + str(len(md3.surfaces)))
    message(log,"Number of Skins: " + str(md3.numSkins))
    message(log,"Offset Frames: " + str(md3.ofsFrames))
    message(log,"Offset Tags: " + str(md3.ofsTags))
    message(log,"Offset Surfaces: " + str(md3.ofsSurfaces))
    message(log,"Offset end: " + str(md3.ofsEnd))

    if dumpall:
        message(log,"Frames:")
        for f in md3.frames:
            message(log," Mins: " + str(f.mins[0]) + " " + str(f.mins[1]) + " " + str(f.mins[2]))
            message(log," Maxs: " + str(f.maxs[0]) + " " + str(f.maxs[1]) + " " + str(f.maxs[2]))
            message(log," Origin(local): " + str(f.localOrigin[0]) + " " + str(f.localOrigin[1]) + " " + str(f.localOrigin[2]))
            message(log," Radius: " + str(f.radius))
            message(log," Name: " + f.name)

        message(log,"Tags:")
        for t in md3.tags:
            message(log," Name: " + t.name)
            message(log," Origin: " + str(t.origin[0]) + " " + str(t.origin[1]) + " " + str(t.origin[2]))
            message(log," Axis[0]: " + str(t.axis[0]) + " " + str(t.axis[1]) + " " + str(t.axis[2]))
            message(log," Axis[1]: " + str(t.axis[3]) + " " + str(t.axis[4]) + " " + str(t.axis[5]))
            message(log," Axis[2]: " + str(t.axis[6]) + " " + str(t.axis[7]) + " " + str(t.axis[8]))

        message(log,"Surfaces:")
        for s in md3.surfaces:
            message(log," Ident: " + s.ident)
            message(log," Name: " + s.name)
            message(log," Flags: " + str(s.flags))
            message(log," # of Frames: " + str(s.numFrames))
            # message(log," # of Shaders: " + str(s.numShaders))
            message(log," # of Verts: " + str(s.numVerts))
            message(log," # of Triangles: " + str(len(s.triangles)))
            message(log," Offset Triangles: " + str(s.ofsTriangles))
            message(log," Offset UVs: " + str(s.ofsUV))
            message(log," Offset Verts: " + str(s.ofsVerts))
            message(log," Offset End: " + str(s.ofsEnd))
            #message(log," Shaders:")
            #for shader in s.shaders:
                #message(log,"  Name: " + shader.name)
                #message(log,"  Index: " + str(shader.index))
            message(log," Shader name: " + s.shader.name)
            message(log," Triangles:")
            for tri in s.triangles:
                message(log,"  Indexes: " + str(tri.indexes[0]) + " " + str(tri.indexes[1]) + " " + str(tri.indexes[2]))
            message(log," UVs:")
            for uv in s.uv:
                message(log,"  U: " + str(uv.u))
                message(log,"  V: " + str(uv.v))
            message(log," Verts:")
            for vert in s.verts:
                message(log,"  XYZ: " + str(vert.xyz[0]) + " " + str(vert.xyz[1]) + " " + str(vert.xyz[2]))
                message(log,"  Normal: " + str(vert.normal))

    shader_count = 0
    vert_count = 0
    tri_count = 0
    for surface in md3.surfaces:
        shader_count += 1 # surface.numShaders
        tri_count += len(surface.triangles)
        vert_count += surface.numVerts
        #if surface.numShaders >= MD3_MAX_SHADERS:
            #message(log,"!Warning: Shader limit (" + str(surface.numShaders) + "/" + str(MD3_MAX_SHADERS) + ") reached for surface " + surface.name)
        if surface.numVerts >= MD3_MAX_VERTICES:
            message(log,"!Warning: Vertex limit (" + str(surface.numVerts) + "/" + str(MD3_MAX_VERTICES) + ") reached for surface " + surface.name)
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

# Copied from Blender 2.79 scripts/addons/io_scene_obj/export_obj.py
def mesh_triangulate(me):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()

# Main function
def save_md3(settings):
    from mathutils import Euler, Matrix, Vector
    from math import radians
    starttime = time.clock()  # start timer
    newlogpath = os.path.splitext(settings.savepath)[0] + ".log"
    dumpall = settings.dumpall
    if settings.logtype == "append":
        log = open(newlogpath,"a")
    elif settings.logtype == "overwrite":
        log = open(newlogpath,"w")
    elif settings.logtype == "blender":
        logname = os.path.basename(newlogpath)
        log = bpy.data.texts.get(logname, bpy.data.texts.new(logname))
        log.clear()
    else:
        log = None
    ref_frame = settings.refframe
    if settings.refframe == -1:
        ref_frame = bpy.context.scene.frame_current
    message(log, "###################### BEGIN ######################")
    model = BlenderModelManager(settings.gzdoom, ref_frame)
    # Set up fix transformation matrix
    model.fix_transform *= Matrix.Scale(settings.scale, 4)
    model.fix_transform *= Matrix.Translation(Vector((
        settings.offsetx, settings.offsety, settings.offsetz)))
    rotation_fix = Euler()
    rotation_fix.z = radians(90)
    model.fix_transform *= rotation_fix.to_matrix().to_4x4()
    model.md3.name = settings.name
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
    print_md3(log, model.md3, dumpall)
    model.save(settings.savepath)
    endtime = time.clock() - starttime
    message(log, "Export took {:.3f} seconds".format(endtime))

from bpy.props import *

class ExportMD3(bpy.types.Operator):
    '''Export to .md3'''
    bl_idname = "export.md3"
    bl_label = 'Export MD3'

    logenum = [
        ("console","Console","log to console"),
        ("append","Append","append to log file"),
        ("overwrite","Overwrite","overwrite log file"),
        ("blender","Blender internal text","Write log to Blender text data block")
    ]

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

    def execute(self, context):
        settings = md3Settings(savepath=self.properties.filepath,
                               name=self.properties.md3name,
                               logtype=self.properties.md3logtype,
                               dumpall=self.properties.md3dumpall,
                               scale=self.properties.md3scale,
                               offsetx=self.properties.md3offsetx,
                               offsety=self.properties.md3offsety,
                               offsetz=self.properties.md3offsetz,
                               refframe=self.properties.md3refframe,
                               gzdoom=self.properties.md3forgzdoom)
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
