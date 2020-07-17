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
        self.xyz = [0.0, 0.0, 0.0]
        self.normal = 0

    def GetSize(self):
        return struct.calcsize(self.binaryFormat)

    # copied from PhaethonH <phaethon@linux.ucla.edu> md3.py
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
    def Encode(self, normal):
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

        # [Nash] commented this out to fix wrong normals for faces pointing straight-down
        #if (x == 0.0) & (y == 0.0) :
        #   if z > 0.0:
        #       return 0
        #   else:
        #       return (128 << 8)

        lng = math.acos(z) * 255 / (2 * math.pi)
        lat = math.atan2(y, x) * 255 / (2 * math.pi)
        retval = ((int(lat) & 0xFF) << 8) | (int(lng) & 0xFF)
        return retval

    def Save(self, file):
        tmpData = [0] * 4
        tmpData[0] = int(self.xyz[0] * MD3_XYZ_SCALE)
        tmpData[1] = int(self.xyz[1] * MD3_XYZ_SCALE)
        tmpData[2] = int(self.xyz[2] * MD3_XYZ_SCALE)
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
    # numVerts = 0
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
        self.ident = ""
        self.name = ""
        self.flags = 0
        self.numFrames = 0
        # self.numShaders = 0
        # self.numVerts = 0
        # self.numTriangles = 0
        self.ofsTriangles = 0
        self.ofsShaders = 0
        self.ofsUV = 0
        self.ofsVerts = 0
        self.ofsEnd
        self.shader = ""
        self.triangles = []
        self.uv = []
        self.verts = []

    def GetSize(self):
        sz = struct.calcsize(self.binaryFormat)
        self.ofsTriangles = sz
        for t in self.triangles:
            sz += t.GetSize()
        self.ofsShaders = sz
        for s in self.shaders:
            sz += s.GetSize()
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
        tmpData[5] = len(self.verts) # self.numVerts
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

        # save the shader coordinates
        for s in self.shaders:
            s.Save(file)

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
        self.ident = 0
        self.version = 0
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
                             offsetz=0.0):
        self.savepath = savepath
        self.name = name
        self.logtype = logtype
        self.dumpall = dumpall
        self.scale = scale
        self.offsetx = offsetx
        self.offsety = offsety
        self.offsetz = offsetz

class BlenderSurface:
    def __init__(self, index, material, first_face):
        self.index = index  # Surface index
        self.material = material  # Blender material name -> Shader
        self.faces = [first_face]  # Blender faces of surface
        self.surface = md3Surface()  # MD3 surface
        self.vertices = []  # Indexes of Blender mesh vertices.

def print_md3(log,md3,dumpall):
    message(log,"Header Information")
    message(log,"Ident: " + str(md3.ident))
    message(log,"Version: " + str(md3.version))
    message(log,"Name: " + md3.name)
    message(log,"Flags: " + str(md3.flags))
    message(log,"Number of Frames: " + str(md3.numFrames))
    message(log,"Number of Tags: " + str(md3.numTags))
    message(log,"Number of Surfaces: " + str(md3.numSurfaces))
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
            message(log," # of Shaders: " + str(s.numShaders))
            message(log," # of Verts: " + str(s.numVerts))
            message(log," # of Triangles: " + str(s.numTriangles))
            message(log," Offset Triangles: " + str(s.ofsTriangles))
            message(log," Offset UVs: " + str(s.ofsUV))
            message(log," Offset Verts: " + str(s.ofsVerts))
            message(log," Offset End: " + str(s.ofsEnd))
            message(log," Shaders:")
            for shader in s.shaders:
                message(log,"  Name: " + shader.name)
                message(log,"  Index: " + str(shader.index))
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
        shader_count += surface.numShaders
        tri_count += surface.numTriangles
        vert_count += surface.numVerts
        if surface.numShaders >= MD3_MAX_SHADERS:
            message(log,"!Warning: Shader limit (" + str(surface.numShaders) + "/" + str(MD3_MAX_SHADERS) + ") reached for surface " + surface.name)
        if surface.numVerts >= MD3_MAX_VERTICES:
            message(log,"!Warning: Vertex limit (" + str(surface.numVerts) + "/" + str(MD3_MAX_VERTICES) + ") reached for surface " + surface.name)
        if surface.numTriangles >= MD3_MAX_TRIANGLES:
            message(log,"!Warning: Triangle limit (" + str(surface.numTriangles) + "/" + str(MD3_MAX_TRIANGLES) + ") reached for surface " + surface.name)

    if md3.numTags >= MD3_MAX_TAGS:
        message(log,"!Warning: Tag limit (" + str(md3.numTags) + "/" + str(MD3_MAX_TAGS) + ") reached for md3!")
    if md3.numSurfaces >= MD3_MAX_SURFACES:
        message(log,"!Warning: Surface limit (" + str(md3.numSurfaces) + "/" + str(MD3_MAX_SURFACES) + ") reached for md3!")
    if md3.numFrames >= MD3_MAX_FRAMES:
        message(log,"!Warning: Frame limit (" + str(md3.numFrames) + "/" + str(MD3_MAX_FRAMES) + ") reached for md3!")

    message(log,"Total Shaders: " + str(shader_count))
    message(log,"Total Triangles: " + str(tri_count))
    message(log,"Total Vertices: " + str(vert_count))

# From Blender 2.79 scripts/addons/io_scene_obj/export_obj.py
def mesh_triangulate(me):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()

# Main function
def save_md3(settings):
    starttime = time.clock()  # start timer
    newlogpath = os.path.splitext(settings.savepath)[0] + ".log"
    dumpall = settings.dumpall
    if settings.logtype == "append":
        log = open(newlogpath,"a")
    elif settings.logtype == "overwrite":
        log = open(newlogpath,"w")
    else:
        log = 0
    message(log,"######################BEGIN######################")
    md3 = md3Object()
    md3.ident = MD3_IDENT
    md3.version = MD3_VERSION
    md3.name = settings.name
    md3.numFrames = (bpy.context.scene.frame_end + 1) - bpy.context.scene.frame_start
    if len(bpy.context.selected_objects) == 0:
        message(log, "Select an object to export!")
    for obj in bpy.context.selected_objects:
        save_object(md3, settings, obj)

def save_object(md3, settings, bobject):
    from mathutils import Euler, Vector
    # Set up rotation fix transformation
    rotation_fix = Euler()
    rotation_fix.z = math.radians(90)
    fix_transform = rotation_fix.to_matrix().to_4x4()
    fix_transform.translation = Vector((
        settings.offsetx, settings.offsety, settings.offsetz))
    if bobject.type == 'MESH':
        save_mesh(md3, bobject, fix_transform)
    elif bobject.type == 'EMPTY':
        save_tag(md3, bobject, fix_transform)

def save_mesh(md3, bmesh, fix_transform):
    # from collections import OrderedDict
    # Export UVs, triangles, and shader from first frame, and then export the
    # vertices for all subsequent frames
    first_frame_saved = False
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end + 1
    surface_infos = []
    for frame in range(start_frame, end_frame):
        bpy.context.scene.frame_set(frame)
        obj_mesh = bmesh.to_mesh(bpy.context.scene, True, 'PREVIEW')
        mesh_triangulate(obj_mesh)
        obj_mesh.transform(fix_transform)
        obj_mesh.calc_tessface()
        if not first_frame_saved:
            material_surfaces = {}
            # Find all surfaces, and export texture coordinates, triangles, and
            # shader (material name) from those.
            for face in obj_mesh.tessfaces:
                mtl = obj_mesh.materials[face.material_index]
                try:
                    mtl_name = mtl["md3shader"]
                except:
                    mtl_name = mtl.name
                if mtl_name not in material_surfaces:
                    surface_count = len(material_surfaces)
                    material_surfaces[mtl_name] = surface_count
                    # Surface index, material name, and surface faces
                    surface_infos.append(
                        BlenderSurface(surface_count, mtl_name, face))
                else:
                    surface_index = material_surfaces[mtl_name]
                    surface_infos[surface_index].faces.append(face)
            # Fill in vertex indices, normals, texture coordinates, and triangle
            # vertex indices for each surface
            for surface_info in surface_infos:
                face_vertices = OrderedDict()
                nsurface = surface_info.surface
                nsurface.shader = surface_info.material
                for face in surface_info.faces:
                    # Should not have more than 3 sides/vertices, since mesh
                    # was triangulated beforehand
                    ntri = md3Triangle()
                    for face_vertex_index, face_vertex in enumerate(face.vertices):
                        # A new vertex for each position, normal, and texture coordinate
                        vertex = obj_mesh.vertices[face_vertex]
                        vertex_pos = tuple(vertex.co)
                        vertex_normal = tuple(face.normal)
                        if face.use_smooth:
                            vertex_normal = tuple(vertex.normal)
                        vertex_uv = tuple(obj_mesh.tessface_uv_textures.active
                                 .data[face.index].uv[face_vertex_index])
                        vertex_id = (vertex_pos, vertex_normal, vertex_uv)
                        if vertex_id not in face_vertices:
                            face_vertices[vertex_id] = len(face_vertices)
                            nvert = md3Vert()
                            nvert.xyz = vertex_pos
                            nvert.normal = vertex_normal
                            nsurface.verts.append(nvert)
                            nuv = md3TexCoord()
                            nuv.u = vertex_uv[0]
                            nuv.v = vertex_uv[1]
                            nsurface.uv.append(nuv)
                        ntri.indexes[face_vertex_index] = face_vertices[vertex_id]
                        surface_info.vertices.append(face_vertex)
                    nsurface.triangles.append(ntri)
            first_frame_saved = True

def save_tag(md3, bempty, fix_transform):
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    position = bempty.location.copy()
    position = fix_transform * position
    orientation = bempty.matrix_world.to_3x3().normalize()
    orientation = fix_transform.to_3x3() * orientation
    ntag = md3Tag()
    ntag.origin = position
    ntag.axis[0:3] = orientation[0]
    ntag.axis[3:6] = orientation[1]
    ntag.axis[6:9] = orientation[2]
    md3.tags.append(ntag)

from bpy.props import *

class ExportMD3(bpy.types.Operator):
    '''Export to .md3'''
    bl_idname = "export.md3"
    bl_label = 'Export MD3'

    logenum = [("console","Console","log to console"),
                         ("append","Append","append to log file"),
                         ("overwrite","Overwrite","overwrite log file")]

    filepath = StringProperty(subtype = 'FILE_PATH',name="File Path", description="Filepath for exporting", maxlen= 1024, default="")
    md3name = StringProperty(name="MD3 Name", description="MD3 header name / skin path (64 bytes)",maxlen=64,default="")
    md3logtype = EnumProperty(name="Save log", items=logenum, description="File logging options",default =str(default_logtype))
    md3dumpall = BoolProperty(name="Dump all", description="Dump all data for md3 to log",default=default_dumpall)
    md3scale = FloatProperty(name="Scale", description="Scale all objects from world origin (0,0,0)",default=1.0,precision=5)
    md3offsetx = FloatProperty(name="Offset X", description="Transition scene along x axis",default=0.0,precision=5)
    md3offsety = FloatProperty(name="Offset Y", description="Transition scene along y axis",default=0.0,precision=5)
    md3offsetz = FloatProperty(name="Offset Z", description="Transition scene along z axis",default=0.0,precision=5)

    def execute(self, context):
        settings = md3Settings(savepath = self.properties.filepath,
                                                    name = self.properties.md3name,
                                                    logtype = self.properties.md3logtype,
                                                    dumpall = self.properties.md3dumpall,
                                                    scale = self.properties.md3scale,
                                                    offsetx = self.properties.md3offsetx,
                                                    offsety = self.properties.md3offsety,
                                                    offsetz = self.properties.md3offsetz)
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
