import random
import math
import math
from scipy.spatial import Delaunay
import numpy as np
from fury import window, actor, ui
from dipy.utils.optpkg import optional_package
import itertools
from vtk.util import numpy_support
vtk, have_vtk, setup_module = optional_package('vtk')


shaderCode = """
//VTK::System::Dec // we still want the default
//Classic Perlin 3D Noise  // add functions for noise calculation
//by Stefan Gustavson
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
vec4 fade(vec4 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

float noise(vec4 P)
{
  vec4 Pi0 = floor(P); // Integer part for indexing
  vec4 Pi1 = Pi0 + 1.0; // Integer part + 1
  Pi0 = mod(Pi0, 289.0);
  Pi1 = mod(Pi1, 289.0);
  vec4 Pf0 = fract(P); // Fractional part for interpolation
  vec4 Pf1 = Pf0 - 1.0; // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = vec4(Pi0.zzzz);
  vec4 iz1 = vec4(Pi1.zzzz);
  vec4 iw0 = vec4(Pi0.wwww);
  vec4 iw1 = vec4(Pi1.wwww);

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);
  vec4 ixy00 = permute(ixy0 + iw0);
  vec4 ixy01 = permute(ixy0 + iw1);
  vec4 ixy10 = permute(ixy1 + iw0);
  vec4 ixy11 = permute(ixy1 + iw1);

  vec4 gx00 = ixy00 / 7.0;
  vec4 gy00 = floor(gx00) / 7.0;
  vec4 gz00 = floor(gy00) / 6.0;
  gx00 = fract(gx00) - 0.5;
  gy00 = fract(gy00) - 0.5;
  gz00 = fract(gz00) - 0.5;
  vec4 gw00 = vec4(0.75) - abs(gx00) - abs(gy00) - abs(gz00);
  vec4 sw00 = step(gw00, vec4(0.0));
  gx00 -= sw00 * (step(0.0, gx00) - 0.5);
  gy00 -= sw00 * (step(0.0, gy00) - 0.5);

  vec4 gx01 = ixy01 / 7.0;
  vec4 gy01 = floor(gx01) / 7.0;
  vec4 gz01 = floor(gy01) / 6.0;
  gx01 = fract(gx01) - 0.5;
  gy01 = fract(gy01) - 0.5;
  gz01 = fract(gz01) - 0.5;
  vec4 gw01 = vec4(0.75) - abs(gx01) - abs(gy01) - abs(gz01);
  vec4 sw01 = step(gw01, vec4(0.0));
  gx01 -= sw01 * (step(0.0, gx01) - 0.5);
  gy01 -= sw01 * (step(0.0, gy01) - 0.5);

  vec4 gx10 = ixy10 / 7.0;
  vec4 gy10 = floor(gx10) / 7.0;
  vec4 gz10 = floor(gy10) / 6.0;
  gx10 = fract(gx10) - 0.5;
  gy10 = fract(gy10) - 0.5;
  gz10 = fract(gz10) - 0.5;
  vec4 gw10 = vec4(0.75) - abs(gx10) - abs(gy10) - abs(gz10);
  vec4 sw10 = step(gw10, vec4(0.0));
  gx10 -= sw10 * (step(0.0, gx10) - 0.5);
  gy10 -= sw10 * (step(0.0, gy10) - 0.5);

  vec4 gx11 = ixy11 / 7.0;
  vec4 gy11 = floor(gx11) / 7.0;
  vec4 gz11 = floor(gy11) / 6.0;
  gx11 = fract(gx11) - 0.5;
  gy11 = fract(gy11) - 0.5;
  gz11 = fract(gz11) - 0.5;
  vec4 gw11 = vec4(0.75) - abs(gx11) - abs(gy11) - abs(gz11);
  vec4 sw11 = step(gw11, vec4(0.0));
  gx11 -= sw11 * (step(0.0, gx11) - 0.5);
  gy11 -= sw11 * (step(0.0, gy11) - 0.5);

  vec4 g0000 = vec4(gx00.x,gy00.x,gz00.x,gw00.x);
  vec4 g1000 = vec4(gx00.y,gy00.y,gz00.y,gw00.y);
  vec4 g0100 = vec4(gx00.z,gy00.z,gz00.z,gw00.z);
  vec4 g1100 = vec4(gx00.w,gy00.w,gz00.w,gw00.w);
  vec4 g0010 = vec4(gx10.x,gy10.x,gz10.x,gw10.x);
  vec4 g1010 = vec4(gx10.y,gy10.y,gz10.y,gw10.y);
  vec4 g0110 = vec4(gx10.z,gy10.z,gz10.z,gw10.z);
  vec4 g1110 = vec4(gx10.w,gy10.w,gz10.w,gw10.w);
  vec4 g0001 = vec4(gx01.x,gy01.x,gz01.x,gw01.x);
  vec4 g1001 = vec4(gx01.y,gy01.y,gz01.y,gw01.y);
  vec4 g0101 = vec4(gx01.z,gy01.z,gz01.z,gw01.z);
  vec4 g1101 = vec4(gx01.w,gy01.w,gz01.w,gw01.w);
  vec4 g0011 = vec4(gx11.x,gy11.x,gz11.x,gw11.x);
  vec4 g1011 = vec4(gx11.y,gy11.y,gz11.y,gw11.y);
  vec4 g0111 = vec4(gx11.z,gy11.z,gz11.z,gw11.z);
  vec4 g1111 = vec4(gx11.w,gy11.w,gz11.w,gw11.w);

  vec4 norm00 = taylorInvSqrt(vec4(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
  g0000 *= norm00.x;
  g0100 *= norm00.y;
  g1000 *= norm00.z;
  g1100 *= norm00.w;

  vec4 norm01 = taylorInvSqrt(vec4(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
  g0001 *= norm01.x;
  g0101 *= norm01.y;
  g1001 *= norm01.z;
  g1101 *= norm01.w;

  vec4 norm10 = taylorInvSqrt(vec4(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
  g0010 *= norm10.x;
  g0110 *= norm10.y;
  g1010 *= norm10.z;
  g1110 *= norm10.w;

  vec4 norm11 = taylorInvSqrt(vec4(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
  g0011 *= norm11.x;
  g0111 *= norm11.y;
  g1011 *= norm11.z;
  g1111 *= norm11.w;

  float n0000 = dot(g0000, Pf0);
  float n1000 = dot(g1000, vec4(Pf1.x, Pf0.yzw));
  float n0100 = dot(g0100, vec4(Pf0.x, Pf1.y, Pf0.zw));
  float n1100 = dot(g1100, vec4(Pf1.xy, Pf0.zw));
  float n0010 = dot(g0010, vec4(Pf0.xy, Pf1.z, Pf0.w));
  float n1010 = dot(g1010, vec4(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
  float n0110 = dot(g0110, vec4(Pf0.x, Pf1.yz, Pf0.w));
  float n1110 = dot(g1110, vec4(Pf1.xyz, Pf0.w));
  float n0001 = dot(g0001, vec4(Pf0.xyz, Pf1.w));
  float n1001 = dot(g1001, vec4(Pf1.x, Pf0.yz, Pf1.w));
  float n0101 = dot(g0101, vec4(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
  float n1101 = dot(g1101, vec4(Pf1.xy, Pf0.z, Pf1.w));
  float n0011 = dot(g0011, vec4(Pf0.xy, Pf1.zw));
  float n1011 = dot(g1011, vec4(Pf1.x, Pf0.y, Pf1.zw));
  float n0111 = dot(g0111, vec4(Pf0.x, Pf1.yzw));
  float n1111 = dot(g1111, Pf1);

  vec4 fade_xyzw = fade(Pf0);
  vec4 n_0w = mix(vec4(n0000, n1000, n0100, n1100), vec4(n0001, n1001, n0101, n1101), fade_xyzw.w);
  vec4 n_1w = mix(vec4(n0010, n1010, n0110, n1110), vec4(n0011, n1011, n0111, n1111), fade_xyzw.w);
  vec4 n_zw = mix(n_0w, n_1w, fade_xyzw.z);
  vec2 n_yzw = mix(n_zw.xy, n_zw.zw, fade_xyzw.y);
  float n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
  return 2.2 * n_xyzw;
}
"""

# functions we are attempting to optimize (minimize)
def func1(x):
    x1 = x[0]
    x2 = x[1]
    fact1 = - math.sin(x1 * math.cos(x2))
    fact2 = - math.exp(abs(1 - math.sqrt(x1 ** 2 + x2 ** 2) / math.pi))
    z = -abs(fact1 * fact2)
    return z

# minimum at (-10,1) bounds are (-15,-5), (-3,3)
def func2(x):
    x1 = x[0]
    x2 = x[1]
    term1 = 100 * math.sqrt(abs(x2 - 0.01 * (x1 ** 2)))
    term2 = 0.01 * abs(x1 + 10)
    y = term1 + term2
    return y


# minimum at (512, 404) bounds are (-512, 512), (-512, 512)
def func3(x):
    x1 = x[0]
    x2 = x[1]
    frac1 = 1 + math.cos(12 * math.sqrt(x1 ** 2 + x2 ** 2))
    frac2 = 0.5 * (x1 ** 2 + x2 ** 2) + 2
    y = -frac1 / frac2
    return y


def func4(x):
    x1 = x[0]
    x2 = x[1]
    y = x1 + x2
    return y

def s_tang(x): #stlbanski tang
    sum = 0
    for i in x:
        sum += i**4 - 16*(i**2) + 5*i
    sum /= 32
    return sum

def auckley(x): # bounds: (-32,32) , (-32,32)
    t1 = 0
    t2 = 0
    s1 = 0
    s2 = 0
    a = 20
    b = 0.2
    c = 2*math.pi
    d = len(x)
    for i in x:
        s1 += i **2
        s2 += math.cos(c*i)

    t1 = -a * math.exp(-b*math.sqrt(s1/d))
    t2 = - math.exp(s2/d)
    z = t1 + t2 + a +math.exp(1)
    return z

def cross_tray(x): # bounds: (-10,10), (-10,10)
    x1 = x[0]
    x2 = x[1]
    f1 = math.sin(x1)*math.sin(x2)
    f2 = math.exp(abs(100 - math.sqrt(x1**2 + x2**2)/math.pi))
    z = -0.0001 * (abs(f1*f2)+1)**0.1
    return z

def drop_wave(x):
    x1 = x[0]
    x2 = x[1]
    f1 = 1 + math.cos(12* math.sqrt(x1**2 + x2**2))
    f2 = 0.5 * (x1**2 + x2**2) + 2
    z = -f1/f2
    return z

def griewank(x):
    sum = 0
    prod = 1
    for i in range(len(x)):
        sum += (x[i]**2)/4000
        prod *= math.cos(x[i]/math.sqrt(i+1))
    z = sum - prod + 1
    return z

def rastrigin(x):
    sum = 0
    for i in x:
        sum += i**2 - 10 * math.cos(2*math.pi*i)
    z = 10*len(x) + sum
    return z

def three_hump(x):
    x1 = x[0]
    x2 = x[1]
    t1 = 2*(x1**2)
    t2 = -1.05 * (x1**4)
    t3 = (x1**6) / 6
    t4 = x1 * x2
    t5 = x2**2
    z= t1+t2+t3+t4+t5
    return z

def easom(x):
    x1 = x[0]
    x2 = x[1]
    f1 = - math.cos(x1) * math.cos(x2)
    f2 = math.exp(-(x1-math.pi)**2 - (x2-math.pi)**2);
    z = f1*f2;
    return z


class Particle:
    def __init__(self, x0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognative constant
        c2 = 2  # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


class PSO():
    def __init__(self, costFunc, x0, bounds, num_particles, maxiter):
        global num_dimensions
        num_dimensions = len(x0)
        self.err_best_g = -1  # best error for group
        self.pos_best_g = []  # best position for group
        self.costFunc = costFunc
        self.bounds = bounds
        self.num_particles = num_particles
        self.maxiter = maxiter
        self.cnt = itertools.count()
        self.old_points = []
        self.new_points = []
        self.lines = []
        self.arrow_actors = []
        self.colors = np.random.rand(num_particles, 3)
        # establish the swarm
        self.swarm = []
        for i in range(0, num_particles):
            x0 = [random.randint(bounds[0][0],bounds[0][1]), random.randint(bounds[1][0], bounds[1][1])] # can put loop for higher dimensions
            #x0 = [random.randint(-15,-5), random.randint(-3,3)]
            self.swarm.append(Particle(x0))
            self.old_points.append([x0[0], x0[1], self.costFunc(x0)])

        self.point_actor = actor.point(np.array(self.old_points), self.colors)
        self.vertices = self.get_vertices(self.costFunc, self.bounds)

        self.renderer = window.renderer()
        self.temp_surface_actor = self.surface(self.vertices, smooth="butterfly")
        self.surface_actor = self.get_shaded_surface(self.temp_surface_actor)
        self.renderer.add(self.surface_actor)
        self.renderer.add(self.point_actor)

        self.showm = window.ShowManager(self.renderer, size=(900, 768), reset_camera=False, order_transparent=True)
        self.showm.initialize()
        #window.show(self.renderer, size=(600, 600), reset_camera=False)
        self.showm.add_timer_callback(100, 2000, self.call_back)
        self.showm.render()
        self.showm.start()
        #window.record(self.showm.ren, size=(900, 768), out_path="viz_timer.png")

    def get_vertices(self, costFunc, bounds):
        vertices = list()
        for x in np.arange(bounds[0][0], bounds[0][1], 0.5):
            for y in np.arange(bounds[1][0], bounds[1][1], 0.5):
                z = costFunc([x, y])
                vertices.append([x,y,z])
        return vertices

    def call_back(self, obj, event):
        c = next(self.cnt)
        # begin optimization loop
        self.new_points = []
        self.old_points = []
        self.lines = []
        dummy_points = list()
        if c < self.maxiter:
            self.renderer.rm(self.point_actor)
            for arrow_actor in self.arrow_actors:
                self.renderer.rm(arrow_actor)

            for j in range(0, self.num_particles):
                self.old_points.append([self.swarm[j].position_i[0], self.swarm[j].position_i[1], self.costFunc(self.swarm[j].position_i)])

            # cycle through particles in swarm and evaluate fitness
            for j in range(0, self.num_particles):
                self.swarm[j].evaluate(self.costFunc)

                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = list(self.swarm[j].position_i)
                    self.err_best_g = float(self.swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, self.num_particles):
                #print(self.swarm[j].velocity_i)
                self.swarm[j].update_velocity(self.pos_best_g)
                #print(self.swarm[j].velocity_i)
                self.swarm[j].update_position(self.bounds)

            for j in range(0, self.num_particles):
                self.new_points.append([self.swarm[j].position_i[0], self.swarm[j].position_i[1], self.costFunc(self.swarm[j].position_i)])

            self.point_actor = actor.point(self.old_points, self.colors)
            self.renderer.add(self.point_actor)

            for j in range(0, self.num_particles):
                start = self.old_points[j]
                end = self.new_points[j]

                arrow_actor = self.arrow(start, end, 2, color= self.colors[j])
                self.renderer.add(arrow_actor)
                self.arrow_actors.append(arrow_actor)

            # self.lines = [self.old_points, self.new_points]
            # self.line_actor = actor.line(self.lines, self.colors)
            # self.renderer.add(self.line_actor)


            self.showm.render()

        else:
            self.showm.exit()

    def arrow(self, start_point, end_point, length, color = None):
        # Create an arrow.
        arrowSource = vtk.vtkArrowSource()

        # Generate a random start and end point
        # random.seed(8775070)
        startPoint = start_point
        endPoint = end_point

        # Compute a basis
        normalizedX = [0 for i in range(3)]
        normalizedY = [0 for i in range(3)]
        normalizedZ = [0 for i in range(3)]

        # The X axis is a vector from start to end
        math = vtk.vtkMath()
        # print(normalizedX)
        math.Subtract(endPoint, startPoint, normalizedX)
        l = math.Norm(normalizedX)
        # print(l)
        math.Normalize(normalizedX)
        # print(normalizedX)

        # The Z axis is an arbitrary vector cross X
        arbitrary = [0 for i in range(3)]
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)
        math.Cross(normalizedX, arbitrary, normalizedZ)
        math.Normalize(normalizedZ)

        # The Y axis is Z cross X
        math.Cross(normalizedZ, normalizedX, normalizedY)
        matrix = vtk.vtkMatrix4x4()

        # Create the direction cosine matrix
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])

        # Apply the transforms
        transform = vtk.vtkTransform()
        transform.Translate(startPoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # Transform the polydata
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(arrowSource.GetOutputPort())

        # Create a mapper and actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        arrow_actor = vtk.vtkActor()

        mapper.SetInputConnection(transformPD.GetOutputPort())

        arrow_actor.SetMapper(mapper)
        arrow_actor.GetProperty().SetColor(color)

        return arrow_actor

    def surface(self, vertices, faces=None, colors=None, smooth=None, subdivision=3):
        vertices = np.asarray(vertices)
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(vertices))
        triangle_poly_data = vtk.vtkPolyData()
        triangle_poly_data.SetPoints(points)

        if colors is not None:
            triangle_poly_data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(colors))

        if faces is None:
            tri = Delaunay(vertices[:, [0, 1]])
            faces = np.array(tri.simplices, dtype='i8')

        if faces.shape[1] == 3:
            triangles = np.empty((faces.shape[0], 4), dtype=np.int64)
            triangles[:, -3:] = faces
            triangles[:, 0] = 3
        else:
            triangles = faces

        if not triangles.flags['C_CONTIGUOUS'] or triangles.dtype != 'int64':
            triangles = np.ascontiguousarray(triangles, 'int64')

        cells = vtk.vtkCellArray()
        cells.SetCells(triangles.shape[0], numpy_support.numpy_to_vtkIdTypeArray(triangles, deep=True))
        triangle_poly_data.SetPolys(cells)

        clean_poly_data = vtk.vtkCleanPolyData()
        clean_poly_data.SetInputData(triangle_poly_data)

        mapper = vtk.vtkPolyDataMapper()
        surface_actor = vtk.vtkActor()

        if smooth is None:
            mapper.SetInputData(triangle_poly_data)
            surface_actor.SetMapper(mapper)

        elif smooth == "loop":
            smooth_loop = vtk.vtkLoopSubdivisionFilter()
            smooth_loop.SetNumberOfSubdivisions(subdivision)
            smooth_loop.SetInputConnection(clean_poly_data.GetOutputPort())
            mapper.SetInputConnection(smooth_loop.GetOutputPort())
            surface_actor.SetMapper(mapper)

        elif smooth == "butterfly":
            smooth_butterfly = vtk.vtkButterflySubdivisionFilter()
            smooth_butterfly.SetNumberOfSubdivisions(subdivision)
            smooth_butterfly.SetInputConnection(clean_poly_data.GetOutputPort())
            mapper.SetInputConnection(smooth_butterfly.GetOutputPort())
            surface_actor.SetMapper(mapper)

        return surface_actor

    def get_shaded_surface(self, surface_actor):
        colors = vtk.vtkNamedColors()
        surface_actor.GetProperty().SetDiffuse(1.0)

        surface_actor.GetProperty().SetSpecular(.5)
        surface_actor.GetProperty().SetOpacity(.5)
        surface_actor.GetProperty().SetSpecularPower(5)
        surface_actor.GetProperty().SetDiffuseColor(colors.GetColor3d("gold"))

        s_mapper = surface_actor.GetMapper()

        s_mapper.AddShaderReplacement(
            vtk.vtkShader.Vertex,
            "//VTK::Normal::Dec",  # // replace the normal block
            True,  # // before the standard replacements
            "//VTK::Normal::Dec\n"  # // we still want the default
            "  out vec4 myVertexMC;\n",
            False  # // only do it once
        );
        s_mapper.AddShaderReplacement(
            vtk.vtkShader.Vertex,
            "//VTK::Normal::Impl",  # // replace the normal block
            True,  # // before the standard replacements
            "//VTK::Normal::Impl\n"  # // we still want the default
            "  myVertexMC = vertexMC;\n",
            False  # // only do it once
        )

        # // Add the code to generate noise
        # // These functions need to be defined outside of main. Use the System::Dec
        # // to declare and implement
        s_mapper.AddShaderReplacement(
            vtk.vtkShader.Fragment,
            "//VTK::System::Dec",
            False,  # // before the standard replacements
            shaderCode,
            False  # // only do it once
        )
        # // Define varying and uniforms for the fragment shader here

        s_mapper.AddShaderReplacement(
            vtk.vtkShader.Fragment,  # // in the fragment shader
            "//VTK::Normal::Dec",  # // replace the normal block
            True,  # // before the standard replacements
            "//VTK::Normal::Dec\n"  # // we still want the default
            "  varying vec4 myVertexMC;\n"
            "  uniform vec3 veincolor = vec3(1.0, 1.0, 1.0);\n"
            "  uniform float veinfreq = 1;\n"
            "  uniform int veinlevels = 2;\n"
            "  uniform float warpfreq = 1;\n"
            "  uniform float warping = .5;\n"
            "  uniform float sharpness = 8.0;\n",
            False  # // only do it once
        )

        s_mapper.AddShaderReplacement(
            vtk.vtkShader.Fragment,  # // in the fragment shader
            "//VTK::Light::Impl",  # // replace the light block
            False,  # // after the standard replacements
            "//VTK::Light::Impl\n"  # // we still want the default calc
            "\n"
            "#define pnoise(x) ((noise(x) + 1.0) / 2.0)\n"
            "#define snoise(x) (2.0 * pnoise(x) - 1.0)\n"
            "  vec3 Ct;\n"
            "  int i;\n"
            "  float turb, freq;\n"
            "  float turbsum;\n"
            "  /* perturb the lookup */\n"
            "  freq = 1.0;\n"
            "  vec4 offset = vec4(0.0,0.0,0.0,0.0);\n"
            "  vec4 noisyPoint;\n"
            "  vec4 myLocalVertexMC = myVertexMC;\n"
            "\n"
            "    for (i = 0;  i < 6;  i += 1) {\n"
            "      noisyPoint[0] = snoise(warpfreq * freq * myLocalVertexMC);\n"
            "      noisyPoint[1] = snoise(warpfreq * freq * myLocalVertexMC);\n"
            "      noisyPoint[2] = snoise(warpfreq * freq * myLocalVertexMC);\n"
            "      noisyPoint[3] = 1.0;\n"
            "      offset += 2.0 * warping * (noisyPoint - 0.5)  / freq;\n"
            "      freq *= 2.0;\n"
            "    }\n"
            "    myLocalVertexMC.x += offset.x;\n"
            "    myLocalVertexMC.y += offset.y;\n"
            "    myLocalVertexMC.z += offset.z;\n"
            "\n"
            "    /* Now calculate the veining function for the lookup area */\n"
            "    turbsum = 0.0;  freq = 1.0;\n"
            "    myLocalVertexMC *= veinfreq;\n"
            "    for (i = 0;  i < veinlevels;  i += 1) {\n"
            "      turb = abs (snoise (myLocalVertexMC));\n"
            "      turb = pow (smoothstep (0.8, 1.0, 1.0 - turb), sharpness) / "
            "freq;\n"
            "      turbsum += (1.0-turbsum) * turb;\n"
            "      freq *= 1.5;\n"
            "      myLocalVertexMC *= 1.5;\n"
            "    }\n"
            "\n"
            "    Ct = mix (diffuseColor, veincolor, turbsum);\n"
            "\n"
            "  fragOutput0.rgb = opacity * (ambientColor + Ct + specular);\n"
            "  fragOutput0.a = opacity;\n",
            False  # // only do it once
        )

        return surface_actor


if __name__ == "__PSO__":
    main()

dim = [0, 0]  # dimensions

#bounds_cross_tray = [(-10,10), (-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
#PSO(cross_tray, dim, bounds_cross_tray, num_particles=100, maxiter=200)

# bounds_auckley = [(-32,32), (-32,32)]
# PSO(auckley, dim, bounds_auckley, num_particles=100, maxiter=200)

#bounds_drop = [(-6,6), (-6,6)]
#PSO(drop_wave, dim, bounds_drop, num_particles=100, maxiter=200)

bounds_griewank = [(-60,60), (-60,60)]
PSO(griewank, dim, bounds_griewank, num_particles=1000, maxiter=200)

#bounds_rastrigin = [(-6,6), (-6,6)]
#PSO(rastrigin, dim, bounds_rastrigin, num_particles=100, maxiter=200)

#bounds_hump = [(-5,5), (-5,5)]
#PSO(three_hump, dim, bounds_hump, num_particles=100, maxiter=200)

# bounds_easom = [(-50,50), (-50,50)]
# PSO(easom, dim, bounds_easom, num_particles=10, maxiter=200)

#bounds_tang = [(-4,4), (-4,4)]
#PSO(s_tang, dim, bounds_tang, num_particles=100, maxiter=200)


