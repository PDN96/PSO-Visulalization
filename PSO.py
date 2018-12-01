import random
import math
import math
from scipy.spatial import Delaunay
import numpy as np
from fury import window, actor, ui
import dipy.io.vtk as io_vtk
import dipy.viz.utils as ut_vtk
from dipy.utils.optpkg import optional_package
import itertools


vtk, have_vtk, setup_module = optional_package('vtk')


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

def func5(x): #stlbanski tang
    sum = 0
    for i in x:
        sum += i**4 - 16*(i**2) + 5*i
    sum /= 2
    return sum

def func6(x): # auckley
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
        self.points = []
        # establish the swarm
        self.swarm = []
        for i in range(0, num_particles):
            x0 = [random.randint(bounds[0][0],bounds[0][1]), random.randint(bounds[1][0], bounds[1][1])] # can put loop for higher dimensions
            #x0 = [random.randint(-15,-5), random.randint(-3,3)]
            self.swarm.append(Particle(x0))
            self.points.append([x0[0], x0[1], self.costFunc(x0)])

        self.point_actor = actor.point(np.array(self.points), (1, 0, 0))
        self.vertices = self.get_vertices(self.costFunc, self.bounds)

        self.renderer = window.renderer(background=(1, 1, 1))
        self.surface_actor = self.surface(self.vertices, smooth="butterfly")
        self.renderer.add(self.surface_actor)
        self.renderer.add(self.point_actor)

        self.showm = window.ShowManager(self.renderer, size=(900, 768), reset_camera=False, order_transparent=True)
        self.showm.initialize()
        #window.show(self.renderer, size=(600, 600), reset_camera=False)
        self.showm.add_timer_callback(100, 2000, self.call_back)
        self.showm.render()
        self.showm.start()
        #window.record(self.showm.ren, size=(900, 768), out_path="viz_timer.png")


    def test2(self):
        result = list()
        pts = np.random.rand(10, 3)
        for p in pts:
            points = list(map(lambda x: x * 10, p))
            result.append(points)

    def test(self, obj, event):
        c = next(self.cnt)
        result = list()
        pts = np.random.rand (10, 3)
        for p in pts:
            points = list(map(lambda x: x * 10, p))
            result.append(points)

        point_actor = actor.point(np.array(result), (1, 0, 0))
        self.renderer.add(point_actor)

        if c > self.maxiter:
            self.showm.exit()
        #print(self.swarm[0].position_i)


    def get_vertices(self, costFunc, bounds):
        vertices = list()
        for x in range(bounds[0][0], bounds[0][1]):
            for y in range(bounds[1][0], bounds[1][1]):
                z = costFunc([x, y])
                vertices.append([x,y,z])
        return vertices

    def call_back(self, obj, event):
        c = next(self.cnt)
        # begin optimization loop
        self.points = []
        if c < self.maxiter:
            self.renderer.rm(self.point_actor)
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, self.num_particles):
                self.swarm[j].evaluate(self.costFunc)

                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = list(self.swarm[j].position_i)
                    self.err_best_g = float(self.swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, self.num_particles):
                self.swarm[j].update_velocity(self.pos_best_g)
                self.swarm[j].update_position(self.bounds)

            for j in range(0, self.num_particles):
                self.points.append([self.swarm[j].position_i[0], self.swarm[j].position_i[1], self.swarm[j].err_i])

            self.point_actor = actor.point(self.points, (1, 0, 0))
            self.renderer.add(self.point_actor)
        else:
            self.showm.exit()


    def surface(self, vertices, faces=None, smooth=None):
        temp1 = np.amax(vertices, axis=0)
        temp2 = np.amin(vertices, axis=0)
        size = list()
        for i in range(len(temp1)):
            if abs(temp1[i]) > abs(temp2[i]):
                size.append(abs(temp1[i]))
            else:
                size.append(abs(temp2[i]))

        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        points = vtk.vtkPoints()
        triangles = vtk.vtkCellArray()
        if faces is None:
            xy = list()
            for coordinate in vertices:
                xy.append([coordinate[0], coordinate[1]])
            tri = Delaunay(xy)
            faces = tri.simplices

        count = 0
        for face in faces:
            p_1 = vertices[face[0]]
            p_2 = vertices[face[1]]
            p_3 = vertices[face[2]]

            points.InsertNextPoint(p_1[0], p_1[1], p_1[2])
            points.InsertNextPoint(p_2[0], p_2[1], p_2[2])
            points.InsertNextPoint(p_3[0], p_3[1], p_3[2])

            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, count)
            triangle.GetPointIds().SetId(1, count + 1)
            triangle.GetPointIds().SetId(2, count + 2)

            triangles.InsertNextCell(triangle)
            count += 3

            r = [int(abs(p_1[0]) / float(size[0]) * 255), int(abs(p_1[1]) / float(size[1]) * 255), int(abs(p_1[2]) / float(size[2]) * 255)]
            colors.InsertNextTypedTuple(r)
            colors.InsertNextTypedTuple(r)
            colors.InsertNextTypedTuple(r)

        trianglePolyData = vtk.vtkPolyData()

        # Add the geometry and topology to the polydata
        trianglePolyData.SetPoints(points)
        trianglePolyData.GetPointData().SetScalars(colors)
        trianglePolyData.SetPolys(triangles)

        # Clean the polydata so that the edges are shared !
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.SetInputData(trianglePolyData)

        mapper = vtk.vtkPolyDataMapper()
        surface_actor = vtk.vtkActor()

        if smooth is None:
            mapper.SetInputData(trianglePolyData)
            surface_actor.SetMapper(mapper)

        elif smooth == "loop":
            smooth_loop = vtk.vtkLoopSubdivisionFilter()
            smooth_loop.SetNumberOfSubdivisions(3)
            smooth_loop.SetInputConnection(cleanPolyData.GetOutputPort())
            mapper.SetInputConnection(smooth_loop.GetOutputPort())
            surface_actor.SetMapper(mapper)

        elif smooth == "butterfly":
            smooth_butterfly = vtk.vtkButterflySubdivisionFilter()
            smooth_butterfly.SetNumberOfSubdivisions(3)
            smooth_butterfly.SetInputConnection(cleanPolyData.GetOutputPort())
            mapper.SetInputConnection(smooth_butterfly.GetOutputPort())
            surface_actor.SetMapper(mapper)

        return surface_actor


if __name__ == "__PSO__":
    main()

dim = [0, 0]  # dimensions
bounds = [(-32, 32), (-32, 32)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(func6, dim, bounds, num_particles=100, maxiter=200)

