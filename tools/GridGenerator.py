import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la
import igl
import meshpy.triangle as triangle
import numpy as np
import igl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def round_trip_connect(start, end):
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]

def laplace_smoothing(V, F, num_iterations=10):
    for _ in range(num_iterations):
        V = igl.laplacian_smooth(V, F)
    return V

do_improve = True
def main():
    points = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    facets = round_trip_connect(0, len(points) - 1)

    def needs_refinement(vertices, area):
        max_area = 0.025
        return bool(area > max_area)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)

    mesh = triangle.build(info, refinement_func=needs_refinement)
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    orig_mesh_points = np.array(mesh.points)
    orig_mesh_tris = np.array(mesh.elements)

    B = igl.boundary_loop(mesh_tris)
    bnd_points = []
    for bidx in B:
        bnd_points.append(mesh_points[bidx])
    bnd_points = np.array(bnd_points)

    adj_lst = igl.adjacency_list(mesh_tris)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    def animate(i):
        global do_improve
        if do_improve:
            eps = 0
            for i in range(0, len(mesh_points)):
                if i not in B:
                    neighbours = adj_lst[i]
                    sum = np.array([0.0, 0.0])
                    for ni in neighbours:
                        sum = np.add(sum, mesh_points[ni])
                    sum = sum / len(neighbours)
                    dp = sum - mesh_points[i]
                    eps = eps + dp
                    mesh_points[i] = mesh_points[i] + 0.01*dp
            err_eps = np.abs(np.sum(eps/len(mesh_points)))
            print("error : " + str(err_eps))
            if err_eps < 1.0e-3:
                do_improve = False
                V = np.c_[mesh_points, np.zeros(len(mesh_points))]
                igl.write_obj("data/data_grid.obj", V, mesh_tris)
                ani.event_source.stop()
                print("saved mesh points : " + str(len(mesh_points)))

        #plt.clf()
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax1.triplot(mesh_points[:, 0], mesh_points[:,1], mesh_tris)
        ax3.scatter(orig_mesh_points[:, 0], orig_mesh_points[:,1])
        ax1.scatter(orig_mesh_points[:, 0], orig_mesh_points[:,1], c=[[1.0, 0.0, 0.0]])
        ax2.triplot(orig_mesh_points[:, 0], orig_mesh_points[:,1], mesh_tris)
        ax1.scatter(bnd_points[:, 0], bnd_points[:,1], c=[[1.0, 0.0, 0.0]])
        ax4.scatter(mesh_points[:, 0], mesh_points[:,1])
        #plt.show()

    ani = animation.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
