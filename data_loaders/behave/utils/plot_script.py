import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = 'D:/ffmpeg_3.4.2/bin/x64/ffmpeg.exe'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, obj_points, hc_mask, oc_mask, title,  figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    # print(f" motion :{joints.shape} obj :{obj_points.shape}: hc {hc_mask.shape} oc: {oc_mask.shape}")

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # data *= 1.3  # scale for visualization


    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    # trajec = data[:, 0, [0, 2]]

    # if hint is not None:
    #     hint[..., 0] -= data[:, 0, 0]
    #     hint[..., 2] -= data[:, 0, 2]

    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5


        plot_xzPlane(MINS[0], MAXS[0] , 0, MINS[2],
                     MAXS[2] )




        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)
        if obj_points is not None:

            x2 = obj_points[ index, :, 0]
            y2 = obj_points[ index, :, 1]
            z2 = obj_points[ index, :, 2]
            ax.scatter(x2, y2, z2, color='grey', s=1, alpha=0.5)


            

        if hc_mask is not None:
            # hc_idx = np.where(hc_mask == 1.0)
            # if hc_idx != None:
            x3 = data[index, hc_mask, 0]
            y3 = data[index, hc_mask, 1]
            z3 = data[index, hc_mask, 2]
            ax.scatter(x3, y3, z3, color='red', s=6 ,alpha=1.0)
        if oc_mask is not None and oc_mask[0]!=-1:
            x4 = x2[oc_mask]
            y4 = y2[oc_mask]
            z4 = z2[oc_mask]
            ax.scatter(x4, y4, z4, color='blue', s=6, alpha=1.0)


        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)

    plt.close()
