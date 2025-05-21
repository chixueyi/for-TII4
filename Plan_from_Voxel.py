#! /usr/bin/env python
# -*-coding:utf-8-*-

import numpy as np
import os
import sys
import copy
import open3d as o3d
from import_pkg.planners import PathPlanner
from utils import get_clock_time, normalize_vector, pointat2quat, bcolors, Observation, VoxelIndexingWrapper
import time
import math
from transforms3d.euler import euler2quat, quat2euler, euler2mat
from transforms3d.quaternions import quat2mat
import rospy
from std_msgs.msg import Float32MultiArray
from relaxed_ik_ros1.msg import EEPoseGoals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from import_pkg.rik_ckp_exep_package_cxy import myPlan
from datetime import datetime



# 点云路径
point_cloud_path = 'org.pcd'
# 保存均一体素化降采样之后的有颜色点云路径
# rgbpc_path = os.path.join('/home/ubt/cxy/rangedik_ws/src/cxy_motion_planning/save_20240118_132743_0/0118', 'uni_rgbpc.pcd')
rgbpc_path = 'uni_rgbpc.pcd'
mesh_path = 'test_mesh2.ply'

# 体素大小
voxel_size = 0.02

# planner_config 参数
planner_config = dict()
planner_config['stop_threshold'] = 0.001
planner_config['savgol_polyorder'] = 3
planner_config['savgol_window_size'] = 20
planner_config['obstacle_map_weight'] = 1
planner_config['max_steps'] = 100
planner_config['obstacle_map_gaussian_sigma'] = 10
planner_config['target_map_weight'] = 10
planner_config['stop_criteria'] = 'no_nearby_equal'
planner_config['target_spacing'] = 1
planner_config['max_curvature'] = 3
planner_config['pushing_skip_per_k'] = 5

# todo，先手动添加目标位置
target_points_eye = np.array([-0.301618, 0.267559, 0.624000]).astype(np.float32)  # 0302
# todo:考虑授粉棒与pre
# 方法一：将运动目标点设为授粉棒（z-0.31）目标前pre（z-0.15）位置，基于eye坐标系
target_points_eye -= np.array([-0.05774, 0.0, 0.46]).astype(np.float32)
# # 方法二：计算运动目标点设为授粉棒（z-0.31）目标前pre（z-0.15）位置，基于eye坐标系，之后根据z值在轨迹中截断
# target_points_tcp_pre = target_points - np.array([-0.05774, 0.0, 0.46]).astype(np.float32)
# target_points_zcut = target_points_tcp_pre[2]
# todo，手动添加start_pose
# start_pos = np.asarray([50, 30, 1])
start_point = np.array([-0.10773310176835739, -0.021171496518312516, -0.08393150749734063]) # 取手眼标定xyz负值，另外voxels_bounds_robot_min[2] = start_point[2]

# 手眼标定结果
gemini2_calib = {
    'qw': 0.7595794464710447, 'qx': 0.02939872631339185, 'qy': -0.023567738216973333, 'qz': -0.6493222167038705,
    'x': 0.10773310176835739, 'y': 0.021171496518312516, 'z': 0.08393150749734063}
clbqw, clbqx, clbqy, clbqz, clbx, clby, clbz = gemini2_calib['qw'],gemini2_calib['qx'],gemini2_calib['qy'],gemini2_calib['qz'],gemini2_calib['x'],gemini2_calib['y'],gemini2_calib['z']
clb_rot_mtx = quat2mat([clbqw, clbqx, clbqy, clbqz])
handeye = np.row_stack([np.column_stack([clb_rot_mtx, [clbx, clby, clbz]]), [0,0,0,1]])

# todo，设置起始和最终四元数
start_quat = euler2quat(-0.005, 0.944, 0.058)   # w,x,y,z #  [ 0.89025 -0.01541  0.45441  0.02696]
print('start_quat', start_quat)
end_quat = euler2quat(-0.266, 1.169, -0.081)  #[ 0.82891 -0.08836  0.55094  0.03964]
print('end_quat', end_quat)

# todo, 手动添加拍照机械臂位姿，为之后将traj转换到world
photo_xyzrpy = np.array([-0.101543, -0.015179, 0.613781, -0.005, 0.944, 0.058])
photo_base2hand_mat3 = euler2mat(photo_xyzrpy[3], photo_xyzrpy[4], photo_xyzrpy[5], axes='sxyz')
current_mat4 = np.array([[photo_base2hand_mat3[0, 0], photo_base2hand_mat3[0, 1], photo_base2hand_mat3[0, 2], photo_xyzrpy[0]],
                         [photo_base2hand_mat3[1, 0], photo_base2hand_mat3[1, 1], photo_base2hand_mat3[1, 2], photo_xyzrpy[1]],
                         [photo_base2hand_mat3[2, 0], photo_base2hand_mat3[2, 1], photo_base2hand_mat3[2, 2], photo_xyzrpy[2]],
                         [0, 0, 0, 1]])
base_eye4 = np.dot(current_mat4, handeye)

class Plan_from_Voxel():
    # 初始化
    def __init__(self, cloud, saved_folder_path=None, voxelsize=None, voxels_bounds_min=None, voxels_bounds_max=None):
        self.cloud = cloud
        self.saved_folder_path = saved_folder_path
        if voxelsize is None:
            self.voxelsize = 0.02
        else:
            self.voxelsize = voxelsize
        self.voxelgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, self.voxelsize)
        if voxels_bounds_min is None:
            self.voxels_bounds_min = self.cloud.get_min_bound().astype(np.float32)
            self.voxels_bounds_min[2] = start_point[2]  # todo, 设置map的最小z值
        else:
            self.voxels_bounds_min = voxels_bounds_min
        if voxels_bounds_max is None:
            self.voxels_bounds_max = self.cloud.get_max_bound().astype(np.float32)
        else:
            self.voxels_bounds_max = voxels_bounds_max
        # 注意这里直接求出来的实际是最大索引值，size要+1
        self.map_size = ((self.voxels_bounds_max - self.voxels_bounds_min) / self.voxelsize).astype(np.int32) + 1

    # 可视化及保存voxelgrid
    def save_voxelgrid(self, mesh_path):
        # o3d.visualization.draw_geometries([self.voxelgrid])
        # o3d.io.write_voxel_grid(mesh_path, self.voxelgrid)

        # downsampled_pcd = self.cloud.voxel_down_sample(voxel_size=voxel_size)
        # vox_mesh = o3d.geometry.TriangleMesh()
        # #
        # # go through all voxelgrid voxels
        # for idx in range(0, len(downsampled_pcd.points)):
        #     voxel_center = downsampled_pcd.points[idx]
        #     curr_color = downsampled_pcd.colors[idx]
        #     primitive = o3d.geometry.TriangleMesh.create_box(width = 0.02, height = 0.02, depth = 0.02)
        #     primitive.paint_uniform_color(curr_color)
        #     primitive.translate(voxel_center, relative=True)
        #     vox_mesh += primitive
        # # 保存网格mesh
        # o3d.io.write_triangle_mesh(mesh_path, vox_mesh)

        # get all voxels in the voxel grid
        vox_mesh = o3d.geometry.TriangleMesh()
        voxels_all = self.voxelgrid.get_voxels()
        # geth the calculated size of a voxel
        voxel_size = self.voxelsize
        # loop through all the voxels
        for voxel in voxels_all:
            # create a cube mesh with a size 1x1x1
            cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
            # paint it with the color of the current voxel
            cube.paint_uniform_color(voxel.color)
            # scale the box using the size of the voxel
            cube.scale(voxel_size, center=cube.get_center())
            # get the center of the current voxel
            voxel_center = self.voxelgrid.origin + voxel.grid_index * self.voxelgrid.voxel_size
            # voxel_center = self.voxelgrid.get_voxel_center_coordinate(voxel.grid_index)
            # translate the box to the center of the voxel
            cube.translate(voxel_center, relative=False)
            # add the box to the TriangleMesh object
            vox_mesh += cube
        vox_mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(vox_mesh)
        o3d.io.write_triangle_mesh(mesh_path, vox_mesh)
        print("voxels2stl saved.----------")

    # 保存彩色代价图
    def save_costmap(self, costmap, costmap_savepath):
        # ***********************************************************************
        # # 创建点云对象
        # point_cloud = o3d.geometry.PointCloud()
        # # 将非零值的点添加到点云中
        # points = []
        # colors = []
        # for a in range(self.map_size[0]):
        #     for b in range(self.map_size[1]):
        #         for c in range(self.map_size[2]):
        #             if costmap[a, b, c] > 0:  # 只添加非零值的点
        #                 points.append([a, b, c])
        #                 colors.append([costmap[a, b, c], 0, 1 - costmap[a, b, c]])  # 使用颜色来表示值的大小
        #
        # point_cloud.points = o3d.utility.Vector3dVector(points)
        # point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # # 保存点云为PLY文件
        # o3d.io.write_point_cloud(costmap_savepath + "point_cloud.ply", point_cloud)
        # ***********************************************************************


        import matplotlib.pyplot as plt

        from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        plt.figure(figsize=(8, 6))
        for zz in range(self.map_size[2]):
            plt.clf()
            # 选择一个深度进行投影
            depth_layer = zz
            projection = costmap[:, :, depth_layer]

            # 使用imshow函数显示投影
            # plt.imshow(projection, cmap='coolwarm', interpolation='nearest')
            plt.imshow(projection, cmap='jet', interpolation='nearest')
            # plt.colorbar(label='Normalized Value')  # 添加颜色条
            cbar = plt.colorbar(label='Normalized Value')  # Add colorbar
            cbar.ax.tick_params(labelsize=20)  # Set colorbar tick label font size

            plt.title(f'Projection at Depth Layer {depth_layer} / {self.map_size[2]}', fontsize=24)
            plt.xlabel('Width', fontsize=20)
            plt.ylabel('Height', fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=20)  # Increase tick font size

            plt.tight_layout()

            # 保存图像
            plt.savefig(costmap_savepath + 'projection_imshow_' + str(zz) + ".png")
            plt.pause(0.01)  # 等待0.5秒

        # 关闭图形窗口
        plt.close()


    # 保存手动体素均一化降采样之后的点云
    def save_univoxelized_rgbpc(self, rgbpc_path):
        # 体素转点云
        point_cloud_np = np.asarray(
            [self.voxelgrid.origin + pt.grid_index * self.voxelgrid.voxel_size for pt in self.voxelgrid.get_voxels()])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
        # 保存颜色
        color_np = np.asarray([pt.color for pt in self.voxelgrid.get_voxels()])
        # color_np = np.asarray([int(pt.color[0] * 255 * 256 * 256) + int(pt.color[1] * 255 * 256) + int(pt.color[3] * 255) for pt in voxel_grid.get_voxels()])
        pcd.colors = o3d.utility.Vector3dVector(color_np)
        # 保存点云pcd
        o3d.io.write_point_cloud(self.saved_folder_path + rgbpc_path, pcd)

    # 将点云转换为map，格子内有点则值为1，否则为0
    def pc2voxel_map(self, points=None):
        """given point cloud, create a fixed size voxel map, and fill in the voxels"""
        if points is None:
            points = self.cloud.points.astype(np.float32)
        points = np.asarray(points.points)
        points = np.clip(points, self.voxels_bounds_min, self.voxels_bounds_max)
        print("points.shape", points.shape)
        # voxelize
        voxel_xyz = (points - self.voxels_bounds_min) / (self.voxels_bounds_max - self.voxels_bounds_min) * (self.map_size - 1)
        # to integer
        _out = np.empty_like(voxel_xyz)
        points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32)
        voxel_map = np.zeros(self.map_size)
        for i in range(points_vox.shape[0]):
            voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
        return voxel_map

    # 将体素索引还原为3d点
    def voxelidx2point(self, voxels):
        """de-voxelize a voxel"""
        # check voxel coordinates are non-negative
        assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
        assert np.all(voxels < self.map_size), f'voxel max: {voxels.max()}'
        voxels = voxels.astype(np.float32)
        # de-voxelize
        pc = voxels / (self.map_size - 1) * (
                    self.voxels_bounds_max - self.voxels_bounds_min) + self.voxels_bounds_min
        return pc

    # 计算3d点所在的体素索引
    def point2voxelidx(self, pc):
        """voxelize a point cloud"""
        pc = pc.astype(np.float32)
        # make sure the point is within the voxel bounds
        pc = np.clip(pc, self.voxels_bounds_min, self.voxels_bounds_max)
        # voxelize
        voxels = (pc - self.voxels_bounds_min) / (self.voxels_bounds_max - self.voxels_bounds_min) * (self.map_size - 1)
        # to integer
        _out = np.empty_like(voxels)
        voxel_idx = np.round(voxels, 0, _out).astype(np.int32)
        assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
        assert np.all(voxels < self.map_size), f'voxel max: {voxels.max()}'
        return voxel_idx

    # 将体素索引轨迹转换为3d点轨迹
    def traj_voxels2points(self, path):
        traj = []
        for i in range(len(path)):
            voxel_xyz = path[i]
            eye_xyz = self.voxelidx2point(voxel_xyz)
            traj.append(eye_xyz)
        # for _ in range(2):
        #     traj.append(eye_xyz)
        return traj

    def save_traj(self, name, traj_xyz, count, color):
        # 可视化测试
        traj_array = np.asarray(traj_xyz)
        traj_pcd = o3d.geometry.PointCloud()
        traj_pcd.points = o3d.utility.Vector3dVector(traj_array)
        # 轨迹点为黑色
        # color_np = np.asarray([0.54117647, 0.16862745, 0.88627451]) * np.ones((len(traj_eye), 3))  # 黑色
        # traj_pcd.colors = o3d.utility.Vector3dVector(color_np)
        traj_pcd.paint_uniform_color(color)  # 绿色轨迹
        # 保存轨迹
        o3d.io.write_point_cloud(self.saved_folder_path + name + str(count) + ".pcd", traj_pcd)

    # 加入四元数, 四元数从初始值到最终值均匀变化
    def traj_join_quat(self, traj_base, start_quat, end_quat):
        traj_xyzquat = []
        traj_nums = len(traj_base)
        step = (start_quat - end_quat) / (traj_nums - 1)
        for i in range(traj_nums):
            traj_xyzquat.append(start_quat + step * i)
        traj_xyzquat = np.concatenate([traj_base, traj_xyzquat], axis=1)
        return traj_xyzquat

    def planall(self, target_points, start_point, planner_config, start_quat, end_quat=None, count=None):
        # 将整个点云根据体素格子数建立为map，并将有点的地方填充为1
        eye_map = self.pc2voxel_map(self.cloud)
        print("map_size: ", self.map_size)
        print("min_bound: ", self.voxels_bounds_min)
        print("max_bound: ", self.voxels_bounds_max)
        # 有点云落在其中的map索引：
        voxel_filled = np.array(np.where(eye_map == 1)).T
        # print("voxel_filled.shape: ", voxel_filled.shape)
        # 建立target_map,将目标点所在位置map值设为1
        target_idx = self.point2voxelidx(target_points)
        # print("target_idx: ", target_idx)
        target_map = np.zeros(self.map_size)
        target_map[target_idx[0], target_idx[1], target_idx[2]] = 1
        # 建立obstacle_map，存在场景点的格子值为1，其他为0，同时将target对应位置修改为0
        obstacle_map = eye_map.copy()
        obstacle_map[target_idx[0], target_idx[1], target_idx[2]] = 0

        # 设置start_pos，获取开始位置所在的索引
        start_idx = self.point2voxelidx(start_point)
        # print("start_pos", start_idx)
        # 路径规划，planner_config 在代码上面的地方
        start_time = time.time()
        print("\n")
        print("-----------------planning from start_idx[%d, %d, %d] to target_idx[%d, %d, %d]-----------------" % (
        start_idx[0], start_idx[1], start_idx[2], target_idx[0], target_idx[1], target_idx[2]))
        print("planner_config: ", planner_config)
        print(f'{bcolors.OKCYAN}[planners.py | {get_clock_time(milliseconds=True)}] starts{bcolors.ENDC}')
        object_centric = False
        planner = PathPlanner(planner_config, map_size=self.map_size)
        raw_path_voxel, path_voxel, planner_info = planner.optimize(start_idx, target_map, obstacle_map, object_centric)
        # path_voxel = raw_path_voxel # 取post_process之前的原始路径
        print(
            f'{bcolors.OKCYAN}[planners.py | {get_clock_time()}] planner time: {time.time() - start_time:.3f}s{bcolors.ENDC}')
        assert len(path_voxel) > 0, 'path_voxel is empty'
        # print("path_voxel", path_voxel)

        # # 保存costmap热值图
        # timestamp = str(datetime.now())
        # costpath_path = "/home/ubt/cxy/cxyplan_ws/saved_pcd_pose/costmap_2d/" + timestamp + "/"
        # os.makedirs(costpath_path)
        # self.save_costmap(planner_info['costmap'], costpath_path)

        # 三维xyz轨迹,base
        traj_optimized = self.traj_voxels2points(path_voxel)
        traj_optimized.append(target_points)
        self.save_traj("traj_optimized", traj_optimized, count, [0, 1, 0])

        # 保存原始轨迹
        traj_origin = self.traj_voxels2points(raw_path_voxel)
        traj_origin.append(target_points)
        self.save_traj("traj_origin", traj_origin, count, [0, 0, 1])

        # 加入四元数
        traj_array = np.asarray(traj_optimized)
        if end_quat is None:
            return traj_array, traj_origin
        else:
            traj_xyzquat = self.traj_join_quat(traj_array, start_quat, end_quat)
            return traj_xyzquat, traj_origin

# test:
def canfd_move(start_joint, ik_solution180):
    canfd_list = []
    print("ik_solution180", ik_solution180)

    for i in range(points_num_org - 1):
        deltai = ik_solution180[i + 1] - ik_solution180[i]
        # print("deltai", deltai)
        if max(deltai) > 40 or min(deltai) < -40:
            print("透传角度差异过大，停止运动")
            return
        if max(deltai) > 10 or min(deltai) < -10:
            d = deltai / 40
            for j in range(1, 41):
                canfd_list.append(ik_solution180[i] + d * j)
        elif max(abs(deltai)) > 6:
            d = deltai / 25
            for j in range(1, 26):
                canfd_list.append(ik_solution180[i] + d * j)
        else:
            d = deltai / 20
            for j in range(1, 21):
                canfd_list.append(ik_solution180[i] + d * j)
            # canfd_list.append(ik_solution180[i + 1])
    return canfd_list

''' # for real:
def canfd_move(robot, start_joint, ik_solution180):
    for i in range(points_num_org - 1):
        deltai = ik_solution180[i + 1] - ik_solution180[i]
        # print("deltai", deltai)
        if max(deltai) > 40 or min(deltai) < -40:
            print("透传角度差异过大，停止运动")
            return
        if max(deltai) > 10 or min(deltai) < -10:
            d = deltai / 40
            for j in range(1, 41):
                canfd_list.append(ik_solution180[i] + d * j)
        elif max(abs(deltai)) > 6:
            d = deltai / 25
            for j in range(1, 26):
                canfd_list.append(ik_solution180[i] + d * j)
        else:
            d = deltai / 20
            for j in range(1, 21):
                canfd_list.append(ik_solution180[i] + d * j)
    # 透传运动
    robot.Movej_Cmd(start_joint, 50, 0, True)
    for i in range(len(canfd_list)):
        robot.Movej_CANFD(canfd_list[i], False)
'''

def draw_joint_lines(canfd_list):
    import matplotlib.pyplot as plt
    # x轴数据
    x = np.linspace(0, len(canfd_list), len(canfd_list))
    canfd_array = np.asarray(canfd_list).T
    # y轴数据（三组）
    y_joint1 = canfd_array[0]
    y_joint2 = canfd_array[1]
    y_joint3 = canfd_array[2]
    y_joint4 = canfd_array[3]
    y_joint5 = canfd_array[4]
    y_joint6 = canfd_array[5]
    plt.figure()
    plt.plot(x, y_joint1, label='joint 1')
    plt.plot(x, y_joint2, label='joint 2')
    plt.plot(x, y_joint3, label='joint 3')
    plt.plot(x, y_joint4, label='joint 4')
    plt.plot(x, y_joint5, label='joint 5')
    plt.plot(x, y_joint6, label='joint 6')
    # 添加标题、x轴和y轴标签
    plt.title('Multiple Line Chart')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

def mcallback(data):
    #print("MCallback MCallback MCallback")
    # 判断接口类型
    if data.codeKey == MOVEJ_CANFD_CB:  # 角度透传
     #   print("透传结果:", data.errCode)
     #   print("当前角度:", data.joint[0], data.joint[1], data.joint[2], data.joint[3], data.joint[4], data.joint[5])
        pass
    elif data.codeKey == MOVEP_CANFD_CB:  # 位姿透传
        print("透传结果:", data.errCode)
        print("当前角度:", data.joint[0], data.joint[1], data.joint[2], data.joint[3], data.joint[4], data.joint[5])
        print("当前位姿:", data.pose.position.x, data.pose.position.y, data.pose.position.z, data.pose.euler.rx,
                data.pose.euler.ry, data.pose.euler.rz)
    elif data.codeKey == FORCE_POSITION_MOVE_CB:  # 力位混合透传
        print("透传结果:", data.errCode)
        print("当前力度：", data.nforce)

if __name__ == "__main__":
    # =============================================================
    #  cxy plan start:
    # =============================================================
    # 使用open3d处理,读取 PCD 格式的点云数据

    pcd = o3d.io.read_point_cloud(point_cloud_path)

    import copy
    # 转换到base下：
    pcdT = copy.deepcopy(pcd).transform(base_eye4)
    # o3d.io.write_point_cloud("org.pcd", pcd)
    # o3d.io.write_point_cloud("test_pcd2base.pcd", pcdT)

    # 从点云创建Plan_from_Voxel实例
    mypfv = Plan_from_Voxel(pcdT, voxel_size)
    # 保存均一体素降采样点云
    mypfv.save_univoxelized_rgbpc(rgbpc_path)
    mypfv.save_voxelgrid(mesh_path)
    print("voxels2ply saved.")



    '''# 开始规划, 注意如果没有输入起始和终止四元数，将只输出traj_base(x,y,z,1)
    ## traj_base = mypfv.planall(target_points, start_point, planner_config)
    # print("traj_base: ", traj_base)
    # 将目标点转到base下：
    target_points = np.dot(base_eye4, np.array([target_points_eye[0], target_points_eye[1], target_points_eye[2], 1]))
    print("target_points:", target_points)
    traj_xyzquat = mypfv.planall(target_points[0:3], start_point, planner_config, start_quat=start_quat, end_quat=end_quat)

    # =============================================================
    #  cxy plan end:
    # =============================================================

    # 逆解及运动部分
    # rangedik
    ik_solution180 = []
    r = RelaxedIK()
    if len(traj_xyzquat) != 0:
        for ii in range(len(traj_xyzquat)):
            solution = r.relaxed_ik.solve_position(traj_xyzquat[ii][0:3], traj_xyzquat[ii][3:], [0, 0, 0, 0, 0, 0])
            for i in range(6):
                ik_solution180.append(solution[i] / math.pi * 180)
    # time.sleep(0.03)

    points_num_org = int(len(ik_solution180) / 6)
    ik_solution180 = np.array(ik_solution180).reshape((points_num_org, 6))
    print('ik_solution180: \n', ik_solution180)'''

    '''# test for myPlan:
    urdf_name = 'rm_65'
    check_threshold = 0.1
    myplan = myPlan(traj_xyzquat, ik_solution180, urdf_name)
    checked_plan = myplan.check_plan()
    if checked_plan is not None:
        print("可以执行")

        # for real:
        '''

    '''# 连接机械臂
            callback = CANFD_Callback(mcallback)
            robot = Arm(RM65, "192.168.1.18", callback) 
            canfd_move(robot, ik_solution180[0], ik_solution180)
            return 1'''
    '''
        # for test:
        canfd_list = canfd_move(ik_solution180[0], ik_solution180)
        draw_joint_lines(canfd_list)'''






