from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("snowman")
class Snowman(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        pass

    def fix_particles(self):
        # ############################ Fix Particles ###############################
        # # 对于雪人，我们只固定最底部的极小一部分（"扎根"在地上），防止它整体平移滑动
        # for i in range(len(self.all_objs)):
        #     sim_particles = torch.tensor(self.all_objs[i].init_particles).to(self.device)
            
        #     # 获取该物体的 Z 轴（高度）范围
        #     z_min = sim_particles[:, 2].min().item()
        #     z_max = sim_particles[:, 2].max().item()
            
        #     # 设定阈值：只固定最底下 5% 的粒子
        #     z_threshold = z_min + (z_max - z_min) * 0.05

        #     print(f"[fix_particles] obj {i}: anchoring base. z_min={z_min:.4f}, z_threshold={z_threshold:.4f}")

        #     # 筛选出贴近地面的粒子
        #     fixed_area_idx = torch.where(sim_particles[:, 2] < z_threshold)[0]
        #     fixed_area_points = sim_particles[fixed_area_idx]
            
        #     print(f"[fix_particles] obj {i}: found {len(fixed_area_points)} particles at the base to fix")
            
        #     fixed_area_list = [tuple(point.tolist()) for point in fixed_area_points]
        #     for point in fixed_area_list:
        #         self.all_objs[i].fix_particle(self.all_objs[i].find_closest_particle(point), 0)
                
        #     print(f"[fix_particles] obj {i}: fixed {len(fixed_area_list)} base particles")
        pass

    def create_force_fields(self):
        if self.config.get('skip_force_fields', False):
            return

        # 获取场景中物体的整体高度边界
        z_min = self.all_obj_occupied_lower_bound[2].cpu().numpy()
        z_max = self.all_obj_occupied_upper_bound[2].cpu().numpy()

        @ti.func
        def force_func(pos, vel, t, i):
            # 1. 定义风向：右上往左下
            # X 为负（向左），Z 为负（向下压），Y 保持 0（不往屏幕前后飘）
            # direction = ti.Vector([-1.0, 0, -0.5], dt=gs.ti_float)
            direction = ti.Vector([0.5, -1.5, -1.5], dt=gs.ti_float)
            direction = direction.normalized()

            # 2. 定义随时间变化的风力（阵风袭来）
            # 假设 dt=0.01，这里让风力在模拟的前期迅速增强到最大并保持
            time_sec = t / 100.0  # 根据实际的 t 步长进行缩放
            strength_base = 2500.0   # 最大风力强度 (可能需要根据你雪人的质量 MPM_rho 微调)
            
            # 0 到 1 之间平滑过渡，模拟风越来越大
            gust_factor = ti.min(time_sec * 1.5, 1.0) 

            # 3. 定义随高度变化的风力（头部受风更大，产生推倒的力矩）
            height_scaler = 0.0
            if pos[2] > z_min:
                # 归一化高度 (0 到 1)
                normalized_h = (pos[2] - z_min) / (z_max - z_min)
                # 使用指数让头部的受力明显大于身体，促使雪人“折断”或“坍塌”
                height_scaler = normalized_h + 0.2
            height_scaler = 1

            # 4. 计算最终加速度 (F=ma，Taichi 这里的 force_field 实际赋予的是加速度)
            acc = direction * strength_base * gust_factor * height_scaler
            
            return acc

        force_field = gs.force_fields.Custom(force_func)
        force_field.activate()
        self.scene.add_force_field(
            force_field = force_field
        )
    