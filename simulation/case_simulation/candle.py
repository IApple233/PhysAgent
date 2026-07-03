from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("candle")
class Candle(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def create_force_fields(self):
        @ti.func
        def smoke_force_func(pos, vel, t, i):
            # 1. 抵消全局重力
            anti_gravity = ti.Vector([0.0, 0.0, 9.8], dt=gs.ti_float)
            
            # 2. 基础浮力：不给死数，而是随着位置和时间有微小的强弱变化
            buoyancy_z = 2.5 + ti.sin(t * 2.0 + pos[0] * 4.0) * 0.5
            
            # 3. 多频扰动 (Fake Turbulence)
            # 频率1：大尺度的主导风向摇摆（慢而大）
            wind_x1 = ti.sin(t * 1.5 + pos[2] * 3.0) * 1.2
            wind_y1 = ti.cos(t * 1.2 + pos[2] * 2.5) * 1.2
            
            # 频率2：小尺度的细节卷曲（快而小，制造灵动的撕裂感）
            wind_x2 = ti.sin(t * 4.0 + pos[1] * 8.0) * 0.6
            wind_y2 = ti.cos(t * 4.5 + pos[0] * 8.0) * 0.6
            
            # 将多频率的扰动合并
            turb_x = wind_x1 + wind_x2
            turb_y = wind_y1 + wind_y2
            
            # 4. 空气阻力 (Drag)
            # 给速度一个反向的微小惩罚，这会让粒子有“推不开空气”的迟滞感，非常关键！
            drag_x = -0.5 * vel[0]
            drag_y = -0.5 * vel[1]
            drag_z = -0.2 * vel[2] # Z轴阻力小一点，让它能顺利上升
            
            # 最终合成的力
            final_force_x = turb_x + drag_x
            final_force_y = turb_y + drag_y
            final_force_z = buoyancy_z + drag_z
            
            wind_and_drag = ti.Vector([final_force_x, final_force_y, final_force_z], dt=gs.ti_float)
            
            return anti_gravity + wind_and_drag

        force_field = gs.force_fields.Custom(smoke_force_func)
        self.scene.add_force_field(force_field)

    def custom_simulation(self, sid):
        pass
        
    def fix_particles(self):
        pass

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)