import numpy as np
import genesis as gs
from simulation.case_simulation.case_handler import CaseHandler, register_case

@register_case("cake")
class Cake(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self.sauce_entity = None

    def add_emitters(self):
        self.scene_upper = self.simulation_upper_bound.cpu().numpy()

        obj_min = self.all_obj_occupied_lower_bound.cpu().numpy()
        obj_max = self.all_obj_occupied_upper_bound.cpu().numpy()
        
        center_x = (obj_min[0] + obj_max[0]) / 2.0
        center_y = (obj_min[1] + obj_max[1]) / 2.0
        
        # 读取刚刚在 DiffSim 中算出的真实画面顶部 Z 坐标
        # 加上一个小小的安全偏移量 (例如 0.05)，确保彻底在屏幕外
        safe_margin = 0.01
        sauce_z_bottom = self.config.get('camera_z_top', obj_max[2]) + safe_margin
        
        volume = self.config.get("sauce_volume", [0.1, 0.1, 0.1])
        
        # 从下往上构建 Box，保证液体底部刚好在画面上方
        lower_bound = (center_x - volume[0]/2, center_y - volume[1]/2, sauce_z_bottom)
        upper_bound = (center_x + volume[0]/2, center_y + volume[1]/2, self.scene_upper[2]-0.005)
        
        print(f"[PourSauceCase] 真实画面顶部 Z={self.config.get('camera_z_top')}")
        print(f"[PourSauceCase] 酱汁已安全生成在画面外: {lower_bound} 到 {upper_bound}")
        # 3. 定义液体的物理材质
        liquid_material = gs.materials.PBD.Liquid(
            rho=self.config.get('pbd_rho', 800.0),
            viscosity_relaxation=self.config.get('pbd_viscosity_relaxation', 0.05)
        )
        
        # 读取酱汁颜色，默认给个巧克力色
        sauce_color = self.config.get("sauce_color", (0.4, 0.2, 0.05, 1.0))
        
        # 4. 实体注入场景
        self.sauce_entity = self.scene.add_entity(
            material=liquid_material,
            morph=gs.morphs.Box(lower=lower_bound, upper=upper_bound),
            surface=gs.surfaces.Default(
                color=sauce_color,
                vis_mode="particle"
            )
        )