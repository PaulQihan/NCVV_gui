import math
import torch
import time
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from .dnerf_utils import *


def print_time(seconds):
    return time.strftime("%M:%S", time.localtime(seconds))

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])
    

class NeRFGUI:
    def __init__(self, opt, render_func=None, trainer=None, train_loader=None, debug=True, \
                max_frame=100, start_frame=0, decoder_fps=20):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        self.training = False
        self.step = 0 # training step 

        self.trainer = trainer
        self.train_loader = train_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']


        self.dynamic_resolution = False
        self.downscale = 1
        self.train_steps = 16


        self.render_func=render_func
        self.max_frame = max_frame


        self.time = 0.
        self.decoder_fps = decoder_fps
        self.max_time = max_frame / decoder_fps

        
        self.play_video = False


        dpg.create_context()
        self.register_dpg()
        # self.test_step()

    def __del__(self):
        dpg.destroy_context()



    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = self.trainer.train_gui(self.train_loader, step=self.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            depth = outputs['depth']
            depth = 1 - depth / np.max(depth)
            return np.expand_dims(depth, -1).repeat(3, -1)

    
    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if self.need_update or self.spp < self.opt.max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            # outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H, self.time, self.bg_color, self.spp, self.downscale)

            # print('------------------------------------------ down_scale : ', self.downscale)

            outputs  = self.render_func(self.cam.pose, self.cam.intrinsics, self.W, self.H, self.time)
            
            # outputs['image'] = np.ones((720, 720, 3)) * 256
            # outputs = np.ones((720, 720, 3)) * 256

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = self.prepare_buffer(outputs)
                # self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            # dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            # dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_log_bitrate", f"{outputs['bitrate']} KB/frame")
            dpg.set_value("_log_IP", outputs['ip'])
            dpg.set_value("_log_URL", outputs['url'])
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):

        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            # add the texture
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)


        # select window
        # def change_text(sender, app_data):
        #     dpg.set_value("text item", f"Mouse Button ID: {app_data}")
        # with dpg.window(label="Select", tag="_select_window", width=800, height=800):
        #     dpg.add_text("Click me with any mouse button", tag="text item")
        #     dpg.add_image("_texture")

        # with dpg.item_handler_registry(tag="widget handler") as handler:
        #     dpg.add_item_clicked_handler(callback=change_text)
        #     dpg.bind_item_handler_registry("text item", "widget handler")



        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300, pos=(0,20)):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not self.opt.test:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")              

            with dpg.group(horizontal=True):
                dpg.add_text("Render time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            # with dpg.group(horizontal=True):
            #     dpg.add_text("SPP: ")
            #     dpg.add_text("1", tag="_log_spp")

            with dpg.group(horizontal=True):
                dpg.add_text("Bitrate: ")
                dpg.add_text("1", tag="_log_bitrate")


            with dpg.group(horizontal=True):
                dpg.add_text("IP: ")
                dpg.add_text("1", tag="_log_IP")

            with dpg.group(horizontal=True):
                dpg.add_text("URL: ")
                dpg.add_text("1", tag="_log_URL")

            # time slider
            def callback_set_time(sender, app_data):
                self.time = app_data
                dpg.set_value('_time_text', f"Time:   {print_time(int(self.time))}")
                self.need_update = True
                # dpg.set_item_label('time_slider', print_time(self.time))





            def update_time_text():
                time = int(self.app_data)
                dpg.set_value('_time_text', f"Time:   {print_time(self.time)}")
                self.need_update = True                


            dpg.add_text(tag='_time_text', label="Time", default_value="Time:   00:00")
            with dpg.group(horizontal=True):                
                dpg.add_slider_int(tag='_time_slider', label="", min_value=0, no_input=False, max_value=self.max_time, format="", default_value=self.time, callback=callback_set_time)
                def callback_set_play_video(sender, app_data):
                    pass
                    if self.play_video:
                        self.play_video = False
                    else:
                        self.play_video = True
                    self.need_update = True

                dpg.add_checkbox(label="Play", default_value=self.play_video, callback=callback_set_play_video)


            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode, callback=callback_change_mode)


                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")
        with dpg.texture_registry():
            img_path_1 = '/home/vrlab/workspace/NCVV/NCVV/gui_module//sample_images/001_xzq.jpg'
            _width, _height, channels, data = dpg.load_image(img_path_1)
            dpg.add_static_texture(_width, _height, data, tag="image_1")

            img_path_2 = '/home/vrlab/workspace/NCVV/NCVV/gui_module//sample_images/000_jywq.jpg'
            _width, _height, channels, data = dpg.load_image(img_path_2)
            dpg.add_static_texture(_width, _height, data, tag="image_2")

            img_path_3 = '/home/vrlab/workspace/NCVV/NCVV/gui_module//sample_images/000_jywq.jpg'
            _width, _height, channels, data = dpg.load_image(img_path_3)
            dpg.add_static_texture(_width, _height, data, tag="image_3")
        
            img_path_4 = '/home/vrlab/workspace/NCVV/NCVV/gui_module//sample_images/000_jywq.jpg'
            _width, _height, channels, data = dpg.load_image(img_path_4)
            dpg.add_static_texture(_width, _height, data, tag="image_4")

        self.select_img_pos_d = {}

        with dpg.window(label="Select", tag="_select_window", width=800, height=800, no_move=True ):
            x_off, y_off = 350, 350
            with dpg.drawlist(width=800, height=800):
                dpg.draw_image("image_1", (100,       50      ), (350,       300      ), uv_min=(0, 0), uv_max=(1, 1), tag="_select_img_1")
                dpg.draw_image("image_2", (100+x_off, 50      ), (350+x_off, 300      ), uv_min=(0, 0), uv_max=(1, 1), tag="_select_img_2")
                dpg.draw_image("image_3", (100,       50+y_off), (350,       300+y_off), uv_min=(0, 0), uv_max=(1, 1), tag="_select_img_3")
                dpg.draw_image("image_4", (100+x_off, 50+y_off), (350+x_off, 300+y_off), uv_min=(0, 0), uv_max=(1, 1), tag="_select_img_4")
            
            self.select_img_pos_d = {
                '_select_img_1':((100,       50      ), (350,       300      )),
                '_select_img_2':((100+x_off, 50      ), (350+x_off, 300      )),
                '_select_img_3':((100,       50+y_off), (350,       300+y_off)),
                '_select_img_4':((100+x_off, 50+y_off), (350+x_off, 300+y_off)),
            }

        def check_mouse_pos(img_pos_d, mpos):
            for img_name, pos in img_pos_d.items():
                if (pos[0][0] <= mpos[0]) and (mpos[0]<pos[1][0]) and (pos[0][1] <= mpos[1]) and (mpos[1] <= pos[1][1]):
                    return img_name
            return None

        ### register select handler
        def callback_select_left_double_click(sender, app_data):
            if not dpg.is_item_focused("_select_window"):
                return

            mouse_pos = dpg.get_mouse_pos()
            select_img_name = check_mouse_pos(self.select_img_pos_d, mouse_pos)
            if select_img_name is not None:
                #TODO (wzy): change it to the corresponding URL
                
                #TODO (wzy): add a flag to disable rendering before selecting an object

                dpg.configure_item('_select_window', collapsed=True )



            # self.need_update = True



        ### register camera handler
        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

            dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left, callback=callback_select_left_double_click)
            # dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left, callback=callback_select_left_double_click)


        dpg.create_viewport(title='NCVV-gui', width=self.W, height=self.H, resizable=False)
        
        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def update_time_slider_text(self):
        time = int(self.time)
        dpg.set_value('_time_text', f"Time:   {print_time(time)}")
        dpg.set_value('_time_slider', time)
    
    def play_video_func(self):
        time_step = 1.0/float(self.decoder_fps)
        while(self.play_video):
            self.time += time_step
            if self.time >= self.max_time:
                self.time = 0.0
            self.update_time_slider_text()       
            self.need_update = True
            self.test_step()
            dpg.render_dearpygui_frame()

    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.play_video:
                self.play_video_func()
            else:
                self.test_step()
                dpg.render_dearpygui_frame()