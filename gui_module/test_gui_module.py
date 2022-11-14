import dearpygui.dearpygui as dpg
import time

def check_mouse_pos(img_pos_d, mpos):
    for img_name, pos in img_pos_d.items():
        if (pos[0][0] <= mpos[0]) and (mpos[0]<pos[1][0]) and (pos[0][1] <= mpos[1]) and (mpos[1] <= pos[1][1]):
            return img_name
    return None

dpg.create_context()
def change_text(sender, app_data):
    dpg.set_value("text item", f"Mouse Button ID: {app_data}")
with dpg.window(tag="_primary_window", width=800, height=800):
    dpg.set_primary_window("_primary_window", True)


with dpg.window(label="Control", tag="_control_window", width=400, height=300, pos=(0,20)):

    # button theme
    with dpg.theme() as theme_button:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)


with dpg.texture_registry():
    _width, _height, channels, data = dpg.load_image('./sample_images/001_xzq.jpg')
    dpg.add_static_texture(_width, _height, data, tag="image_1")

    _width, _height, channels, data = dpg.load_image('./sample_images/000_jywq.jpg')
    dpg.add_static_texture(_width, _height, data, tag="image_2")




with dpg.window(width=800, height=800, tag='_select_window'):
    x_off, y_off = 350, 280
    with dpg.drawlist(width=800, height=800):
        dpg.draw_image("image_1", (100,       20      ), (350,       270      ), uv_min=(0, 0), uv_max=(1, 1))
        dpg.draw_image("image_2", (100+x_off, 20      ), (350+x_off, 270      ), uv_min=(0, 0), uv_max=(1, 1))
        dpg.draw_image("image_2", (100,       20+y_off), (350,       270+y_off), uv_min=(0, 0), uv_max=(1, 1))
        dpg.draw_image("image_2", (100+x_off, 20+y_off), (350+x_off, 270+y_off), uv_min=(0, 0), uv_max=(1, 1))

    select_img_pos_d = {
        '_select_img_1':((100,       50      ), (350,       300      )),
        '_select_img_2':((100+x_off, 50      ), (350+x_off, 300      )),
        '_select_img_3':((100,       50+y_off), (350,       300+y_off)),
        '_select_img_4':((100+x_off, 50+y_off), (350+x_off, 300+y_off)),
    }


    def callback_select_left_double_click(sender, app_data):
        if not dpg.is_item_focused("_select_window"):
            return

        mouse_pos = dpg.get_mouse_pos()
        select_img_name = check_mouse_pos(select_img_pos_d, mouse_pos)
        if select_img_name is not None:
            #TODO: change it to the corresponding URL
            print('--- select_img_name', select_img_name)
            print('--- state : ', dpg.get_item_state('_select_window'))

            dpg.configure_item('_select_window', collapsed=True )
            # dpg.configure_item('_select_window', show=True)

    with dpg.handler_registry():
        dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left, callback=callback_select_left_double_click)


dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()