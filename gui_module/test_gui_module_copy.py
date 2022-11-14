# import dearpygui.dearpygui as dpg
# dpg.create_context()
# def toggle_layer2(sender):
#     show_value = dpg.get_value(sender)
#     dpg.configure_item("layer2", show=show_value)
# with dpg.window(label="Tutorial"):
#     dpg.add_checkbox(label="show layer", callback=toggle_layer2, default_value=True)
#     with dpg.drawlist(width=300, height=300):
#         with dpg.draw_layer():
#             dpg.draw_line((10, 10), (100, 100), color=(255, 0, 0, 255), thickness=1)
#             dpg.draw_text((0, 0), "Origin", color=(250, 250, 250, 255), size=15)
#             dpg.draw_arrow((50, 70), (100, 65), color=(0, 200, 255), thickness=1,size=10)
#         with dpg.draw_layer(tag="layer2"):
#             dpg.draw_line((10, 60), (100, 160), color=(255, 0, 0, 255), thickness=1)
#             dpg.draw_arrow((50, 120), (100, 115), color=(0, 200, 255), thickness=1,size=10)
# dpg.create_viewport(title='Custom Title', width=800, height=600)
# dpg.setup_dearpygui()
# dpg.show_viewport()
# dpg.start_dearpygui()
# dpg.destroy_context()

import dearpygui.dearpygui as dpg
dpg.create_context()
with dpg.window(label="Main"):
    with dpg.menu_bar():
        with dpg.menu(label="Themes"):
            dpg.add_menu_item(label="Dark")
            dpg.add_menu_item(label="Light")
            dpg.add_menu_item(label="Classic")
            with dpg.menu(label="Other Themes"):
                dpg.add_menu_item(label="Purple")
                dpg.add_menu_item(label="Gold")
                dpg.add_menu_item(label="Red")
        with dpg.menu(label="Tools"):
            dpg.add_menu_item(label="Show Logger")
            dpg.add_menu_item(label="Show About")
        with dpg.menu(label="Oddities"):
            dpg.add_button(label="A Button")
            dpg.add_simple_plot(label="Menu plot", default_value=(0.3, 0.9, 2.5, 8.9),height=80)


dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()