# gui.py
"""The GUI for the gamma camera uniformity program
   Copyright (c) 2024 Alex Toogood-Johnson, Lyall Stewart, Josh Scates, Leah Wells, Zac Baker"""

########## IMPORTS ##########

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_ALL
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from dicom_functions import *
from uniformity_functions import UniformityLayer
from tkinter import filedialog
import os
import ast
from concurrent.futures import ProcessPoolExecutor
import time

########## CONSTANTS ##########

HEIGHT, WIDTH = 540, 960
FILEPATH = os.path.dirname(os.path.abspath(__file__))

ctk.set_default_color_theme(read_config_file("colour"))
ctk.set_appearance_mode(read_config_file("colour_theme"))

########## CLASSES ##########


class Tk(ctk.CTk, TkinterDnD.DnDWrapper):
    """Sourced from: https://stackoverflow.com/questions/75526264/using-drag-and-drop-files-or-file-picker-with-customtkinter
       Allows for drag and drop file opening within customtkinter"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


class Gui(Tk):
    """The main GUI class for the Gamma Camera Uniformity program"""

    def __init__(self, *args, **kwargs) -> None:
        Tk.__init__(self, *args, **kwargs)

        BUTTON_WIDTH = 150
        TEXTBOX_WIDTH = 200
        COMPONENT_HEIGHT = 30

        self.bind('<Left>', lambda event: self.left_button_callback())
        self.bind('<Right>', lambda event: self.right_button_callback())
        self.bind('<Up>', lambda event: self.left_button_callback())
        self.bind('<Down>', lambda event: self.right_button_callback())
        self.bind('<o>', lambda event: self.open_button_callback())
        self.bind('<s>', lambda event: self.apply_convolution())
        self.bind('<BackSpace>', lambda event: self.revert_changes())

        self.title("Gamma Camera Uniformity")
        self.geometry(f"{WIDTH}x{HEIGHT}")
        self.resizable(True, True)

        self.photo_frame = ctk.CTkFrame(self, height=HEIGHT - 20, width=HEIGHT - 20)
        self.photo_frame.place(x=10, y=10)
        self.canvas = ctk.CTkCanvas(self.photo_frame, width=HEIGHT - 20, height=HEIGHT - 20)
        self.canvas.place(x=0, y=0)
        self.canvas_image = None
        self.tk_image = None

        self.dnd_box = ctk.CTkEntry(self.photo_frame, width=HEIGHT - 20, height=HEIGHT - 20, state="readonly")
        self.dnd_box.place(x=0, y=0)
        self.dnd_box.drop_target_register(DND_ALL)
        self.dnd_box.dnd_bind("<<Drop>>", self.get_path)

        self.tab_view = ctk.CTkTabview(self, width=WIDTH - HEIGHT - 10, height=HEIGHT - 10)
        self.tab_view.place(x=HEIGHT, y=0)
        self.tab_1 = self.tab_view.add("    DICOM Operations    ")
        self.tab_2 = self.tab_view.add("   Uniformity Calculations   ")
        self.tab_3 = self.tab_view.add("    Settings    ")

        self.frame_tab_1 = ctk.CTkFrame(self.tab_1, width=WIDTH - HEIGHT - 20)
        self.frame_tab_1.pack(side='top', pady=40)
        self.combobox_values = [str(i) for i in range(1, 40)]
        self.left_button = ctk.CTkButton(self.frame_tab_1, text="Left", width=70, height=10, command=self.left_button_callback, font=("", 20), state="disabled")
        self.select_layer = ctk.CTkComboBox(self.frame_tab_1, width=190, height=10, state="disabled", font=("", 20), values=self.combobox_values, command=self.combobox_callback)
        self.right_button = ctk.CTkButton(self.frame_tab_1, text="Right", width=70, height=10, command=self.right_button_callback, font=("", 20), state="disabled")
        self.left_button.pack(side='left', padx=10)
        self.select_layer.pack(side='left', padx=10)
        self.right_button.pack(side='left', padx=10)

        self.frame_tab_2 = ctk.CTkFrame(self.tab_1, width=WIDTH - HEIGHT - 20)
        self.frame_tab_2.pack()
        self.open_button = ctk.CTkButton(self.frame_tab_2, text="   Open   ", width=30, height=20, font=("", 20), command=self.open_button_callback)
        self.close_button = ctk.CTkButton(self.frame_tab_2, text="   Close   ", width=30, height=20, font=("", 20), state="disabled", command=self.close_button_callback)
        self.save_button = ctk.CTkButton(self.frame_tab_2, text="  Save Layer  ", width=30, height=20, font=("", 20), state="disabled", command=self.save_button_callback)
        self.open_button.pack(side='left', padx=10)
        self.save_button.pack(side='left', padx=10)
        self.close_button.pack(side='left', padx=10)
        self.open_button.configure(state='normal')

        self.frame_tab_4 = ctk.CTkFrame(self.tab_1, width=WIDTH - HEIGHT - 20)
        self.frame_tab_4.pack(pady=40)
        self.apply_convolution_button = ctk.CTkButton(self.frame_tab_4, text="  Smooth  ", width=30, height=20, font=("", 20), state="disabled", command=self.apply_convolution)
        self.reduce_image_button = ctk.CTkButton(self.frame_tab_4, text="   Reduce   ", width=30, height=20, font=("", 20), state="disabled", command=self.reduce_image)
        self.revert_changes_button = ctk.CTkButton(self.frame_tab_4, text="   Revert   ", width=30, height=20, font=("", 20), state="disabled", command=self.revert_changes)
        self.apply_convolution_button.pack(side='left', padx=10)
        self.reduce_image_button.pack(side='left', padx=10)
        self.revert_changes_button.pack(side='left', padx=10)

        self.frame_tab_5 = ctk.CTkFrame(self.tab_1, width=WIDTH - HEIGHT - 20, height=190)
        self.frame_tab_5.pack(side='bottom', pady=20)
        self.logo = Image.open(os.path.join(FILEPATH, "logo.png"))
        self.logo = self.logo.resize((WIDTH - HEIGHT - 20, 190), Image.LANCZOS)
        self.tk_logo = ImageTk.PhotoImage(self.logo)
        self.logo_label = ctk.CTkLabel(self.frame_tab_5, image=self.tk_logo, text="")
        self.logo_label.pack()

        self.frame_tab_6 = ctk.CTkFrame(self.tab_3, width=WIDTH - HEIGHT - 20)
        self.frame_tab_6.pack(pady=10)
        self.about_button = ctk.CTkButton(self.frame_tab_6, text="    About   ", width=30, height=20, font=("", 20), command=lambda: AboutGUI().mainloop())
        self.help_button = ctk.CTkButton(self.frame_tab_6, text="    Help    ", width=30, height=20, font=("", 20), command=lambda: HelpGUI().mainloop())
        self.licence_button = ctk.CTkButton(self.frame_tab_6, text="   Licence   ", width=30, height=20, font=("", 20), command=lambda: LicenceGUI().mainloop())
        self.about_button.pack(side='left', padx=10)
        self.help_button.pack(side='left', padx=10)
        self.licence_button.pack(side='left', padx=10)

        self.frame_tab_7 = ctk.CTkFrame(self.tab_3, width=WIDTH - HEIGHT - 20)
        self.frame_tab_7.pack(pady=(10, 2), anchor='w')
        self.change_saving_directory_button = ctk.CTkButton(self.frame_tab_7, text="Save To", width=30, height=30, font=("", 15), command=self.save_directory_button_callback)
        dirname = read_config_file("default_file_saving_directory")
        self.saving_directory_name = ctk.CTkLabel(self.frame_tab_7, text=self.normalize_directory(dirname), font=("courier", 15))
        self.change_saving_directory_button.pack(side='left', padx=10)
        self.saving_directory_name.pack(side='left', padx=10, pady=5)

        self.frame_tab_8 = ctk.CTkFrame(self.tab_3, width=WIDTH - HEIGHT - 20)
        self.frame_tab_8.pack(pady=(2, 10), anchor='w')
        self.change_opening_directory_button = ctk.CTkButton(self.frame_tab_8, text="Open At", width=30, height=30, font=("", 15), command=self.open_directory_button_callback)
        dirname = read_config_file("default_file_opening_directory")
        self.opening_directory_name = ctk.CTkLabel(self.frame_tab_8, text=self.normalize_directory(dirname), font=("courier", 15))
        self.change_opening_directory_button.pack(side='left', padx=10)
        self.opening_directory_name.pack(side='left', padx=10, pady=10)

        self.frame_tab_9 = ctk.CTkFrame(self.tab_3, width=WIDTH - HEIGHT - 20)
        self.frame_tab_9.pack(pady=10, anchor='w')
        self.theme_menu = ctk.CTkOptionMenu(self.frame_tab_9, font=("", 15), width=BUTTON_WIDTH, command=lambda event: self.theme_changed(event), values=["Light", "Dark", "System"])
        self.theme_menu.pack(side='left', padx=10)
        self.theme_menu.set(read_config_file("colour_theme"))
        self.colour_menu = ctk.CTkOptionMenu(self.frame_tab_9, font=("", 15), width=BUTTON_WIDTH, command=lambda event: self.colour_changed(event), values=["blue", "green", "dark-blue"])
        self.colour_menu.pack(side='left', padx=10)
        self.colour_menu.set(read_config_file("colour"))

        self.frame_tab_10 = ctk.CTkFrame(self.tab_3, width=WIDTH-HEIGHT-20)
        self.frame_tab_10.pack(pady=5, anchor='w')

        self.change_default_crop_size_button = ctk.CTkButton(self.frame_tab_10, text="Update Crop Size", width=BUTTON_WIDTH, height=COMPONENT_HEIGHT, font=("", 15), command=self.change_crop_size_callback)
        self.change_default_crop_size_button.pack(side='left', padx=10)
        self.crop_size_textbox = ctk.CTkTextbox(self.frame_tab_10, width=TEXTBOX_WIDTH, height=1, font=("", 15))
        self.crop_size_textbox.insert("1.0", str(read_config_file("crop_amount")))
        self.crop_size_textbox.pack(side='left', padx=10)

        self.frame_tab_11 = ctk.CTkFrame(self.tab_3, width=WIDTH - HEIGHT - 20)
        self.frame_tab_11.pack(pady=(10, 0), anchor='w')
        self.change_fov_radius_button = ctk.CTkButton(self.frame_tab_11, text="Update FoV Radius", width=BUTTON_WIDTH, height=COMPONENT_HEIGHT, font=("", 15), command=self.change_fov_radius_callback)
        self.change_fov_radius_button.pack(side='left', padx=10)
        self.fov_radius_textbox = ctk.CTkTextbox(self.frame_tab_11, width=TEXTBOX_WIDTH, height=1, font=("", 15))
        self.fov_radius_textbox.insert("1.0", str(read_config_file("fov_radius")))

        self.fov_radius_textbox.pack(side='left', padx=10, pady=10)
        
        self.frame_tab_14 = ctk.CTkFrame(self.tab_3, width=WIDTH-HEIGHT-20)
        self.frame_tab_14.pack(pady=(0, 10), anchor='w')
        self.draw_crop_size_button = ctk.CTkButton(self.frame_tab_14, text="Draw FoV Radius", width=BUTTON_WIDTH + TEXTBOX_WIDTH + 20, height=COMPONENT_HEIGHT, font=("", 15), command=self.draw_fov_radius_callback)
        self.draw_crop_size_button.pack(side='left', padx=10, pady=10)
        self.draw_crop_size_button.configure(state="disabled")

        self.frame_tab_13 = ctk.CTkFrame(self.tab_3, width=WIDTH-HEIGHT-20)
        self.frame_tab_13.pack(anchor='w')

        self.change_step_button = ctk.CTkButton(self.frame_tab_13, text="Update Step", width=BUTTON_WIDTH, height=COMPONENT_HEIGHT, font=("", 15), command=self.change_step)
        self.change_step_button.pack(side='left', padx=10)
        self.step_textbox = ctk.CTkTextbox(self.frame_tab_13, width=TEXTBOX_WIDTH, height=1, font=("", 15))
        self.step_textbox.insert("1.0", str(read_config_file("step")))
        self.step_textbox.pack(side='left', padx=10)

        self.frame_tab_15 = ctk.CTkFrame(self.tab_3, width=WIDTH - HEIGHT - 20)
        self.frame_tab_15.pack(pady=10, anchor='w')
        self.reset_button = ctk.CTkButton(self.frame_tab_15, text="Reset to recommended settings", width=BUTTON_WIDTH + TEXTBOX_WIDTH + 20, height=30, font=("", 15), command=self.reset_settings)
        self.reset_button.pack(side='left', padx=10, pady=10)

        self.differential_uniformity_frame = ctk.CTkFrame(self.tab_2, width=WIDTH - HEIGHT - 20)
        self.differential_uniformity_frame.pack(side='top', pady=40, anchor='w')

        self.get_differential_uniformity_button = ctk.CTkButton(self.differential_uniformity_frame, text="Get Uniformity", width=30, height=20, font=("", 15), command=self.uniformity_callback)
        self.get_differential_uniformity_button.pack(side='left', padx=10, anchor='w')
        self.differential_uniformity_label = ctk.CTkLabel(self.differential_uniformity_frame, text="Differential Uniformity: ", font=("", 15))
        self.differential_uniformity_label.pack(side='left', padx=10)
        self.differential_text_output = ctk.CTkTextbox(self.tab_2, width=30, height=10, font=("", 15))
        self.differential_text_output.pack(padx=10, pady=10, expand=True, fill='both')

        self.integral_uniformity_frame = ctk.CTkFrame(self.tab_2, width=WIDTH - HEIGHT - 20)
        self.integral_uniformity_frame.pack(pady=40, anchor='w')
        self.get_integral_uniformity_button = ctk.CTkButton(self.integral_uniformity_frame, text="Get Uniformity", width=30, height=20, font=("", 15), command=self.uniformity_callback)
        self.get_integral_uniformity_button.pack(side='left', padx=10, anchor='w')
        self.integral_uniformity_label = ctk.CTkLabel(self.integral_uniformity_frame, text="Integral Uniformity: ", font=("", 15))
        self.integral_uniformity_label.pack(side='left', padx=10)
        self.integral_text_output = ctk.CTkTextbox(self.tab_2, width=30, height=10, font=("", 15))
        self.integral_text_output.pack(padx=10, pady=10, expand=True, fill='both')

        self.get_differential_uniformity_button.configure(state="disabled")
        self.get_integral_uniformity_button.configure(state="disabled")

    def change_step(self) -> None:
        step_size = self.step_textbox.get("1.0", "end-1c")
        if step_size.isdigit() and int(step_size) > 0 and int(step_size) < 40:
            edit_config_file("crop", int(step_size))
        else:
            self.step_size.delete("1.0", "end")
            self.step_size.insert("1.0", str(read_config_file("step")))

    def reset_settings(self) -> None:
        self.crop_size_textbox.delete("1.0", "end")
        self.crop_size_textbox.insert("1.0", "40")
        self.fov_radius_textbox.delete("1.0", "end")
        self.fov_radius_textbox.insert("1.0", "80")
        self.step_textbox.delete("1.0", "end")
        self.step_textbox.insert("1.0", "2")
        self.change_crop_size_callback()

    def uniformity_callback(self) -> None:
        self.cropped_dicom_image = apply_convolution(self.current_dicom_image, read_config_file("convolution"))
        self.cropped_dicom_image = crop(self.current_dicom_image, read_config_file("crop_amount"))

        self.differential_text_output.insert("1.0", "Loading...")
        self.integral_text_output.insert("1.0", "Loading...")

        differential_vals = []
        integral_vals = []

        with ProcessPoolExecutor() as executor:
            future_to_type = {}
            for layer_index in range(0, read_config_file("crop_amount") - 1, 2):
                dicom_slice = self.current_dicom_image[layer_index]
                normalized_slice = ((dicom_slice - np.min(dicom_slice)) / (np.max(dicom_slice) - np.min(dicom_slice)) * 255).astype(np.uint8)

                image = Image.fromarray(normalized_slice)
                image = image.convert("RGB")
                image = image.resize((HEIGHT - 20, HEIGHT - 20), Image.LANCZOS)

                layer = UniformityLayer(np.array(image), read_config_file("fov_radius"))
                layer.crop_to_circle()

                future_to_type[executor.submit(layer.differential)] = 'differential'
                future_to_type[executor.submit(layer.integral)] = 'integral'

            start_time = time.time()
            for future, type_label in future_to_type.items():
                result = future.result()
                if type_label == 'differential':
                    differential_vals.append(result)
                else:
                    integral_vals.append(result)
            end_time = time.time()

        if differential_vals:
            self.display_uniformity(integral_vals, differential_vals)
        else:
            print("No differential values to display.")

        print(f"Total Execution Time: {end_time - start_time} seconds")

    def display_uniformity(self, integral: int, differential: int) -> None:
        self.differential_uniformity_results = differential
        self.differential_uniformity_label.configure(text=f" Differential Uniformity: {round(max(self.differential_uniformity_results), 2)}%")
        fov = read_config_file("fov_radius")
        convolution = read_config_file("convolution")
        textbox_text = f"Differential Uniformity Results:\nFoV Radius {fov} px \nConvolution: {convolution}\n\n"
        for i, result in enumerate(self.differential_uniformity_results):
            textbox_text += f"Layer {(i+1) * read_config_file('step')}: {result}\n"
        self.differential_text_output.delete("1.0", "end")
        self.differential_text_output.insert("1.0", textbox_text)

        self.integral_uniformity_results = integral
        self.integral_uniformity_label.configure(text=f"Integral Uniformity: {round(max(self.integral_uniformity_results), 2)}%")
        fov = read_config_file("fov_radius")
        convolution = read_config_file("convolution")
        textbox_text = f"Integral Uniformity Results:\nFoV Radius {fov} px \nConvolution: {convolution}\n\n"
        for i, result in enumerate(self.integral_uniformity_results):
            textbox_text += f"Layer {(i+1) * read_config_file('step')}: {result}\n"
        self.integral_text_output.delete("1.0", "end")
        self.integral_text_output.insert("1.0", textbox_text)

    def change_crop_size_callback(self) -> None:
        """Changes the crop size stored in the config file"""

        crop_size = self.crop_size_textbox.get("1.0", "end-1c")
        if crop_size.isdigit() and int(crop_size) > 0 and int(crop_size) < 100:
            edit_config_file("crop_amount", int(crop_size))
        else:
            self.crop_size_textbox.delete("1.0", "end")
            self.crop_size_textbox.insert("1.0", str(read_config_file("crop_amount")))

    def change_fov_radius_callback(self) -> None:
        """Changes the field of view radius stored in the config file"""
        fov_rad = self.fov_radius_textbox.get("1.0", "end-1c")
        if fov_rad.isdigit() and int(fov_rad) > 0 and int(fov_rad) < 100:
            edit_config_file("fov_radius", int(fov_rad))
        else:
            self.fov_radius_textbox.delete("1.0", "end")
            self.fov_radius_textbox.insert("1.0", str(read_config_file("fov_radius")))

    def draw_fov_radius_callback(self) -> None:
        """Draws the field of view radius on the image"""
        self.display_image(int(self.select_layer.get()), int(self.fov_radius_textbox.get("1.0", "end")))

    def colour_changed(self, event) -> None:
        """Changes the colour of the GUI"""
        colour = self.colour_menu.get()
        ctk.set_default_color_theme(colour)
        edit_config_file("colour", colour)

    def theme_changed(self, event) -> None:
        """Changes the colour theme of the GUI"""
        theme = self.theme_menu.get()
        ctk.set_appearance_mode(theme)
        edit_config_file("colour_theme", theme)

    def save_directory_button_callback(self) -> None:
        """Changes the directory stored in the config file for saving files"""
        directory = filedialog.askdirectory()
        if directory:
            if os.path.isdir(directory):
                self.saving_directory_name.configure(text=self.normalize_directory(directory))
                edit_config_file("default_file_saving_directory", directory)

    def normalize_directory(self, directory: str) -> str:
        """Makes all directory names 30 character long for display purposes"""
        if len(directory) >= 25:
            directory = '"...' + directory[-25:] + '"'
        else:
            directory = '"' + directory + '"' + ' ' * (28 - len(directory))
        return directory

    def open_directory_button_callback(self) -> None:
        """Changes the directory stored in the config file for opening files"""
        directory = filedialog.askdirectory()
        if directory:
            if os.path.isdir(directory):
                self.opening_directory_name.configure(text=self.normalize_directory(directory))
                edit_config_file("default_file_opening_directory", directory)

    def reduce_image(self) -> None:
        """Reduces the image by the amount specified in the config file, and displays in the GUI"""
        if self.reduce_image_button.cget("state") != 'normal': return
        self.current_dicom_image = crop(self.current_dicom_image, read_config_file("crop_amount"))
        self.reduce_image_button.configure(state="disabled")
        self.apply_convolution_button.configure(state="disabled")
        self.revert_changes_button.configure(state="normal")
        self.select_layer.set("1")
        self.left_button.configure(state="disabled")
        self.combobox_values = [str(i) for i in range(1, read_config_file("crop_amount") - 1)]
        self.display_image(int(self.select_layer.get()))

    def apply_convolution(self) -> None:
        """Applies a convolution to the image, and displays in the GUI"""
        if self.apply_convolution_button.cget("state") != 'normal': return
        self.current_dicom_image = apply_convolution(self.current_dicom_image, read_config_file("convolution"))
        self.display_image(int(self.select_layer.get()))
        self.apply_convolution_button.configure(state="disabled")
        self.reduce_image_button.configure(state="normal")
        self.revert_changes_button.configure(state="normal")

    def revert_changes(self) -> None:
        """Reverts the visual changes made to the image - convolution and/or cropping"""
        if self.revert_changes_button.cget("state") != 'normal': return
        self.current_dicom_image = self.original_dicom_image
        self.display_image(int(self.select_layer.get()))
        self.reduce_image_button.configure(state="disabled")
        self.apply_convolution_button.configure(state="normal")
        self.revert_changes_button.configure(state="disabled")

    def combobox_callback(self, choice: str) -> None:
        """Callback for selection of layer number from combobox"""
        if int(choice) == 1:
            self.left_button.configure(state="disabled")
            self.right_button.configure(state="normal")
        elif choice == self.combobox_values[-1]:
            self.left_button.configure(state="normal")
            self.right_button.configure(state="disabled")
        else:
            self.left_button.configure(state="normal")
            self.right_button.configure(state="normal")
        self.display_image(int(choice))

    def left_button_callback(self) -> None:
        """Callback for left button press on layer selection"""
        if self.left_button.cget("state") != 'normal': return
        self.select_layer.set(int(self.select_layer.get()) - 1)
        if int(self.select_layer.get()) == 1:
            self.left_button.configure(state="disabled")
        if self.select_layer.get() != self.combobox_values[-1]:
            self.right_button.configure(state="normal")
        self.display_image(int(self.select_layer.get()))

    def right_button_callback(self):
        """Callback for the right button press on layer selection"""
        if self.right_button.cget("state") != 'normal': return
        self.select_layer.set(int(self.select_layer.get()) + 1)
        if int(self.select_layer.get()) > 1:
            self.left_button.configure(state="normal")
        if self.select_layer.get() == self.combobox_values[-1]:
            self.right_button.configure(state="disabled")
        self.display_image(int(self.select_layer.get()))

    def open_button_callback(self) -> None:
        """Callback for the open button press, opens a file dialog to select a DICOM file to open"""
        if self.open_button.cget("state") != 'normal': return
        default = read_config_file("default_file_opening_directory")
        if os.path.isdir(default):
            file_path = filedialog.askopenfilename(filetypes=[("DICOM Files", "*.dcm")], initialdir=default)
        else:
            file_path = filedialog.askopenfilename(filetypes=[("DICOM Files", "*.dcm")], initialdir=os.getcwd())
        if file_path:
            self.title(f"Gamma Camera Uniformity - {file_path.split('/')[-1]}")
            self.fit_dicom_image(file_path)

    def close_button_callback(self) -> None:
        """Callback for the close button press, resets the GUI to its initial state"""
        self.current_dicom_image = None
        self.original_dicom_image = None
        self.combobox_values = []
        self.select_layer.set("")
        self.left_button.configure(state="disabled")
        self.right_button.configure(state="disabled")
        self.select_layer.configure(state="disabled")
        self.open_button.configure(state="normal")
        self.save_button.configure(state="disabled")
        self.close_button.configure(state="disabled")
        self.reduce_image_button.configure(state="disabled")
        self.apply_convolution_button.configure(state="disabled")
        self.revert_changes_button.configure(state="disabled")
        self.get_integral_uniformity_button.configure(state="disabled")
        self.get_differential_uniformity_button.configure(state="disabled")
        self.differential_text_output.delete("1.0", "end")
        self.integral_text_output.delete("1.0", "end")
        self.differential_uniformity_label.configure(text="Uniformity: ")
        self.integral_uniformity_label.configure(text="Uniformity: ")
        self.title("Gamma Camera Uniformity")
        self.draw_crop_size_button.configure(state="disabled")
        self.canvas.delete(self.canvas_image)
        self.dnd_box.lift()

    def save_button_callback(self) -> None:
        """Callback for the save button press, saves the currently viewed layer as a PNG file"""
        dicom_slice = self.current_dicom_image[int(self.select_layer.get()) - 1]
        normalized_slice = ((dicom_slice - np.min(dicom_slice)) / (np.max(dicom_slice) - np.min(dicom_slice)) * 255).astype(np.uint8)
        default = read_config_file("default_file_saving_directory")
        if os.path.isdir(default):
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")], initialdir=default)
        else:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")], initialdir=os.getcwd())
        if file_path:
            Image.fromarray(normalized_slice).save(file_path)

    def get_path(self, event) -> None:
        image_path = event.data.replace("{", "").replace("}", "")
        if image_path.endswith("dcm"):
            self.fit_dicom_image(image_path)

    def fit_dicom_image(self, image_path) -> None:
        self.current_dicom_image = load_dicom_image(image_path)[1]
        if not len(self.current_dicom_image.shape) == 3:
            self.current_dicom_image = None
            return  # Rejects 2D images
        self.original_dicom_image = self.current_dicom_image
        self.combobox_values = [str(i) for i in range(1, self.current_dicom_image.shape[0] + 1)]
        self.right_button.configure(state="normal")
        self.select_layer.configure(state="normal")
        self.select_layer.set("1")
        self.open_button.configure(state="disabled")
        self.save_button.configure(state="normal")
        self.close_button.configure(state="normal")
        self.apply_convolution_button.configure(state="normal")
        self.get_integral_uniformity_button.configure(state="normal")
        self.get_differential_uniformity_button.configure(state="normal")
        self.draw_crop_size_button.configure(state="normal")
        self.display_image(1)

    def display_image(self, layer: int, fov_rad: int = 0) -> None:
        dicom_slice = self.current_dicom_image[layer - 1]
        normalized_slice = ((dicom_slice - np.min(dicom_slice)) / (np.max(dicom_slice) - np.min(dicom_slice)) * 255).astype(np.uint8)

        image = Image.fromarray(normalized_slice)
        image = image.convert("RGB")
        image = image.resize((HEIGHT - 20, HEIGHT - 20), Image.LANCZOS)
        if fov_rad > 0:  # Providing a FoV of 0 indicates that the user does not want a FoV drawn
            draw = ImageDraw.Draw(image)
            center_x, center_y = (HEIGHT - 20) // 2, (HEIGHT - 20) // 2
            top_left = (center_x - fov_rad, center_y - fov_rad)
            bottom_right = (center_x + fov_rad, center_y + fov_rad)
            draw.ellipse([top_left, bottom_right], outline="red", width=2)

        self.tk_image = ImageTk.PhotoImage(image)
        if self.canvas_image:
            self.canvas.delete(self.canvas_image)
        self.canvas_image = self.canvas.create_image(-1, -1, anchor="nw", image=self.tk_image, tags="image")
        self.canvas.lift(self.canvas_image)
        self.dnd_box.lower()


class LicenceGUI(Tk):
    def __init__(self, *args, **kwargs) -> None:
        Tk.__init__(self, *args, **kwargs)

        self.title("Licence")
        self.geometry("500x300")
        self.resizable(False, False)

        self.licence_text = ctk.CTkTextbox(self, width=480, height=280)
        self.licence_text.place(x=10, y=10)
        self.licence_text.insert("1.0", open(os.path.join(FILEPATH, "LICENCE")).read())
        self.licence_text.configure(state="disabled")


class AboutGUI(Tk):
    def __init__(self, *args, **kwargs) -> None:
        Tk.__init__(self, *args, **kwargs)
        self.title("About")
        self.geometry("500x300")
        self.resizable(False, False)
        text = """
Gamma Camera Uniformity Program
Developed at Exeter Maths School as part of the Exeter Maths Certificate
in 2024.
The purpose of this software is to perform calculations to determine the
uniformity values of DICOM SPECT images taken from a gamma camera.

For more information, and to view the source code, visit the GitHub page:
https://github.com/ExeMS/NHS-EMC2024

This software is distributed under the MIT License.
Copyright (c) 2024 Alex Toogood-Johnson, Lyall Stewart, Josh Scates, Leah Wells, Zac Baker.

               """
        self.about_text = ctk.CTkTextbox(self, width=480, height=280)
        self.about_text.place(x=10, y=10)
        self.about_text.insert("1.0", text)
        self.about_text.configure(state="disabled")


class HelpGUI(Tk):
    def __init__(self, *args, **kwargs) -> None:
        Tk.__init__(self, *args, **kwargs)
        self.title("Help")
        self.geometry("500x300")
        self.resizable(False, False)
        text = """        
This section provides some information on how to use this program, although for further details please contact the authors. To start, open a DICOM file by selecting it from the file directory, which you can access by typing 'o' on your keyboard or by clicking the 'open' button. You can also directly open a file by holding it over the left half of the program, which functions as a drag and drop box as long as there isn't currently a file open. 

The chosen file must be a .dcm file, and must be a SPECT image in 3 dimensions in order to be opened by the program. Once an image has been opened, you can close the image by clicking the 'close' button. In order to view each layer of the DICOM image, you can either click the left, right, up or down buttons on your keyboard, select the left or right buttons on the gui, or manually select a layer from the combobox. Please note that the first and last 20-40 layers in a DICOM uniformity image may be blank or incomplete. 

The 'reduce' button, which you can either click, or press 'r' on your keyboard, fixes this by removing the first n and last n layers of the image. The default and recommended value is 44, although this can be changed in the settings tab. 

The smooth button, which you can either click or press 's' on your keyboard, applies a 9 point weighted convolution in order to smooth each layer of the DICOM image. The convolution used is [[1,2,1],[2,4,2],[1,2,1]] as specified in the NEMA handbook. We strongly recommend not changing this, however if necessary than this can be manually changed inside 'config.json'.

The revert button, which you can either click, or press backspace on your keyboard, visually reverts the smoothing and / or cropping changes to the DICOM image. The save layer button saves the currently selected layer of the DICOM image as a PNG file. The directory at which this opens, as well as the directory which is opened when you want to open a DICOM file, can be specified in the settings tab. 

In the middle tab of the program, there are options to calculate integral and differential uniformity, although please note that pressing one of these buttons also triggers the other. Due to a high number of calculations in order to find the uniformities, this may take up to a minute, during which time the program may not respond. In the labels you will be able to see the overall differential or integral uniformity, whereas in the larger text box there will be details about each layer in the DICOM image. In the third tab, which contains the settings for the program, there are buttons linking to the about, help and licence sections. Add a bit more stuff here.
"""
        self.about_text = ctk.CTkTextbox(self, width=480, height=280)
        self.about_text.place(x=10, y=10)
        self.about_text.insert("1.0", text)
        self.about_text.configure(state="disabled")

########## MAIN ##########


if __name__ == "__main__":
    app = Gui()
    app.mainloop()
