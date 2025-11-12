import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import shutil

class CrackLabeler:
    def __init__(self, master, image_path, save_path):
        self.master = master
        self.master.title(f"Crack Labeler: {os.path.basename(image_path)}")
        self.image_path = image_path
        self.save_path = save_path

        # Load image
        self.orig_image = Image.open(self.image_path).convert("L")
        self.tk_image = ImageTk.PhotoImage(self.orig_image)
        self.width, self.height = self.orig_image.size

        # Canvas
        self.canvas = tk.Canvas(master, width=self.width, height=self.height)
        self.canvas.pack()
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # Crack data
        self.cracks = []
        self.preview_line = None  # for temporary line preview
        self.current_start = None
        self.draw_mode = True

        # Buttons
        button_frame = tk.Frame(master)
        button_frame.pack(fill=tk.X)
        self.draw_btn = tk.Button(button_frame, text="Draw", command=self.set_draw_mode)
        self.draw_btn.pack(side=tk.LEFT)
        self.delete_btn = tk.Button(button_frame, text="Delete", command=self.set_delete_mode)
        self.delete_btn.pack(side=tk.LEFT)
        self.submit_btn = tk.Button(button_frame, text="Submit", command=self.submit)
        self.submit_btn.pack(side=tk.LEFT)

        # Bind mouse
        self.canvas.bind("<Button-1>", self.left_click)
        self.canvas.bind("<Motion>", self.mouse_move)
        self.canvas.bind("<Button-3>", self.right_click)

        self.finished = False  # Flag when labeling finished

    def set_draw_mode(self):
        self.draw_mode = True
        self.current_start = None

    def set_delete_mode(self):
        self.draw_mode = False
        self.current_start = None

    def left_click(self, event):
        x, y = event.x, event.y
        if self.draw_mode:
            if self.current_start is None:
                # Start drawing
                self.current_start = (x, y)
            else:
                # Finalize the line
                self.cracks.append((self.current_start, (x, y)))
                self.canvas.create_line(*self.current_start, x, y, fill="black", width=12)
                self.current_start = None

                # Remove preview line after confirming
                if self.preview_line is not None:
                    self.canvas.delete(self.preview_line)
                    self.preview_line = None
        else:
            tol = 5
            for crack in self.cracks:
                (x0, y0), (x1, y1) = crack
                if self.point_near_line((x, y), (x0, y0), (x1, y1), tol):
                    self.cracks.remove(crack)
                    self.redraw_canvas()
                    break


    def mouse_move(self, event):
        """Show a temporary preview line while drawing."""
        if self.draw_mode and self.current_start is not None:
            x, y = event.x, event.y

            # Remove old preview line
            if self.preview_line is not None:
                self.canvas.delete(self.preview_line)

            # Draw new preview line
            self.preview_line = self.canvas.create_line(
                *self.current_start, x, y, fill="gray", dash=(4, 2), width=4
            )


    def right_click(self, event):
        if self.draw_mode and self.current_start is not None:
            self.current_start = None

    def point_near_line(self, p, a, b, tol):
        px, py = p
        x0, y0 = a
        x1, y1 = b
        if (x0 == x1) and (y0 == y1):
            return np.hypot(px-x0, py-y0) <= tol
        t = max(0, min(1, ((px-x0)*(x1-x0)+(py-y0)*(y1-y0)) / ((x1-x0)**2 + (y1-y0)**2)))
        closest = (x0 + t*(x1-x0), y0 + t*(y1-y0))
        return np.hypot(px-closest[0], py-closest[1]) <= tol

    def redraw_canvas(self):
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(self.orig_image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        for crack in self.cracks:
            (x0, y0), (x1, y1) = crack
            self.canvas.create_line(x0, y0, x1, y1, fill="black", width=12)

    def submit(self):
        label_img = Image.new("L", (self.width, self.height), color=255)
        draw = ImageDraw.Draw(label_img)
        for (x0, y0), (x1, y1) in self.cracks:
            draw.line([x0, y0, x1, y1], fill=0, width=16)
        label_img.save(self.save_path)
        self.finished = True
        self.master.quit()


def label_all_images(unlabeled_dir, image_dir, label_dir):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files = [f for f in os.listdir(unlabeled_dir) if f.lower().endswith(IMAGE_EXTS)]

    for file_name in files:
        img_path = os.path.join(unlabeled_dir, file_name)
        label_path = os.path.join(label_dir, file_name)

        root = tk.Tk()
        app = CrackLabeler(root, img_path, label_path)
        root.mainloop()
        root.destroy()

        if app.finished:
            shutil.move(img_path, os.path.join(image_dir, file_name))
            print(f"Labeled {file_name}")
