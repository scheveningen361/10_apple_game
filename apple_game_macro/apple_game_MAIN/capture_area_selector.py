
import tkinter as tk

class AreaSelector:
    def __init__(self, master):
        self.master = master
        # Tkinter window settings: fullscreen, semi-transparent
        self.master.attributes("-fullscreen", True)
        self.master.attributes("-alpha", 0.3)
        self.master.wait_visibility(self.master)
        self.master.configure(bg='grey')

        self.start_x = None
        self.start_y = None
        self.rect = None

        # Create canvas to draw drag area
        self.canvas = tk.Canvas(self.master, cursor="cross", bg="grey", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        """Records starting coordinates on mouse click and begins drawing rectangle."""
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        """Updates rectangle size in real-time during mouse drag."""
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        """Calculates final coordinates and outputs them, then exits."""
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Calculate top-left and bottom-right coordinates regardless of start/end points
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        print(f"Capture Area Coordinates: x={int(x1)}, y={int(y1)}, width={int(x2-x1)}, height={int(y2-y1)}")
        self.master.quit()

if __name__ == '__main__':
    root = tk.Tk()
    app = AreaSelector(root)
    root.mainloop()
    # Destroy window completely after mainloop ends
    root.destroy()
