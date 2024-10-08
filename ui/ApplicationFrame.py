import tkinter as tk

class ApplicationFrame(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Application Frame")

        # Создаем канвас
        self.canvas = tk.Canvas(self, width=512, height=512)
        self.canvas.pack()

        # Рисуем прямоугольники
        self.canvas.create_rectangle(100, 100, 300, 300, fill="#ff0000")  # Красный прямоугольник
        self.canvas.create_rectangle(200, 200, 400, 400, outline="#0000ff")  # Синий контур

if __name__ == "__main__":
    app = ApplicationFrame()
    app.mainloop()
