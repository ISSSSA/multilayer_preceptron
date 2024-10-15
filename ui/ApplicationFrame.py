import tkinter as tk
from tkinter import Button, Canvas, filedialog, Label, Entry
from PIL import Image, ImageDraw


GRID_SIZE = 32
PIXEL_SIZE = 15


class PixelArtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art 32x32")
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        self.canvas = Canvas(main_frame, width=GRID_SIZE * PIXEL_SIZE, height=GRID_SIZE * PIXEL_SIZE)
        self.canvas.grid(row=0, column=0, columnspan=4, padx=5, pady=5)

        self.clear_button = Button(main_frame, text="Очистить", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, padx=5, pady=5)

        self.save_button = Button(main_frame, text="Сохранить", command=self.save_image)
        self.save_button.grid(row=1, column=1, padx=5, pady=5)

        self.load_button = Button(main_frame, text="Загрузить изображение", command=self.load_image)
        self.load_button.grid(row=1, column=2, padx=5, pady=5)

        self.epochs_label = Label(main_frame, text="Количество эпох:")
        self.epochs_label.grid(row=2, column=0, sticky="e", padx=5, pady=5)

        self.epochs_entry = Entry(main_frame)
        self.epochs_entry.grid(row=2, column=1, padx=5, pady=5)

        self.lr_label = Label(main_frame, text="Скорость обучения:")
        self.lr_label.grid(row=3, column=0, sticky="e", padx=5, pady=5)

        self.lr_entry = Entry(main_frame)
        self.lr_entry.grid(row=3, column=1, padx=5, pady=5)

        self.train_button = Button(main_frame, text="Начать обучение", command=self.start_training)
        self.train_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        self.pixels = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.drawing = False
        self.image = Image.new("1", (GRID_SIZE, GRID_SIZE), color=1)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.update_canvas()

    def start_draw(self, event):
        self.drawing = True
        self.toggle_pixel(event)

    def stop_draw(self, event):
        self.drawing = False

    def draw_pixel(self, event):
        if self.drawing:
            self.toggle_pixel(event)

    def toggle_pixel(self, event):
        x = event.x // PIXEL_SIZE
        y = event.y // PIXEL_SIZE

        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.pixels[y][x] = 1 - self.pixels[y][x]
            color = "white" if self.pixels[y][x] == 0 else "black"
            self.canvas.create_rectangle(x * PIXEL_SIZE, y * PIXEL_SIZE,
                                         (x + 1) * PIXEL_SIZE, (y + 1) * PIXEL_SIZE,
                                         fill=color, outline="gray")

            self.draw.point((x, y), fill=self.pixels[y][x])

    def clear_canvas(self):
        self.pixels = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.image = Image.new("1", (GRID_SIZE, GRID_SIZE), color=1)
        self.draw = ImageDraw.Draw(self.image)
        self.update_canvas()

    def update_canvas(self):
        self.canvas.delete("all")
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = "white" if self.pixels[y][x] == 0 else "black"
                self.canvas.create_rectangle(x * PIXEL_SIZE, y * PIXEL_SIZE,
                                             (x + 1) * PIXEL_SIZE, (y + 1) * PIXEL_SIZE,
                                             fill=color, outline="gray")

    def save_image(self):
        self.image.save("pixel_art.bmp")
        print("Изображение сохранено как pixel_art.bmp")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.bmp")])
        if file_path:
            image = Image.open(file_path).convert("1")
            image = image.resize((GRID_SIZE, GRID_SIZE))
            self.pixels = self.image_to_pixel_array(image)
            self.update_canvas()
            print("Изображение загружено и преобразовано в массив пикселей.")

    def image_to_pixel_array(self, image):
        pixel_array = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                pixel_value = image.getpixel((x, y))
                pixel_array[y][x] = 1 if pixel_value == 0 else 0
        print(pixel_array)
        return pixel_array

    def start_training(self):
        epochs = self.epochs_entry.get()
        learning_rate = self.lr_entry.get()

        print(f"Начало обучения с {epochs} эпохами и скоростью обучения {learning_rate}.")


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelArtApp(root)
    root.mainloop()
