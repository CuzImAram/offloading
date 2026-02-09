import tkinter as tk
import random

class SeamCarvingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Seam Carving Offloading (Fixed Steps)")
        self.root.geometry("1100x750")
        
        # Grid Setup
        self.width = 10
        self.height = 8
        self.cell_size = 50
        
        # Initialdaten
        self.pixels = [[random.randint(50, 200) for _ in range(self.width)] for _ in range(self.height)]
        self.energy = []
        self.cum_energy = []
        self.seam = []
        self.seam_start_x = -1
        self.current_pass = 1

        # State Mapping basierend auf enlarge.c
        # 1: calculateEnergySobel, 2: calculateMinEnergySums, 3: traceAllSeams, 4: insertAllSeams
        self.step_num = 1 
        self.state = "START" 

        # --- UI SETUP ---
        self.header = tk.Label(root, text=f"PASS {self.current_pass}", font=("Arial", 16, "bold"))
        self.header.pack(pady=5)

        # Hardware & Step Log
        self.log_frame = tk.Frame(root, bg="#2c3e50", padx=10, pady=8)
        self.log_frame.pack(fill=tk.X, padx=10)
        
        self.step_label = tk.Label(self.log_frame, text="Step 1", fg="#f1c40f", bg="#2c3e50", font=("Consolas", 14, "bold"))
        self.step_label.pack(side=tk.LEFT)
        
        self.hw_label = tk.Label(self.log_frame, text="HW: GPU", fg="#e74c3c", bg="#2c3e50", font=("Consolas", 12, "bold"))
        self.hw_label.pack(side=tk.LEFT, padx=20)
        
        self.meaning_label = tk.Label(self.log_frame, text="Zahlen = Graustufen (0-255)", fg="#bdc3c7", bg="#2c3e50", font=("Arial", 10, "italic"))
        self.meaning_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(root, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.info = tk.Label(root, text="Bereit für Energieberechnung (Sobel)", font=("Arial", 11))
        self.info.pack()

        self.btn = tk.Button(root, text="Nächster Schritt", command=self.next_step, bg="#34495e", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn.pack(fill=tk.X, padx=20, pady=15)
        
        self.draw_grid()

    def update_ui_text(self, step, hw, task, meaning, color="#3498db"):
        self.step_label.config(text=f"Step {step}")
        self.hw_label.config(text=f"HW: {hw}", fg="#e74c3c" if hw=="GPU" else "#f1c40f")
        self.meaning_label.config(text=f"Bedeutung: {meaning}")
        self.info.config(text=task, fg=color)
        self.header.config(text=f"PASS {self.current_pass}")

    def draw_grid(self, seam_line=None):
        self.canvas.delete("all")
        current_w = len(self.pixels[0])
        
        # Dynamische Zellengröße bei Fullscreen/vielen Klicks
        display_w = self.canvas.winfo_width()
        if display_w > 100:
            self.cell_size = min(60, (display_w - 100) / current_w)

        for y in range(self.height):
            for x in range(current_w):
                val = self.pixels[y][x]
                x0, y0 = x * self.cell_size + 40, y * self.cell_size + 40
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                
                # Farblogik
                if self.state == "ENERGY" and self.energy:
                    e_val = self.energy[y][x]
                    color = f"#{min(e_val*4, 255):02x}0000"
                    disp = e_val
                elif self.state == "CUMULATIVE" and self.cum_energy:
                    ce_val = self.cum_energy[y][x]
                    color = f"#{min(int(ce_val/4), 255):02x}{min(int(ce_val/4), 255):02x}ff"
                    disp = ce_val
                else:
                    color = f"#{val:02x}{val:02x}{val:02x}"
                    disp = val
                
                outline = "cyan" if (self.state == "FIND_START" and y == self.height-1 and x == self.seam_start_x) else "#444"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=outline, width=1)
                
                # Zahlen fix: Jetzt bis zu einer Breite von 40 Spalten sichtbar
                if current_w < 40:
                    font_size = int(self.cell_size / 4)
                    self.canvas.create_text(x0+self.cell_size/2, y0+self.cell_size/2, 
                                            text=str(disp), fill="white", font=("Arial", font_size))

        if seam_line:
            for i in range(len(seam_line)-1):
                p1, p2 = seam_line[i], seam_line[i+1]
                self.canvas.create_line(p1[0]*self.cell_size+40+self.cell_size/2, p1[1]*self.cell_size+40+self.cell_size/2, 
                                         p2[0]*self.cell_size+40+self.cell_size/2, p2[1]*self.cell_size+40+self.cell_size/2, 
                                         fill="#e74c3c", width=3)

    def next_step(self):
        if self.state == "START":
            self.calc_energy()
            self.state = "ENERGY"
            self.step_num = 1
            self.update_ui_text(1, "GPU", "calculateEnergySobel (Offloading)", "Kantenstärke/Gradienten")
        
        elif self.state == "ENERGY":
            self.calc_cumulative()
            self.state = "CUMULATIVE"
            self.step_num = 2
            self.update_ui_text(2, "GPU", "calculateMinEnergySums (DP)", "Kumulative Kosten")
            
        elif self.state == "CUMULATIVE":
            self.find_seam_start()
            self.state = "FIND_START"
            self.step_num = 3
            self.update_ui_text(3, "CPU", "findAllSeams & qsort", "Kumulative Kosten (Startpunkt-Suche)")
            
        elif self.state == "FIND_START":
            self.trace_seam_path()
            self.state = "TRACING"
            self.step_num = 3 # Gehört im C-Code zum selben Block (Finding/Tracing)
            self.update_ui_text(3, "GPU", "traceAllSeams (Backtracking)", "Kumulative Kosten (Pfad)")
            
        elif self.state == "TRACING":
            self.enlarge_image()
            self.state = "START"
            self.step_num = 4
            self.update_ui_text(4, "GPU", "insertAllSeams (Resizing)", "Graustufen (Neuer Pixel eingefügt)")
            self.current_pass += 1
            
        self.draw_grid(seam_line=self.seam if self.state in ["TRACING", "START"] else None)

    # ... (Methoden calc_energy, calc_cumulative, find_seam_start, trace_seam_path, enlarge_image identisch zum Vorherigen) ...
    def calc_energy(self):
        w = len(self.pixels[0])
        self.energy = [[0 for _ in range(w)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(w):
                dx = abs(self.pixels[y][(x+1)%w] - self.pixels[y][x-1])
                dy = abs(self.pixels[(y+1)%self.height][x] - self.pixels[y-1][x])
                self.energy[y][x] = dx + dy

    def calc_cumulative(self):
        w = len(self.pixels[0])
        self.cum_energy = [[0 for _ in range(w)] for _ in range(self.height)]
        for x in range(w): self.cum_energy[0][x] = self.energy[0][x]
        for y in range(1, self.height):
            for x in range(w):
                prev = [self.cum_energy[y-1][px] for px in [x-1, x, x+1] if 0 <= px < w]
                self.cum_energy[y][x] = self.energy[y][x] + min(prev)

    def find_seam_start(self):
        last_row = self.cum_energy[self.height-1]
        self.seam_start_x = last_row.index(min(last_row))

    def trace_seam_path(self):
        w = len(self.pixels[0])
        self.seam = [(self.seam_start_x, self.height-1)]
        curr_x = self.seam_start_x
        for y in range(self.height-2, -1, -1):
            opts = [px for px in [curr_x-1, curr_x, curr_x+1] if 0 <= px < w]
            curr_x = min(opts, key=lambda x: self.cum_energy[y][x])
            self.seam.append((curr_x, y))

    def enlarge_image(self):
        new_pixels = []
        for y in range(self.height):
            row = list(self.pixels[y])
            sx = [p[0] for p in self.seam if p[1] == y][0]
            row.insert(sx, row[sx])
            new_pixels.append(row)
        self.pixels = new_pixels

if __name__ == "__main__":
    root = tk.Tk()
    app = SeamCarvingVisualizer(root)
    root.mainloop()