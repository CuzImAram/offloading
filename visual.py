import tkinter as tk
from tkinter import filedialog, messagebox
import random
from PIL import Image, ImageTk 
import copy

class SeamCarvingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Seam Carving High-Res Visualizer (No Numbers)")
        self.root.geometry("1400x900")
        
        # Standard-Setup für ein detaillierteres Gitter
        self.width = 80
        self.height = 60
        self.cell_size = 8
        
        # Initialdaten (Rauschen als Startbild)
        self.pixels = [[random.randint(50, 200) for _ in range(self.width)] for _ in range(self.height)]
        self.energy = []
        self.cum_energy = []
        self.seam = []
        self.seam_start_x = -1
        self.current_pass = 1
        self.state = "START"
        self.step_num = 1
        self.history = []

        # --- UI SETUP ---
        self.main_container = tk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.sim_frame = tk.Frame(self.main_container)
        self.sim_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Rechte Seite: Legende
        self.legend_frame = tk.Frame(self.main_container, bg="#fdfdfd", width=250, highlightbackground="#ccc", highlightthickness=1)
        self.legend_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.setup_legend()

        # Header
        self.header = tk.Label(self.sim_frame, text=f"PASS {self.current_pass}", font=("Arial", 16, "bold"))
        self.header.pack(pady=5)

        # Hardware & Step Log
        self.log_frame = tk.Frame(self.sim_frame, bg="#2c3e50", padx=10, pady=8)
        self.log_frame.pack(fill=tk.X, padx=10)
        
        self.step_label = tk.Label(self.log_frame, text="Step 1", fg="#f1c40f", bg="#2c3e50", font=("Consolas", 14, "bold"))
        self.step_label.pack(side=tk.LEFT)
        
        self.hw_label = tk.Label(self.log_frame, text="HW: GPU", fg="#e74c3c", bg="#2c3e50", font=("Consolas", 12, "bold"))
        self.hw_label.pack(side=tk.LEFT, padx=20)
        
        self.canvas = tk.Canvas(self.sim_frame, bg="#222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.info = tk.Label(self.sim_frame, text="Bereit für Energieberechnung (Sobel)", font=("Arial", 11))
        self.info.pack()

        # Button Frame
        self.btn_frame = tk.Frame(self.sim_frame)
        self.btn_frame.pack(fill=tk.X, padx=20, pady=15)

        self.load_btn = tk.Button(self.btn_frame, text="Bild laden (.jpg)", command=self.load_image, bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=15)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.back_btn = tk.Button(self.btn_frame, text="<- Back", command=self.back_step, bg="#95a5a6", fg="white", font=("Arial", 10, "bold"), width=15)
        self.back_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(self.btn_frame, text="Next Step ->", command=self.next_step, bg="#34495e", fg="white", font=("Arial", 12, "bold"))
        self.next_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.canvas.bind("<Configure>", lambda e: self.draw_grid())
        self.draw_grid()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path: return

        try:
            img = Image.open(file_path).convert('L') 
            # Höhere Auflösung (z.B. max 100px Breite für gute Sichtbarkeit)
            img.thumbnail((100, 80)) 
            self.width, self.height = img.size
            self.pixels = [[img.getpixel((x, y)) for x in range(self.width)] for y in range(self.height)]
            
            self.current_pass = 1
            self.history = []
            self.state = "START"
            self.energy = []
            self.cum_energy = []
            self.seam = []
            self.update_ui_text(1, "GPU", "Neues Bild geladen. Bereit für Energieberechnung.")
            self.draw_grid()
        except Exception as e:
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden: {e}")

    def draw_grid(self, seam_line=None):
        self.canvas.delete("all")
        w_can, h_can = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w_can < 100: return
        
        current_w = len(self.pixels[0])
        # Zellengröße dynamisch berechnen, aber ohne Platz für Text zu lassen
        self.cell_size = min((w_can - 60) / current_w, (h_can - 60) / self.height)

        for y in range(self.height):
            for x in range(current_w):
                val = self.pixels[y][x]
                x0, y0 = x * self.cell_size + 30, y * self.cell_size + 30
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                
                # Farblogik (Steps aus enlarge.c)
                if self.state == "ENERGY" and self.energy:
                    e_val = self.energy[y][x]
                    color = f"#{min(e_val*4, 255):02x}0000" # Rot für Energie
                elif self.state == "CUMULATIVE" and self.cum_energy:
                    ce_val = self.cum_energy[y][x]
                    # Blau für kumulative Summen
                    color = f"#{min(int(ce_val/10), 255):02x}{min(int(ce_val/10), 255):02x}ff"
                else:
                    color = f"#{val:02x}{val:02x}{val:02x}" # Normales Graustufenbild
                
                # Gitterlinien bei hoher Auflösung fast unsichtbar (gray20)
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#333", width=0 if current_w > 60 else 1)

        if seam_line:
            mid = self.cell_size / 2
            for i in range(len(seam_line)-1):
                p1, p2 = seam_line[i], seam_line[i+1]
                # Seam-Linie kräftig Rot
                self.canvas.create_line(p1[0]*self.cell_size+30+mid, p1[1]*self.cell_size+30+mid, 
                                         p2[0]*self.cell_size+30+mid, p2[1]*self.cell_size+30+mid, 
                                         fill="#ff0000", width=2)

    def next_step(self):
        self.save_state()
        if self.state == "START":
            self.calc_energy()
            self.state = "ENERGY"
            self.update_ui_text(1, "GPU", "calculateEnergySobel: Wichtige Kanten erkannt.")
        elif self.state == "ENERGY":
            self.calc_cumulative()
            self.state = "CUMULATIVE"
            self.update_ui_text(2, "GPU", "calculateMinEnergySums: Kosten-Pfade berechnet.")
        elif self.state == "CUMULATIVE":
            self.find_seam_start()
            self.state = "FIND_START"
            self.update_ui_text(3, "CPU", "findAllSeams: Startpunkt-Suche (Sequenziell).")
        elif self.state == "FIND_START":
            self.trace_seam_path()
            self.state = "TRACING"
            self.update_ui_text(3, "GPU", "traceAllSeams: Pfad-Rückverfolgung (Parallel).")
        elif self.state == "TRACING":
            self.enlarge_image()
            self.state = "START"
            self.update_ui_text(4, "GPU", "insertAllSeams: Pixel dupliziert (Bild verbreitert).")
            self.current_pass += 1
        self.draw_grid(seam_line=self.seam if self.state in ["TRACING", "START"] else None)

    # --- Die Logik-Methoden bleiben identisch zu enlarge.c ---
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

    def back_step(self):
        if not self.history: return
        prev = self.history.pop()
        self.pixels, self.energy, self.cum_energy = prev['pixels'], prev['energy'], prev['cum_energy']
        self.seam, self.seam_start_x, self.current_pass = prev['seam'], prev['seam_start_x'], prev['current_pass']
        self.state, self.step_num = prev['state'], prev['step_num']
        self.update_ui_text(self.step_num, "CPU" if self.state == "FIND_START" else "GPU", prev['info_text'])
        self.draw_grid(seam_line=self.seam if self.state in ["TRACING", "START"] else None)

    def save_state(self):
        self.history.append({'pixels': copy.deepcopy(self.pixels), 'energy': copy.deepcopy(self.energy), 'cum_energy': copy.deepcopy(self.cum_energy), 'seam': copy.deepcopy(self.seam), 'seam_start_x': self.seam_start_x, 'current_pass': self.current_pass, 'state': self.state, 'step_num': self.step_num, 'info_text': self.info.cget("text")})

    def update_ui_text(self, step, hw, task):
        self.step_label.config(text=f"Step {step}"); self.hw_label.config(text=f"HW: {hw}"); self.info.config(text=task); self.header.config(text=f"PASS {self.current_pass}")

    def setup_legend(self):
        tk.Label(self.legend_frame, text="LEGENDE", font=("Arial", 12, "bold"), bg="#fdfdfd").pack(pady=10)
        for col, txt in [("#888", "Originalbild"), ("#a00", "Energie Map (Sobel)"), ("#00a", "Kumulative Kosten"), ("#f00", "Bester Seam (Pfad)")]:
            f = tk.Frame(self.legend_frame, bg="#fdfdfd"); f.pack(anchor="w", padx=15, pady=2)
            tk.Label(f, bg=col, width=2, relief="sunken").pack(side=tk.LEFT); tk.Label(f, text=txt, font=("Arial", 9), bg="#fdfdfd").pack(side=tk.LEFT, padx=5)

if __name__ == "__main__":
    root = tk.Tk(); app = SeamCarvingVisualizer(root); root.mainloop()