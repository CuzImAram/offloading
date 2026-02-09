import tkinter as tk
import random

class SeamCarvingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Seam Carving Offloading Visualization")
        
        # Grid Setup
        self.width = 10
        self.height = 8
        self.cell_size = 50
        
        # Initialisiere Pixel
        self.pixels = [[random.randint(50, 200) for _ in range(self.width)] for _ in range(self.height)]
        
        # Diese Matrizen werden jetzt dynamisch in den Funktionen erstellt
        self.energy = []
        self.cum_energy = []
        
        self.state = "START" 
        self.seam = []

        # UI Setup
        self.header = tk.Label(root, text="Seam Carving: Initial Image", font=("Arial", 14, "bold"))
        self.header.pack(pady=10)
        
        self.canvas = tk.Canvas(root, width=800, height=500, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.info = tk.Label(root, text="Klicke 'Next Step' um die Energy Map zu berechnen", font=("Arial", 10))
        self.info.pack(pady=5)
        
        self.btn = tk.Button(root, text="Next Step", command=self.next_step, bg="#2c3e50", fg="white", font=("Arial", 10, "bold"), width=20)
        self.btn.pack(pady=10)
        
        self.draw_grid()

    def draw_grid(self, seam_line=None):
        self.canvas.delete("all")
        current_w = len(self.pixels[0])
        
        for y in range(self.height):
            for x in range(current_w):
                val = self.pixels[y][x]
                x0, y0 = x * self.cell_size + 20, y * self.cell_size + 20
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                
                # Farblogik basierend auf Phase
                if self.state == "ENERGY" and self.energy:
                    e_val = self.energy[y][x]
                    e_norm = min(e_val * 2, 255)
                    color = f"#{e_norm:02x}0000"
                    disp_val = e_val
                elif self.state == "CUMULATIVE" and self.cum_energy:
                    ce_val = self.cum_energy[y][x]
                    ce_norm = min(int(ce_val / 5), 255)
                    color = f"#{ce_norm:02x}{ce_norm:02x}ff"
                    disp_val = ce_val
                else:
                    color = f"#{val:02x}{val:02x}{val:02x}"
                    disp_val = val
                
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray")
                # Wert nur anzeigen wenn Zellen groß genug sind
                if current_w < 15:
                    text_col = "green" if (self.state == "START" and val > 150) else "white"
                    self.canvas.create_text(x0+25, y0+25, text=str(disp_val), fill=text_col, font=("Arial", 8))

        if seam_line:
            for i in range(len(seam_line)-1):
                p1, p2 = seam_line[i], seam_line[i+1]
                self.canvas.create_line(p1[0]*self.cell_size+45, p1[1]*self.cell_size+45, 
                                         p2[0]*self.cell_size+45, p2[1]*self.cell_size+45, 
                                         fill="red", width=4)

    def next_step(self):
        if self.state == "START":
            self.calc_energy()
            self.state = "ENERGY"
            self.header.config(text="Phase 1: Energy Map (Sobel)")
            self.info.config(text="Berechnung der Kantenstärke (Gradienten).")
        
        elif self.state == "ENERGY":
            self.calc_cumulative()
            self.state = "CUMULATIVE"
            self.header.config(text="Phase 2: Accumulated Energy")
            self.info.config(text="Dynamic Programming: Akkumulierte Kosten von oben nach unten.")
            
        elif self.state == "CUMULATIVE":
            self.trace_seam()
            self.state = "TRACING"
            self.header.config(text="Phase 3: Seam Tracing")
            self.info.config(text="Backtracking: Der Pfad mit der geringsten Energie wurde gefunden.")
            
        elif self.state == "TRACING":
            self.enlarge_image()
            self.state = "START"
            self.header.config(text="Phase 4: Image Enlarged")
            self.info.config(text="Ein Seam wurde eingefügt. Bildbreite erhöht.")
            
        self.draw_grid(seam_line=self.seam if self.state in ["TRACING", "START"] else None)

    def calc_energy(self):
        """ Entspricht calculateEnergySobel in enlarge.c """
        current_w = len(self.pixels[0])
        # WICHTIG: Matrix auf aktuelle Breite initialisieren
        self.energy = [[0 for _ in range(current_w)] for _ in range(self.height)]
        
        for y in range(self.height):
            for x in range(current_w):
                # Gradientenberechnung (vereinfacht)
                dx = abs(self.pixels[y][(x+1)%current_w] - self.pixels[y][x-1])
                dy = abs(self.pixels[(y+1)%self.height][x] - self.pixels[y-1][x])
                self.energy[y][x] = dx + dy

    def calc_cumulative(self):
        """ Entspricht calculateMinEnergySums in enlarge.c """
        current_w = len(self.pixels[0])
        self.cum_energy = [[0 for _ in range(current_w)] for _ in range(self.height)]
        
        for x in range(current_w):
            self.cum_energy[0][x] = self.energy[0][x]
        
        for y in range(1, self.height):
            for x in range(current_w):
                # Suche Minimum der 3 oberen Nachbarn
                prev_options = []
                for dx in [-1, 0, 1]:
                    if 0 <= x+dx < current_w:
                        prev_options.append(self.cum_energy[y-1][x+dx])
                self.cum_energy[y][x] = self.energy[y][x] + min(prev_options)

    def trace_seam(self):
        """ Entspricht traceAllSeams in enlarge.c """
        current_w = len(self.pixels[0])
        self.seam = []
        # Start am Minimum der letzten Zeile
        last_row = self.cum_energy[self.height-1]
        curr_x = last_row.index(min(last_row))
        self.seam.append((curr_x, self.height-1))
        
        for y in range(self.height-2, -1, -1):
            possible_x = [curr_x-1, curr_x, curr_x+1]
            best_x = curr_x
            min_val = float('inf')
            for px in possible_x:
                if 0 <= px < current_w and self.cum_energy[y][px] < min_val:
                    min_val = self.cum_energy[y][px]
                    best_x = px
            curr_x = best_x
            self.seam.append((curr_x, y))

    def enlarge_image(self):
        """ Entspricht insertAllSeams in enlarge.c """
        new_pixels = []
        for y in range(self.height):
            row = list(self.pixels[y])
            # Finde x-Koordinate des Seams in dieser Zeile
            seam_x = [p[0] for p in self.seam if p[1] == y][0]
            # Pixel duplizieren (Offloading Logik: Bild vergrößern)
            row.insert(seam_x, row[seam_x])
            new_pixels.append(row)
        
        self.pixels = new_pixels
        self.seam = [] # Reset Seam für nächsten Durchlauf

if __name__ == "__main__":
    root = tk.Tk()
    app = SeamCarvingVisualizer(root)
    root.mainloop()