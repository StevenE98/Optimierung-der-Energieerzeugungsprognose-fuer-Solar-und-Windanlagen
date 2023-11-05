import tkinter as tk


root = tk.Tk()

root.title("Solarenergie Prognose")

root.geometry("800x400") #Standard-Größe
root.minsize(width=400, height=400) #kleinste Fenster
root.maxsize(width=1024, height=768) #größte Fenster
root.resizable(width= True, height= True)

label1 = tk.Label(root, text="Solarenergie Prognose", bg="green" )
label1.pack(side="top", expand= True  ,fill="x")

label2 = tk.Label(root, text="Freiburg", bg="red")
label2.pack(side="bottom"  ,fill="x")

bild1 = tk.PhotoImage(file="sonne.png") #bild
label3 = tk.Label(root, image=bild1)
label3.pack(side="right")

root.mainloop()