from tkinter import *
from tkinter import filedialog

def import_file():
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if file_path:
        print("Selected file:", file_path)

window = Tk()

# Window name
window.title("Welcome to the Walking or Jumping Predictor")

# Window size
window.geometry('1000x1000')

# Main label when app opens
mainLabel = Label(window, text="Ready to predict whether you are Walking or Jumping?", font=("Times New Roman", 15, "bold"))
mainLabel.pack(pady=20)

# Button to select file
selectFile = Button(window, text="Select Input File", command=import_file)
selectFile.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()
