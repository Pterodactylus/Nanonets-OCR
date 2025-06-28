import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import subprocess
import threading
import os
from pathlib import Path
import sys # Import sys

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nanonets OCR GUI")
        self.root.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        # Input/Output Frame
        io_frame = ttk.LabelFrame(self.root, text="Input/Output Settings")
        io_frame.pack(padx=10, pady=10, fill="x")

        # Input Directory
        ttk.Label(io_frame, text="Input Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.input_dir_entry = ttk.Entry(io_frame, width=50)
        self.input_dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(io_frame, text="Browse", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=5)

        # Output Directory
        ttk.Label(io_frame, text="Output Directory:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.output_dir_entry = ttk.Entry(io_frame, width=50)
        self.output_dir_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(io_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)

        # OCR Parameters Frame
        params_frame = ttk.LabelFrame(self.root, text="OCR Parameters")
        params_frame.pack(padx=10, pady=10, fill="x")

        # Device Selection (CPU/GPU)
        ttk.Label(params_frame, text="Processing Device:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.device_var = tk.StringVar(value="cpu")
        ttk.Radiobutton(params_frame, text="CPU", variable=self.device_var, value="cpu").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(params_frame, text="GPU", variable=self.device_var, value="gpu").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Batch Size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.batch_size_spinbox = ttk.Spinbox(params_frame, from_=1, to=10, width=5, format="%d")
        self.batch_size_spinbox.set(2)
        self.batch_size_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # PDF Chunk Size
        ttk.Label(params_frame, text="PDF Chunk Size:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.pdf_chunk_size_spinbox = ttk.Spinbox(params_frame, from_=1, to=10, width=5, format="%d")
        self.pdf_chunk_size_spinbox.set(1)
        self.pdf_chunk_size_spinbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Max Image Size
        ttk.Label(params_frame, text="Max Image Width:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.max_width_spinbox = ttk.Spinbox(params_frame, from_=100, to=4000, width=8, format="%d")
        self.max_width_spinbox.set(1080)
        self.max_width_spinbox.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(params_frame, text="Max Image Height:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.max_height_spinbox = ttk.Spinbox(params_frame, from_=100, to=4000, width=8, format="%d")
        self.max_height_spinbox.set(1080)
        self.max_height_spinbox.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Start Button
        self.start_button = ttk.Button(self.root, text="Start OCR", command=self.start_ocr_process)
        self.start_button.pack(pady=10)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5)

        # Log Display
        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=90, height=15)
        self.log_text.pack(padx=10, pady=10, fill="both", expand=True)
        self.log_text.config(state=tk.DISABLED) # Make it read-only

    def browse_input_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir_entry.delete(0, tk.END)
            self.input_dir_entry.insert(0, directory)

    def browse_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, directory)

    def log_message(self, message):
        # Log to GUI
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END) # Auto-scroll to the end
        self.log_text.config(state=tk.DISABLED)
        
        # Also log to console for debugging
        print(f"[GUI] {message}")

    def start_ocr_process(self):
        input_dir = self.input_dir_entry.get()
        output_dir = self.output_dir_entry.get()
        device = self.device_var.get()
        batch_size = self.batch_size_spinbox.get()
        pdf_chunk_size = self.pdf_chunk_size_spinbox.get()
        max_width = self.max_width_spinbox.get()
        max_height = self.max_height_spinbox.get()

        if not input_dir or not output_dir:
            self.log_message("Error: Please select both input and output directories.")
            return

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END) # Clear previous logs
        self.log_text.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        self.log_message("Starting OCR process...")
        self.start_button.config(state=tk.DISABLED) # Disable button during processing

        # Run OCR in a separate thread to keep GUI responsive
        ocr_thread = threading.Thread(target=self._run_ocr_in_thread, args=(input_dir, output_dir, device, batch_size, pdf_chunk_size, max_width, max_height))
        ocr_thread.start()

    def _run_ocr_in_thread(self, input_dir, output_dir, device, batch_size, pdf_chunk_size, max_width, max_height):
        try:
            # Determine the path to batch_ocr.py within the PyInstaller bundle
            if getattr(sys, 'frozen', False): # Check if running in a PyInstaller bundle
                # If frozen, we need to use the bundled Python interpreter
                # In PyInstaller, sys.executable points to the main executable
                # and batch_ocr.py is in the _internal directory
                bundle_dir = os.path.dirname(sys.executable)
                script_path = os.path.join(bundle_dir, '_internal', 'batch_ocr.py')
                
                # Use the bundled Python executable
                python_executable = sys.executable
                
                # Set up environment to use bundled libraries
                env = os.environ.copy()
                env['PYTHONPATH'] = os.path.join(bundle_dir, '_internal')
                
                # For PyInstaller bundles, we need to run the script directly with the bundled interpreter
                # but we'll import and run it as a module instead of subprocess to ensure all dependencies work
                self._run_ocr_directly(input_dir, output_dir, device, batch_size, pdf_chunk_size, max_width, max_height)
                return
            else:
                # Not frozen, running as a regular Python script
                script_path = "batch_ocr.py"
                python_executable = sys.executable
                env = None

            command = [
                python_executable, script_path,
                "--input-dir", input_dir,
                "--output-dir", output_dir,
                "--batch-size", str(batch_size),
                "--pdf-chunk-size", str(pdf_chunk_size),
                "--max-image-width", str(max_width),
                "--max-image-height", str(max_height)
            ]
            if device == "gpu":
                command.append("--gpu")
            else:
                command.append("--cpu")

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, env=env)

            total_files = 0
            processed_files = 0

            for line in process.stdout:
                self.root.after(0, self.log_message, line.strip()) # Update GUI from main thread

                # Attempt to parse total files and progress
                if "Found" in line and "files to process" in line:
                    try:
                        total_files = int(line.split("Found")[1].split("files")[0].strip())
                        self.root.after(0, lambda: self.progress_bar.config(maximum=total_files))
                    except ValueError:
                        pass
                elif "Processing:" in line and "[" in line and "]" in line:
                    try:
                        parts = line.split("[")[1].split("]")[0].split("/")
                        processed_files = int(parts[0])
                        total_files_from_line = int(parts[1])
                        if total_files_from_line > total_files: # Update total files if a more accurate count appears
                            total_files = total_files_from_line
                            self.root.after(0, lambda: self.progress_bar.config(maximum=total_files))
                        self.root.after(0, lambda: self.progress_bar.config(value=processed_files))
                    except (ValueError, IndexError):
                        pass

            process.wait() # Wait for the process to finish

            if process.returncode == 0:
                self.root.after(0, self.log_message, "OCR process completed successfully!")
            else:
                self.root.after(0, self.log_message, f"OCR process failed with error code {process.returncode}")

        except FileNotFoundError:
            self.root.after(0, self.log_message, "Error: 'python' or 'batch_ocr.py' not found. Ensure Python is in your PATH and batch_ocr.py is in the same directory.")
        except Exception as e:
            self.root.after(0, self.log_message, f"An unexpected error occurred: {e}")
        finally:
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL)) # Re-enable button

    def _run_ocr_directly(self, input_dir, output_dir, device, batch_size, pdf_chunk_size, max_width, max_height):
        """Run OCR directly by importing the batch_ocr module when in PyInstaller bundle"""
        try:
            # Import the batch_ocr module directly
            import batch_ocr
            
            # Create the processor with the specified parameters
            use_cpu = device == "cpu"
            processor = batch_ocr.MemoryOptimizedBatchOCRProcessor(
                use_cpu=use_cpu,
                batch_size=int(batch_size),
                max_pdf_pages_per_chunk=int(pdf_chunk_size),
                max_image_size=(int(max_width), int(max_height)),
                model_path="nanonets/Nanonets-OCR-s"
            )
            
            # Redirect stdout to capture the output
            import io
            import contextlib
            
            # Create a string buffer to capture output
            output_buffer = io.StringIO()
            
            # Capture both stdout and stderr
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                # Process documents
                processor.process_documents_folder(input_dir, output_dir, "ocr_results.csv")
            
            # Get the captured output
            captured_output = output_buffer.getvalue()
            
            # Send output to GUI line by line
            for line in captured_output.split('\n'):
                if line.strip():
                    self.root.after(0, self.log_message, line.strip())
            
            self.root.after(0, self.log_message, "OCR process completed successfully!")
            
        except Exception as e:
            self.root.after(0, self.log_message, f"Direct OCR execution failed: {e}")
            import traceback
            self.root.after(0, self.log_message, f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
