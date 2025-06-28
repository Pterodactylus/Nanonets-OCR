import os
import time
import gc
from datetime import datetime
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import fitz  # PyMuPDF for PDF handling
from contextlib import contextmanager

class TimingProfiler:
    """Simple timing profiler for real-time step timing"""
    
    @contextmanager
    def time_operation(self, operation_name, show_timing=True):
        """Context manager for timing operations with real-time output"""
        start_time = time.time()
        try:
            yield
        finally:
            if show_timing:
                duration = time.time() - start_time
                print(f"   ⏱️  {operation_name}: {duration:.2f}s")

class MemoryOptimizedBatchOCRProcessor:
    def __init__(self, use_cpu=True, batch_size=2, max_pdf_pages_per_chunk=1, max_image_size=(1080, 1080)):
        self.model_path = "nanonets/Nanonets-OCR-s"
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.pdf'}
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = []
        self.use_cpu = use_cpu
        self.batch_size = batch_size
        self.max_pdf_pages_per_chunk = max_pdf_pages_per_chunk
        self.max_image_size = max_image_size
        
        # Model components - loaded on demand
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_loaded = False
        
        # Initialize timing profiler
        self.timing_profiler = TimingProfiler()
        
        print(f"🚀 Memory-Optimized OCR Processor initialized")
        print(f"   Batch size: {batch_size} files")
        print(f"   PDF chunk size: {max_pdf_pages_per_chunk} pages")
        print(f"   Max image size: {max_image_size}")
        print(f"   Device: {'CPU' if use_cpu else 'GPU'}")


    @contextmanager
    def model_context(self):
        """Context manager for loading/unloading model"""
        try:
            self._load_model()
            yield
        finally:
            self._unload_model()

    def _load_model(self):
        """Load model components into memory"""
        if self.model_loaded:
            return
            
        with self.timing_profiler.time_operation("model_loading"):
            print("🔄 Loading OCR model...")
            
            # Set device configuration
            if self.use_cpu:
                device_config = {"": "cpu"}
                torch_dtype = torch.float32
            else:
                device_config = "auto"
                torch_dtype = "auto"
            
            # Load model with standard attention only
            with self.timing_profiler.time_operation("standard_model_loading"):
                #snapshot_download(repo_id="nanonets/Nanonets-OCR-s", local_dir="./nanonets-Nanonets-OCR-s", repo_type="model")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch_dtype, 
                    device_map=device_config,
                    trust_remote_code=True
                )
                print("✅ Using standard attention")
            
            
            with self.timing_profiler.time_operation("tokenizer_processor_loading"):
                self.model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
                self.model_loaded = True
            
            print(f"✅ Model loaded on {'CPU' if self.use_cpu else 'GPU'}")
            self._print_memory_usage()

    def _unload_model(self):
        """Unload model components from memory"""
        if not self.model_loaded:
            return
            
        with self.timing_profiler.time_operation("model_unloading"):
            print("🗑️  Unloading model...")
            
            # Delete model components
            del self.model
            del self.tokenizer
            del self.processor
            
            self.model = None
            self.tokenizer = None
            self.processor = None
            self.model_loaded = False
            
            # Clear GPU cache if using GPU
            if not self.use_cpu and torch.cuda.is_available():
                with self.timing_profiler.time_operation("gpu_cache_clearing"):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # Force garbage collection
            with self.timing_profiler.time_operation("garbage_collection"):
                gc.collect()
            
            print("✅ Model unloaded, memory cleared")
            self._print_memory_usage()

    def _print_memory_usage(self):
        """Print current memory usage"""
        if not self.use_cpu and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"   GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

    def _clear_memory(self):
        """Clear memory between files"""
        if not self.use_cpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _resize_image_if_needed(self, image):
        """Resize image maintaining aspect ratio if it's too large to save memory"""
        if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
            # Calculate scaling factor to fit within bounds while preserving aspect ratio
            scale_w = self.max_image_size[0] / image.size[0]
            scale_h = self.max_image_size[1] / image.size[1]
            scale = min(scale_w, scale_h)
            
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            print(f"   Resizing large image from {image.size} to {new_size} (preserving aspect ratio)")
            image = image.resize(new_size, Image.Resampling.BILINEAR)
        return image

    def ocr_image(self, image_path, max_new_tokens=4096):
        """Perform OCR on a single image using the Nanonets model"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Use model_context() to load model.")
            
        with self.timing_profiler.time_operation("single_image_ocr"):
            prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
            
            # Load and resize image if needed
            with self.timing_profiler.time_operation("image_loading"):
                image = Image.open(image_path)
                image = self._resize_image_if_needed(image)
            
            with self.timing_profiler.time_operation("prompt_preparation"):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt},
                    ]},
                ]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
                inputs = inputs.to(self.model.device)
            
            output_ids = None
            generated_ids = None
            try:
                # Optimized generation parameters for speed
                with self.timing_profiler.time_operation("model_inference"):
                    output_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=max_new_tokens, 
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                with self.timing_profiler.time_operation("text_decoding"):
                    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                    output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    result = output_text[0]
            finally:
                # Clean up tensors
                with self.timing_profiler.time_operation("tensor_cleanup"):
                    if inputs is not None:
                        del inputs
                    if output_ids is not None:
                        del output_ids
                    if generated_ids is not None:
                        del generated_ids
                    self._clear_memory()
            
            return result

    def ocr_images_batch(self, image_paths, max_new_tokens=4096):
        """Perform OCR on multiple images in a single batch for maximum efficiency"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Use model_context() to load model.")
            
        with self.timing_profiler.time_operation("batch_image_ocr"):
            prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
            
            # Load and resize all images
            with self.timing_profiler.time_operation("batch_image_loading"):
                images = []
                for image_path in image_paths:
                    image = Image.open(image_path)
                    image = self._resize_image_if_needed(image)
                    images.append(image)
            
            # Create batch messages
            with self.timing_profiler.time_operation("batch_prompt_preparation"):
                all_messages = []
                all_texts = []
                
                for i, image_path in enumerate(image_paths):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": prompt},
                        ]},
                    ]
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    all_messages.append(messages)
                    all_texts.append(text)
                
                # Process batch
                inputs = self.processor(text=all_texts, images=images, padding=True, return_tensors="pt")
                inputs = inputs.to(self.model.device)
            
            output_ids = None
            generated_ids = None
            try:
                # Optimized generation parameters for speed
                with self.timing_profiler.time_operation("batch_model_inference"):
                    output_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=max_new_tokens, 
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                with self.timing_profiler.time_operation("batch_text_decoding"):
                    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(inputs.input_ids))]
                    output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                return output_texts
            finally:
                # Clean up tensors
                with self.timing_profiler.time_operation("tensor_cleanup"):
                    if inputs is not None:
                        del inputs
                    if output_ids is not None:
                        del output_ids
                    if generated_ids is not None:
                        del generated_ids
                    self._clear_memory()

    def pdf_to_images_chunked(self, pdf_path, temp_dir="temp_images"):
        """Convert PDF pages to images in chunks to save memory"""
        os.makedirs(temp_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"   PDF has {total_pages} pages, processing in chunks of {self.max_pdf_pages_per_chunk}")
        
        all_text = ""
        
        for chunk_start in range(0, total_pages, self.max_pdf_pages_per_chunk):
            chunk_end = min(chunk_start + self.max_pdf_pages_per_chunk, total_pages)
            chunk_image_paths = []
            
            # Convert chunk of pages to images
            for page_num in range(chunk_start, chunk_end):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                image_path = os.path.join(temp_dir, f"{Path(pdf_path).stem}_page_{page_num + 1}.png")
                pix.save(image_path)
                chunk_image_paths.append(image_path)
            
            # Process this chunk
            print(f"   Processing pages {chunk_start + 1}-{chunk_end}...")
            for i, img_path in enumerate(chunk_image_paths):
                page_num = chunk_start + i + 1
                page_text = self.ocr_image(img_path)
                all_text += f"\n--- Page {page_num} ---\n{page_text}\n"
            
            # Clean up chunk images immediately
            for img_path in chunk_image_paths:
                try:
                    os.remove(img_path)
                except:
                    pass
            
            # Clear memory after each chunk
            self._clear_memory()
        
        doc.close()
        return all_text

    def analyze_content(self, text):
        """Analyze extracted text for metadata"""
        metadata = {
            'word_count': len(text.split()),
            'has_tables': '<table>' in text.lower() or 'html' in text.lower(),
            'has_equations': '$' in text or 'latex' in text.lower(),
            'has_images': '<img>' in text.lower(),
            'has_watermarks': '<watermark>' in text.lower(),
            'has_page_numbers': '<page_number>' in text.lower(),
            'character_count': len(text)
        }
        return metadata

    def get_file_info(self, file_path):
        """Get basic file information"""
        stat = os.stat(file_path)
        return {
            'file_size': stat.st_size,
            'creation_time': datetime.fromtimestamp(stat.st_ctime),
            'modification_time': datetime.fromtimestamp(stat.st_mtime),
            'file_extension': Path(file_path).suffix.lower()
        }

    def process_single_file(self, file_path):
        """Process a single file and return metadata"""
        start_time = time.time()
        file_info = self.get_file_info(file_path)
        filename = Path(file_path).name
        
        print(f"├── Loading file... ✓")
        
        try:
            if file_info['file_extension'] == '.pdf':
                print(f"├── Converting and processing PDF in chunks...")
                extracted_text = self.pdf_to_images_chunked(file_path)
                # Count pages by counting page markers
                page_count = extracted_text.count("--- Page ")
            else:
                print(f"├── Running OCR...")
                extracted_text = self.ocr_image(file_path)
                page_count = 1
            
            print(f"├── Analyzing content... ✓")
            content_metadata = self.analyze_content(extracted_text)
            
            processing_time = time.time() - start_time
            
            result = {
                'filename': filename,
                'file_path': str(file_path),
                'file_size_bytes': file_info['file_size'],
                'file_format': file_info['file_extension'],
                'creation_time': file_info['creation_time'].isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'processing_date': datetime.now().isoformat(),
                'page_count': page_count,
                'word_count': content_metadata['word_count'],
                'character_count': content_metadata['character_count'],
                'has_tables': content_metadata['has_tables'],
                'has_equations': content_metadata['has_equations'],
                'has_images': content_metadata['has_images'],
                'has_watermarks': content_metadata['has_watermarks'],
                'has_page_numbers': content_metadata['has_page_numbers'],
                'extracted_text': extracted_text,
                'status': 'success'
            }
            
            print(f"└── ✅ Complete! ({processing_time:.1f}s)")
            print(f"    Results: {content_metadata['word_count']} words, "
                  f"{'tables detected' if content_metadata['has_tables'] else 'no tables'}, "
                  f"{'equations detected' if content_metadata['has_equations'] else 'no equations'}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_result = {
                'filename': filename,
                'file_path': str(file_path),
                'file_size_bytes': file_info['file_size'],
                'file_format': file_info['file_extension'],
                'creation_time': file_info['creation_time'].isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'processing_date': datetime.now().isoformat(),
                'page_count': 0,
                'word_count': 0,
                'character_count': 0,
                'has_tables': False,
                'has_equations': False,
                'has_images': False,
                'has_watermarks': False,
                'has_page_numbers': False,
                'extracted_text': '',
                'status': 'failed',
                'error_message': str(e)
            }
            
            print(f"└── ❌ Failed: {str(e)}")
            self.failed_files.append(filename)
            return error_result

    def _format_markdown_content(self, result):
        """Format the content for markdown files with metadata"""
        content = []
        content.append("# OCR Extraction Results")
        content.append("")
        
        # Document Information
        content.append("## Document Information")
        content.append("")
        content.append(f"- **Filename:** {result['filename']}")
        content.append(f"- **File Format:** {result['file_format']}")
        content.append(f"- **File Size:** {result['file_size_bytes']:,} bytes")
        content.append(f"- **Created:** {result['creation_time']}")
        content.append("")
        
        # Processing Information
        content.append("## Processing Information")
        content.append("")
        content.append(f"- **Processed:** {result['processing_date']}")
        content.append(f"- **Processing Time:** {result['processing_time_seconds']} seconds")
        content.append(f"- **Pages Processed:** {result['page_count']}")
        content.append("")
        
        # Content Analysis
        content.append("## Content Analysis")
        content.append("")
        content.append(f"- **Word Count:** {result['word_count']:,}")
        content.append(f"- **Character Count:** {result['character_count']:,}")
        content.append(f"- **Contains Tables:** {'✅ Yes' if result['has_tables'] else '❌ No'}")
        content.append(f"- **Contains Equations:** {'✅ Yes' if result['has_equations'] else '❌ No'}")
        content.append(f"- **Contains Images:** {'✅ Yes' if result['has_images'] else '❌ No'}")
        content.append(f"- **Contains Watermarks:** {'✅ Yes' if result['has_watermarks'] else '❌ No'}")
        content.append(f"- **Contains Page Numbers:** {'✅ Yes' if result['has_page_numbers'] else '❌ No'}")
        content.append("")
        
        # Extracted Text
        content.append("## Extracted Text")
        content.append("")
        content.append("---")
        content.append("")
        content.append(result['extracted_text'])
        
        return "\n".join(content)

    def output_file_exists(self, file_path, output_dir="extracted_text"):
        """Check if output file already exists for the given input file"""
        # Create safe filename
        original_path = Path(file_path)
        relative_path_str = str(original_path.relative_to(original_path.parts[0]))
        safe_filename = relative_path_str.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
        base_name = Path(safe_filename).stem
        
        md_file = os.path.join(output_dir, f"{base_name}.md")
        return os.path.exists(md_file)

    def get_output_filename(self, file_path, output_dir="extracted_text"):
        """Get the expected output filename for a given input file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create safe filename
        original_path = Path(file_path)
        relative_path_str = str(original_path.relative_to(original_path.parts[0]))
        safe_filename = relative_path_str.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
        base_name = Path(safe_filename).stem
        
        md_file = os.path.join(output_dir, f"{base_name}.md")
        
        # Handle filename conflicts
        counter = 1
        while os.path.exists(md_file):
            base_name_with_counter = f"{Path(safe_filename).stem}_{counter}"
            md_file = os.path.join(output_dir, f"{base_name_with_counter}.md")
            counter += 1
        
        return md_file

    def save_result_immediately(self, result, output_dir="extracted_text"):
        """Save result immediately to disk to avoid memory accumulation"""
        # Save markdown file only
        if result['status'] == 'success' and result['extracted_text']:
            md_file = self.get_output_filename(result['file_path'], output_dir)
            
            # Save markdown file
            content = self._format_markdown_content(result)
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)

    def process_documents_folder(self, folder_path="example_files", output_dir="extracted_text", csv_file="ocr_results.csv"):
        """Process all supported documents in the specified folder with memory optimization"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ Folder '{folder_path}' does not exist!")
            return
        
        # Find all supported files recursively
        with self.timing_profiler.time_operation("file_discovery"):
            all_files = []
            for ext in self.supported_formats:
                all_files.extend(folder_path.rglob(f"*{ext}"))
                all_files.extend(folder_path.rglob(f"*{ext.upper()}"))
        
        self.total_files = len(all_files)
        
        if self.total_files == 0:
            print(f"📁 No supported documents found in '{folder_path}'")
            print(f"   Supported formats: {', '.join(self.supported_formats)}")
            return
        
        print(f"\n🔍 Found {self.total_files} files to process")
        print(f"📂 Processing documents from: {folder_path}")
        print(f"💾 Results will be saved to: {output_dir}/")
        print("=" * 60)
        
        # Filter out files that already have output files
        with self.timing_profiler.time_operation("output_file_checking"):
            files_to_process = []
            skipped_files = 0
            
            for file_path in all_files:
                if self.output_file_exists(file_path, output_dir):
                    skipped_files += 1
                    print(f"⏭️  Skipping {file_path.name} (output already exists)")
                else:
                    files_to_process.append(file_path)
            
            if skipped_files > 0:
                print(f"\n📋 Skipped {skipped_files} files with existing output")
                print(f"🔄 Processing {len(files_to_process)} remaining files")
                self.total_files = len(files_to_process)
                
                if self.total_files == 0:
                    print("✅ All files already processed!")
                    return
        
        # Clear existing CSV file
        if os.path.exists(csv_file):
            os.remove(csv_file)
        
        start_time = time.time()
        successful_files = 0
        
        # Load model once for all batches
        with self.model_context():
            # Process files in batches
            for batch_start in range(0, self.total_files, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.total_files)
                batch_files = files_to_process[batch_start:batch_end]
                
                print(f"\n🔄 Processing batch {batch_start//self.batch_size + 1} "
                      f"(files {batch_start + 1}-{batch_end})")
                # Separate image files from PDFs for true batch processing
                image_files = [f for f in batch_files if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}]
                pdf_files = [f for f in batch_files if Path(f).suffix.lower() == '.pdf']
                
                # Process image files in true batches
                if image_files:
                    print(f"🖼️  Batch processing {len(image_files)} image files...")
                    try:
                        # Process all images in a single batch call
                        batch_texts = self.ocr_images_batch([str(f) for f in image_files])
                        
                        # Process results
                        for i, (file_path, extracted_text) in enumerate(zip(image_files, batch_texts)):
                            file_index = batch_start + len(pdf_files) + i + 1
                            self.processed_files = file_index
                            elapsed_time = time.time() - start_time
                            avg_time_per_file = elapsed_time / file_index if file_index > 0 else 0
                            estimated_remaining = avg_time_per_file * (self.total_files - file_index)
                            
                            print(f"\n[{file_index}/{self.total_files}] Processing: {file_path.name}")
                            print(f"⏱️  Elapsed: {elapsed_time:.1f}s | Est. remaining: {estimated_remaining:.1f}s")
                            print(f"├── Loading file... ✓")
                            print(f"├── Running batch OCR... ✓")
                            print(f"├── Analyzing content... ✓")
                            
                            # Create result metadata
                            start_time_single = time.time()
                            file_info = self.get_file_info(file_path)
                            content_metadata = self.analyze_content(extracted_text)
                            processing_time = time.time() - start_time_single
                            
                            result = {
                                'filename': Path(file_path).name,
                                'file_path': str(file_path),
                                'file_size_bytes': file_info['file_size'],
                                'file_format': file_info['file_extension'],
                                'creation_time': file_info['creation_time'].isoformat(),
                                'processing_time_seconds': round(processing_time, 2),
                                'processing_date': datetime.now().isoformat(),
                                'page_count': 1,
                                'word_count': content_metadata['word_count'],
                                'character_count': content_metadata['character_count'],
                                'has_tables': content_metadata['has_tables'],
                                'has_equations': content_metadata['has_equations'],
                                'has_images': content_metadata['has_images'],
                                'has_watermarks': content_metadata['has_watermarks'],
                                'has_page_numbers': content_metadata['has_page_numbers'],
                                'extracted_text': extracted_text,
                                'status': 'success'
                            }
                            
                            print(f"└── ✅ Complete! ({processing_time:.1f}s)")
                            print(f"    Results: {content_metadata['word_count']} words, "
                                  f"{'tables detected' if content_metadata['has_tables'] else 'no tables'}, "
                                  f"{'equations detected' if content_metadata['has_equations'] else 'no equations'}")
                            
                            # Save result immediately
                            self.save_result_immediately(result, output_dir)
                            successful_files += 1
                            
                    except Exception as e:
                        print(f"❌ Batch processing failed: {e}")
                        # Fall back to individual processing
                        for file_path in image_files:
                            file_index = batch_start + len(pdf_files) + image_files.index(file_path) + 1
                            result = self.process_single_file(file_path)
                            self.save_result_immediately(result, output_dir)
                            if result['status'] == 'success':
                                successful_files += 1
                
                # Process PDF files individually (they need special handling)
                for i, file_path in enumerate(pdf_files):
                    file_index = batch_start + i + 1
                    self.processed_files = file_index
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / file_index if file_index > 0 else 0
                    estimated_remaining = avg_time_per_file * (self.total_files - file_index)
                    
                    print(f"\n[{file_index}/{self.total_files}] Processing: {file_path.name}")
                    print(f"⏱️  Elapsed: {elapsed_time:.1f}s | Est. remaining: {estimated_remaining:.1f}s")
                    
                    result = self.process_single_file(file_path)
                    
                    # Save result immediately
                    self.save_result_immediately(result, output_dir)
                    
                    if result['status'] == 'success':
                        successful_files += 1
                    
                    # Clear memory after each file
                    self._clear_memory()
        
        # Model is automatically unloaded when exiting the context
        print(f"✅ All batches complete, model unloaded")
        
        total_time = time.time() - start_time
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 PROCESSING COMPLETE!")
        print(f"📊 Total files processed: {self.total_files}")
        print(f"✅ Successful: {successful_files}")
        print(f"❌ Failed: {len(self.failed_files)}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📈 Average time per file: {total_time/self.total_files:.1f}s")
        
        if self.failed_files:
            print(f"\n❌ Failed files: {', '.join(self.failed_files)}")
        
        print(f"\n💾 Results saved to:")
        print(f"   📄 Markdown files: {output_dir}/")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory-Optimized Batch OCR Processing')
    
    # Create mutually exclusive group for CPU/GPU selection
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--cpu', action='store_true', help='Use CPU for processing (default)')
    device_group.add_argument('--gpu', action='store_true', help='Use GPU for processing')
    
    parser.add_argument('--input-dir', default='example_files', help='Input directory containing documents (default: example_files)')
    parser.add_argument('--output-dir', default='extracted_text', help='Output directory for markdown files (default: extracted_text)')
    parser.add_argument('--csv-file', default='ocr_results.csv', help='CSV file for results summary (default: ocr_results.csv)')
    parser.add_argument('--batch-size', type=int, default=2, help='Number of files to process per batch (default: 2)')
    parser.add_argument('--pdf-chunk-size', type=int, default=1, help='Number of PDF pages to process at once (default: 1)')
    parser.add_argument('--max-image-width', type=int, default=1080, help='Maximum image width in pixels (default: 1080)')
    parser.add_argument('--max-image-height', type=int, default=1080, help='Maximum image height in pixels (default: 1080)')
    
    args = parser.parse_args()
    
    # Determine device usage - CPU is default when neither flag is specified
    if args.gpu:
        use_cpu = False
    else:
        use_cpu = True  # Default to CPU, whether --cpu is specified or neither flag is used
    processor = MemoryOptimizedBatchOCRProcessor(
        use_cpu=use_cpu,
        batch_size=args.batch_size,
        max_pdf_pages_per_chunk=args.pdf_chunk_size,
        max_image_size=(args.max_image_width, args.max_image_height)
    )
    
    # Process documents
    processor.process_documents_folder(args.input_dir, args.output_dir, args.csv_file)

if __name__ == "__main__":
    main()
    # export TRANSFORMERS_VERBOSITY=error && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python batch_ocr.py --gpu --batch-size 4 --pdf-chunk-size 1 --max-image-width 1024 --max-image-height 1024 --input-dir "example_files"
