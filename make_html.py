import os
import nbformat
from google.colab import drive, files

def make_html(notebook_name):
    """
    Searches for a notebook in Google Drive, checks for saved outputs, 
    cleans corrupt widget metadata, converts to HTML, and downloads it.
    """
    
    # 1. Mount Drive if needed
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')

    # 2. Handle file extension
    if not notebook_name.endswith('.ipynb'):
        notebook_name += '.ipynb'

    print(f"Searching for '{notebook_name}'...")

    # 3. Search for the file in Google Drive
    found_path = None
    search_root = '/content/drive/MyDrive'
    
    target = notebook_name.lower()

    for root, dirs, files_in_dir in os.walk(search_root):
        for file in files_in_dir:
            fname = file.lower()

            # Match exact name
            if fname == target:
                found_path = os.path.join(root, file)
                print(f"Found exact match: {found_path}")
                break

            # Match "Copy of..." variants
            if fname.endswith(target) and fname != target:
                found_path = os.path.join(root, file)
                print(f"Found matching variant: {found_path}")
                break

        if found_path:
            break

    if not found_path:
        print(f"❌ Error: Could not find '{notebook_name}' in Google Drive.")
        print("   Tip: Ensure you have SAVED the notebook (Ctrl+S) and the name matches.")
        return

    # 4. Check for Outputs & Clean Metadata
    print("Checking notebook status...")
    try:
        with open(found_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # --- NEW SAFETY CHECK: Scan for outputs ---
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        total_code = len(code_cells)
        with_output = sum(1 for c in code_cells if c.get('outputs'))

        print(f"   - Status: {with_output}/{total_code} code cells have outputs.")

        # If there are code cells, but NO outputs, it's likely a save error.
        if total_code > 0 and with_output == 0:
            print("\n" + "="*60)
            print("⚠️ WARNING: NO OUTPUTS DETECTED! ⚠️")
            print("The HTML file will contain your code, but NO results or plots.")
            print("Possible causes:")
            print("   1. You ran the cells but forgot to SAVE (Ctrl+S).")
            print("   2. You really haven't run any code yet.")
            print("="*60)
            
            user_choice = input("Do you want to create the HTML anyway? (y/n): ")
            if user_choice.lower() != 'y':
                print("Aborted. Please Save your notebook and try again.")
                return
        # ------------------------------------------
        
        # Remove broken widget metadata if present (fixes 'state' KeyErrors)
        if 'widgets' in nb.metadata:
            del nb.metadata['widgets']
            print("   - Cleaned corrupt widget metadata.")
        
        # Save to a temporary file in the VM
        temp_nb_path = "/content/temp_conversion_source.ipynb"
        with open(temp_nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    except Exception as e:
        print(f"❌ Error reading notebook structure: {e}")
        return

    # 5. Convert to HTML
    print("Converting to HTML...")
    # Using --template classic for better compatibility with educational tools
    exit_code = os.system(f'jupyter nbconvert --to html --template classic "{temp_nb_path}"')
    
    if exit_code != 0:
        print("❌ Error: Conversion command failed.")
        return

    # 6. Rename and Download
    temp_html_path = temp_nb_path.replace('.ipynb', '.html')
    final_output_name = notebook_name.replace('.ipynb', '.html')

    if os.path.exists(temp_html_path):
        # Move to content root and rename
        dest_path = f"/content/{final_output_name}"
        if os.path.exists(dest_path):
            os.remove(dest_path)
            
        os.rename(temp_html_path, dest_path)
        
        print(f"✅ Success. Downloading {final_output_name}...")
        files.download(dest_path)
    else:
        print("❌ Error: HTML file was not created.")
