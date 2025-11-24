import os
import nbformat
from google.colab import drive, files

def make_html(notebook_name):
    """
    Searches for a notebook in Google Drive, cleans corrupt widget metadata, 
    converts it to HTML (static), and downloads it.
    """
    
    # 1. Mount Drive if needed
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')

    # 2. Handle file extension
    if not notebook_name.endswith('.ipynb'):
        notebook_name += '.ipynb'

    print(f"Searching for '{notebook_name}'...")

    # 3. Search for the file
    found_path = None
    search_root = '/content/drive/MyDrive'
    
    target = notebook_name.lower()

    for root, dirs, files_in_dir in os.walk(search_root):
        for file in files_in_dir:
            fname = file.lower()

            # 1. exact match
            if fname == target:
                found_path = os.path.join(root, file)
                print(f"Found exact match: {found_path}")
                break

            # 2. matches "Copy of ...", "Copy of Copy of ...", etc.
            if fname.endswith(target):
                found_path = os.path.join(root, file)
                print(f"Found matching variant: {found_path}")
                break

        if found_path:
            break

    
    #for root, dirs, files_in_dir in os.walk(search_root):
    #    if notebook_name in files_in_dir:
    #        found_path = os.path.join(root, notebook_name)
    #        print(f"Found: {found_path}")
    #        break

    if not found_path:
        print(f"❌ Error: Could not find '{notebook_name}' in Google Drive. (Did you save it?)")
        return

    # 4. Clean Metadata (Fixes KeyError: 'state')
    print("Preparing and cleaning notebook...")
    try:
        with open(found_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Remove broken widget metadata if present
        if 'widgets' in nb.metadata:
            del nb.metadata['widgets']
            print(" - Cleaned corrupt widget metadata.")
        
        # Save to a temporary file in the VM
        temp_nb_path = "/content/temp_conversion_source.ipynb"
        with open(temp_nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    except Exception as e:
        print(f"❌ Error reading notebook structure: {e}")
        return

    # 5. Convert to HTML
    print("Converting to HTML...")
    # We use os.system to run the shell command safely within the python function
    # --template classic is used for maximum compatibility
    exit_code = os.system(f'jupyter nbconvert --to html --template classic "{temp_nb_path}"')
    
    if exit_code != 0:
        print("❌ Error: Conversion command failed.")
        return

    # 6. Rename and Download
    temp_html_path = temp_nb_path.replace('.ipynb', '.html')
    final_output_name = notebook_name.replace('.ipynb', '.html')

    if os.path.exists(temp_html_path):
        # Rename the temp file to the actual notebook name
        if os.path.exists(f"/content/{final_output_name}"):
            os.remove(f"/content/{final_output_name}")
            
        os.rename(temp_html_path, f"/content/{final_output_name}")
        
        print(f"✅ Success. Downloading {final_output_name}...")
        files.download(f"/content/{final_output_name}")
    else:
        print("❌ Error: HTML file was not created.")
