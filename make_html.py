import os
import nbformat


def _capture_live_notebook():
    """Grab the notebook exactly as it is on screen, via the Colab frontend.

    Returns (nb_node, detected_name), or (None, None) if it isn't available
    (e.g. not running in Colab, or the frontend API changed).
    """
    try:
        from google.colab import _message
        payload = _message.blocking_request("get_ipynb", request="", timeout_sec=60)
        nb = nbformat.from_dict(payload["ipynb"])
        name = nb.metadata.get("colab", {}).get("name")
        return nb, name
    except Exception as e:
        print(f"   - Live capture unavailable ({e}); will look in Google Drive instead.")
        return None, None


def _find_in_drive(notebook_name):
    """Fallback: mount Drive and search for the saved notebook file by name."""
    from google.colab import drive

    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')

    print(f"Searching for '{notebook_name}'...")
    search_root = '/content/drive/MyDrive'
    target = notebook_name.lower()

    for root, dirs, files_in_dir in os.walk(search_root):
        for file in files_in_dir:
            fname = file.lower()
            # Exact name
            if fname == target:
                path = os.path.join(root, file)
                print(f"Found exact match: {path}")
                return path
            # "Copy of ...", "Solution_...", etc.
            if fname.endswith(target) and fname != target:
                path = os.path.join(root, file)
                print(f"Found matching variant: {path}")
                return path
    return None


def make_html(notebook_name=None):
    """
    Export the current notebook to HTML for Canvas submission.

    Preferred path: capture the LIVE notebook from the Colab frontend, so the
    export always reflects what is on screen right now -- no save required, and
    the filename is detected automatically. If that is unavailable, it falls
    back to searching Google Drive for a saved file called `notebook_name`.

    Usage (Colab):  make_html()                         # auto-detect everything
                    make_html("Week4_Workshop.ipynb")   # force a name if needed
    """
    from google.colab import files

    # --- 1. Get the notebook (live first, Drive as fallback) ---
    print("Capturing notebook...")
    nb, detected_name = _capture_live_notebook()

    if nb is not None:
        print("   - Captured the live notebook directly (no save needed).")
        if not notebook_name:
            notebook_name = detected_name or "notebook.ipynb"
    else:
        if not notebook_name:
            print("\u274c Error: could not capture the live notebook and no filename was given.")
            print("   Tip: pass the name, e.g. make_html('Week4_Workshop.ipynb').")
            return
        if not notebook_name.endswith('.ipynb'):
            notebook_name += '.ipynb'
        found_path = _find_in_drive(notebook_name)
        if not found_path:
            print(f"\u274c Error: Could not find '{notebook_name}' in Google Drive.")
            print("   Tip: SAVE the notebook (Ctrl+S) and check that the name matches.")
            return
        try:
            with open(found_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
        except Exception as e:
            print(f"\u274c Error reading notebook: {e}")
            return

    if not notebook_name.endswith('.ipynb'):
        notebook_name += '.ipynb'

    # --- 2. Sanity-check outputs (non-blocking: never halts a 'Run all') ---
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    total_code = len(code_cells)
    with_output = sum(1 for c in code_cells if c.get('outputs'))
    print(f"   - Status: {with_output}/{total_code} code cells have outputs.")

    if total_code > 0 and with_output == 0:
        print("\n" + "=" * 60)
        print("\u26a0\ufe0f  WARNING: NO OUTPUTS DETECTED!")
        print("The HTML will contain your code but NO results or plots.")
        print("If that's not what you want: run all cells, then re-run this cell.")
        print("Creating the HTML anyway so your run is not interrupted...")
        print("=" * 60 + "\n")

    # --- 3. Clean corrupt widget metadata (fixes 'state' KeyErrors) ---
    if 'widgets' in nb.metadata:
        del nb.metadata['widgets']
        print("   - Cleaned corrupt widget metadata.")

    # --- 4. Write a temp copy and convert to HTML ---
    temp_nb_path = "/content/temp_conversion_source.ipynb"
    with open(temp_nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("Converting to HTML...")
    # --template classic keeps compatibility with educational tools.
    exit_code = os.system(
        f'jupyter nbconvert --to html --template classic "{temp_nb_path}"'
    )
    if exit_code != 0:
        print("\u274c Error: the nbconvert command failed.")
        return

    # --- 5. Rename and download ---
    temp_html_path = temp_nb_path.replace('.ipynb', '.html')
    final_output_name = notebook_name.replace('.ipynb', '.html')

    if os.path.exists(temp_html_path):
        dest_path = f"/content/{final_output_name}"
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_html_path, dest_path)
        print(f"\u2705 Success. Downloading {final_output_name} ...")
        files.download(dest_path)
    else:
        print("\u274c Error: HTML file was not created.")
