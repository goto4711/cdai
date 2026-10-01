import os
import datetime
import nbformat

# Times in the export stamp are shown in this time zone (Colab itself runs in UTC).
STAMP_TIMEZONE = "Europe/Amsterdam"


def _now_local():
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo(STAMP_TIMEZONE))
    except Exception:
        return datetime.datetime.now(datetime.timezone.utc)


def _fmt_time(dt):
    try:
        from zoneinfo import ZoneInfo
        dt = dt.astimezone(ZoneInfo(STAMP_TIMEZONE))
    except Exception:
        pass
    return dt.strftime("%d %b %Y, %H:%M %Z").lstrip("0")


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
    """Fallback: mount Drive and search for the saved notebook file by name.

    Collects every match. An exact name beats a variant ("Copy of ...",
    "Solution_..."); within the same kind, the most recently saved file wins.
    Returns (path, modified_datetime) or (None, None).
    """
    from google.colab import drive

    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')

    print(f"Searching for '{notebook_name}'...")
    search_root = '/content/drive/MyDrive'
    target = notebook_name.lower()

    matches = []  # (is_exact, mtime, path)
    for root, dirs, files_in_dir in os.walk(search_root):
        # Skip trashed and hidden folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d.lower() != 'trash']
        for file in files_in_dir:
            fname = file.lower()
            if fname == target or fname.endswith(target):
                path = os.path.join(root, file)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                matches.append((fname == target, mtime, path))

    if not matches:
        return None, None

    matches.sort(key=lambda m: (m[0], m[1]), reverse=True)
    if len(matches) > 1:
        print(f"   - Found {len(matches)} files that match:")
        for is_exact, mtime, path in matches:
            when = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc)
            print(f"       {'exact  ' if is_exact else 'variant'}  last saved {_fmt_time(when)}  {path}")

    is_exact, mtime, path = matches[0]
    saved = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc)
    print(f"   - Using: {path}")
    print(f"     (last saved {_fmt_time(saved)}). If this is not your latest version, "
          "save the notebook (Ctrl+S) and run this cell again.")
    return path, saved


def _check_run_order(nb):
    """Warn (never stop) if code cells were not run, or not run top to bottom."""
    def _text(src):  # source can be a string or a list of lines
        return ''.join(src) if isinstance(src, list) else (src or '')

    counts = [c.get('execution_count') for c in nb.cells
              if c.cell_type == 'code' and _text(c.get('source')).strip()]
    not_run = sum(1 for n in counts if n is None)
    ran = [n for n in counts if n is not None]
    out_of_order = sum(1 for a, b in zip(ran, ran[1:]) if b < a)

    if not_run:
        print(f"   - Note: {not_run} code cell(s) have not been run, so they have no results in the HTML.")
    if out_of_order:
        print(f"   - Note: the cells were not run strictly from top to bottom ({out_of_order} place(s)). "
              "That is fine if you re-ran some cells, but check that the HTML shows the results you want.")


def _stamp(nb, notebook_name, method_note):
    """Put a short line at the top of the export saying when and how it was made."""
    text = f"*Exported {_fmt_time(_now_local())} from `{notebook_name}` ({method_note}).*"
    cell = nbformat.from_dict({"cell_type": "markdown", "metadata": {}, "source": text})
    if nb.get("nbformat", 4) == 4 and nb.get("nbformat_minor", 0) >= 5:
        cell["id"] = "export-stamp"  # cell ids are only allowed from nbformat 4.5 on
    nb.cells.insert(0, cell)


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
        method_note = "live capture of the notebook on screen"
    else:
        if not notebook_name:
            print("❌ Error: could not capture the live notebook and no filename was given.")
            print("   Tip: pass the name, e.g. make_html('Week4_Workshop.ipynb').")
            return
        if not notebook_name.endswith('.ipynb'):
            notebook_name += '.ipynb'
        found_path, saved = _find_in_drive(notebook_name)
        if not found_path:
            print(f"❌ Error: Could not find '{notebook_name}' in Google Drive.")
            print("   Tip: SAVE the notebook (Ctrl+S) and check that the name matches.")
            return
        try:
            with open(found_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
        except Exception as e:
            print(f"❌ Error reading notebook: {e}")
            return
        drive_path = found_path.replace('/content/drive/MyDrive/', 'My Drive/')
        method_note = f"saved copy in Google Drive: {drive_path}, last saved {_fmt_time(saved)}"

    if not notebook_name.endswith('.ipynb'):
        notebook_name += '.ipynb'

    # --- 2. Sanity-check outputs and run order (non-blocking: never halts a 'Run all') ---
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    total_code = len(code_cells)
    with_output = sum(1 for c in code_cells if c.get('outputs'))
    print(f"   - Status: {with_output}/{total_code} code cells have outputs.")

    if total_code > 0 and with_output == 0:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: NO OUTPUTS DETECTED!")
        print("The HTML will contain your code but NO results or plots.")
        print("If that's not what you want: run all cells, then re-run this cell.")
        print("Creating the HTML anyway so your run is not interrupted...")
        print("=" * 60 + "\n")

    _check_run_order(nb)

    # --- 3. Clean corrupt widget metadata (fixes 'state' KeyErrors) ---
    if 'widgets' in nb.metadata:
        del nb.metadata['widgets']
        print("   - Cleaned corrupt widget metadata.")

    # --- 4. Stamp, write a temp copy and convert to HTML ---
    _stamp(nb, notebook_name, method_note)

    temp_nb_path = "/content/temp_conversion_source.ipynb"
    with open(temp_nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("Converting to HTML...")
    # --template classic keeps compatibility with educational tools.
    exit_code = os.system(
        f'jupyter nbconvert --to html --template classic "{temp_nb_path}"'
    )
    if exit_code != 0:
        print("❌ Error: the nbconvert command failed.")
        return

    # --- 5. Rename and download ---
    temp_html_path = temp_nb_path.replace('.ipynb', '.html')
    final_output_name = notebook_name.replace('.ipynb', '.html')

    if os.path.exists(temp_html_path):
        dest_path = f"/content/{final_output_name}"
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_html_path, dest_path)
        print(f"✅ Success. Downloading {final_output_name} ...")
        files.download(dest_path)
    else:
        print("❌ Error: HTML file was not created.")
