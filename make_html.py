import os
import json
import datetime
import nbformat

VERSION = "2026-10-11"

# Times in the export stamp are shown in this time zone (Colab itself runs in UTC).
STAMP_TIMEZONE = "Europe/Amsterdam"

# Where Colab keeps its files, and where Google Drive is mounted.
WORK_DIR = "/content"
DRIVE_ROOT = "/content/drive/MyDrive"

# Folders the Drive search never enters: Colab's trash and the large shared model caches.
SKIP_FOLDERS = {"trash", ".trash", "cda-cache"}


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
        payload = _message.blocking_request("get_ipynb", request="", timeout_sec=120)
        # Read it like a file, so that cell sources (lists of lines) are joined properly
        nb = nbformat.reads(json.dumps(payload["ipynb"]), as_version=4)
        name = nb.metadata.get("colab", {}).get("name")
        return nb, name
    except Exception as e:
        print(f"   - Live capture unavailable ({e}); will look in Google Drive instead.")
        return None, None


def _find_in_drive(notebook_name):
    """Fallback: mount Drive and search for the saved notebook file by name.

    Looks in 'Colab Notebooks' first, then in the rest of My Drive. An exact name
    beats a variant ("Copy of ...", "Solution_..."); within the same kind, the most
    recently saved file wins. Returns (path, modified_datetime) or (None, None).
    """
    from google.colab import drive

    if not os.path.exists(DRIVE_ROOT):
        print("Mounting Google Drive...")
        drive.mount(os.path.dirname(DRIVE_ROOT))

    print(f"Searching for '{notebook_name}'...")
    target = notebook_name.lower()

    def search(top, skip=()):
        found = []  # (is_exact, mtime, path)
        for root, dirs, files_in_dir in os.walk(top):
            dirs[:] = [d for d in dirs
                       if not d.startswith('.') and d.lower() not in SKIP_FOLDERS
                       and os.path.join(root, d) not in skip]
            for file in files_in_dir:
                fname = file.lower()
                if fname == target or fname.endswith(target):
                    path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(path)
                    except OSError:
                        continue
                    found.append((fname == target, mtime, path))
        return found

    colab_folder = os.path.join(DRIVE_ROOT, "Colab Notebooks")
    matches = search(colab_folder) if os.path.isdir(colab_folder) else []
    if not matches:
        matches = search(DRIVE_ROOT, skip={colab_folder})
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

    # The cell that calls make_html() is still running while the notebook is captured,
    # so it has no execution count yet - leave it out of the check.
    counts = [c.get('execution_count') for c in nb.cells
              if c.cell_type == 'code' and _text(c.get('source')).strip()
              and 'make_html(' not in _text(c.get('source'))]
    not_run = sum(1 for n in counts if n is None)
    ran = [n for n in counts if n is not None]
    out_of_order = sum(1 for a, b in zip(ran, ran[1:]) if b < a)

    if not_run:
        print(f"   - Note: {not_run} code cell(s) have not been run, so they have no results in the HTML.")
    if out_of_order:
        print(f"   - Note: the cells were not run strictly from top to bottom ({out_of_order} place(s)). "
              "That is fine if you re-ran some cells, but check that the HTML shows the results you want.")


def _clean_widgets(nb):
    """Remove widget state, which often breaks the conversion ('state' KeyErrors).

    Widgets (sliders, progress bars) cannot work in a static HTML file anyway; their
    text versions stay in the output.
    """
    cleaned = False
    if 'widgets' in nb.metadata:
        del nb.metadata['widgets']
        cleaned = True
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        if 'widgets' in cell.get('metadata', {}):
            del cell.metadata['widgets']
            cleaned = True
        for out in cell.get('outputs', []):
            data = out.get('data')
            if data and 'application/vnd.jupyter.widget-view+json' in data:
                del data['application/vnd.jupyter.widget-view+json']
                cleaned = True
    return cleaned


def _stamp(nb, notebook_name, method_note):
    """Put a short line at the top of the export saying when and how it was made."""
    text = (f"*Exported {_fmt_time(_now_local())} from `{notebook_name}` "
            f"({method_note}; make_html {VERSION}).*")
    cell = nbformat.from_dict({"cell_type": "markdown", "metadata": {}, "source": text})
    if nb.get("nbformat", 4) == 4 and nb.get("nbformat_minor", 0) >= 5:
        cell["id"] = "export-stamp"  # cell ids are only allowed from nbformat 4.5 on
    nb.cells.insert(0, cell)


def _to_html(nb):
    """Convert in Python, so that any error is shown in the cell."""
    from nbconvert import HTMLExporter
    try:
        exporter = HTMLExporter(template_name="classic")
    except Exception:
        exporter = HTMLExporter()  # older or newer nbconvert without the classic template
    body, _ = exporter.from_notebook_node(nb)
    return body


def _download(dest_path):
    """Download the file, and say where it is in case the browser blocks the download."""
    name = os.path.basename(dest_path)
    try:
        from google.colab import files
        files.download(dest_path)
        print(f"✅ Success. Your browser should now download {name}.")
        print(f"   If no download appears: click the folder icon on the left of Colab, find '{name}', "
              "click the three dots next to it and choose Download.")
        return
    except Exception as e:
        print(f"⚠️  The automatic download did not start ({e}).")
    print(f"   Click the folder icon on the left of Colab, find '{name}', "
          "click the three dots next to it and choose Download.")
    if os.path.isdir(DRIVE_ROOT):
        try:
            import shutil
            shutil.copy(dest_path, os.path.join(DRIVE_ROOT, name))
            print(f"   A copy is also in your Google Drive: My Drive/{name}")
        except Exception:
            pass


def make_html(notebook_name=None):
    """
    Export the current notebook to HTML for Canvas submission.

    Preferred path: capture the LIVE notebook from the Colab frontend, so the
    export always reflects what is on screen right now -- no save required, and
    the filename is detected automatically. If that is unavailable, it falls
    back to searching Google Drive for a saved file called `notebook_name`.
    `notebook_name` also names the HTML file.

    Usage (Colab):  make_html()                         # auto-detect everything
                    make_html("Week4_Workshop.ipynb")   # name the HTML file, and the fallback search
    """
    print(f"make_html {VERSION}")

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
        try:
            found_path, saved = _find_in_drive(notebook_name)
        except Exception as e:
            print(f"❌ Error while searching Google Drive: {e}")
            return
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
        drive_path = found_path.replace(DRIVE_ROOT + '/', 'My Drive/')
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
        print("If you did run them: in Colab, go to Edit → Notebook settings and untick")
        print("'Omit code cell output when saving this notebook'. Then run all cells again.")
        print("Creating the HTML anyway so your run is not interrupted...")
        print("=" * 60 + "\n")

    _check_run_order(nb)

    # --- 3. Clean widget data (fixes 'state' KeyErrors) ---
    _clean_widgets(nb)

    # --- 4. Stamp and convert to HTML ---
    _stamp(nb, notebook_name, method_note)

    print("Converting to HTML...")
    try:
        body = _to_html(nb)
    except Exception as e:
        print(f"❌ Error: the conversion to HTML failed: {type(e).__name__}: {e}")
        print("   Tip: restart the session (Runtime → Restart session), run all cells, and try again.")
        return

    # --- 5. Write and download ---
    dest_path = os.path.join(WORK_DIR, notebook_name[:-len('.ipynb')] + '.html')
    try:
        with open(dest_path, 'w', encoding='utf-8') as f:
            f.write(body)
    except Exception as e:
        print(f"❌ Error: could not write the HTML file: {e}")
        return
    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"   - Wrote {os.path.basename(dest_path)} ({size_mb:.1f} MB).")
    _download(dest_path)
