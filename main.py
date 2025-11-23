"""
FastAPI-based note-taking service with user authentication, folder organization,
attachments (files and audio), and password‑protected notes.  This application
stores its data in a local SQLite database and uses Jinja2 templates to
render HTML pages.  It implements the features described by the user:

* Users are authenticated via username/password.  Only an administrator may
  create new user accounts.
* Each user owns their own folders and notes; data is isolated between
  users.
* Notes can be organized into folders, edited with a rich text editor
  (powered by Quill), and protected with an optional password.  When a
  password is set on a note, a visitor must supply the correct password
  before the note contents are displayed.
* Notes support multiple attachments.  Uploaded files are saved on disk
  under the ``attachments/`` directory.  The service attempts to
  detect common audio formats by their filename extension and will render
  audio attachments with a built‑in player.

The implementation relies only on packages already available in the
runtime environment (FastAPI, Starlette, Jinja2 and argon2 for secure
hashing).  It does not require external package installation.  Database
operations use Python’s builtin ``sqlite3`` module.

Security considerations
=======================

* User passwords and note passwords are hashed using Argon2 via
  ``argon2-cffi``.  The ``PasswordHasher`` class generates a unique salt
  and uses a secure algorithm to derive a hash from the raw password.
  According to Django’s password management documentation, hashed passwords
  include the algorithm, salt and iterations (work factor) components
  separated by dollar signs【667483293786990†L532-L540】.  Verifying a
  password involves calling ``PasswordHasher.verify`` with the raw
  password and stored hash.

* File uploads are stored to disk using the standard FastAPI ``UploadFile``
  interface.  When handling uploaded files, the application writes
  incoming data in chunks, as recommended by Django’s file upload
  documentation【655965192673326†L134-L142】.  Each uploaded file is saved
  with a randomly generated prefix to avoid collisions.

* The rich text editor stores HTML in the database.  Django’s security
  guide warns that storing unsanitized HTML can expose an application to
  cross‑site scripting (XSS) attacks【539960806992458†L78-L119】.  In this
  implementation we deliberately use Quill, which produces a restricted
  subset of HTML and does not allow arbitrary script tags.  However,
  because user input is ultimately untrusted, the template uses
  Jinja2’s ``|safe`` filter only when displaying note content.  In a
  production system you should run HTML through a sanitizer such as
  ``bleach`` before rendering.

"""

import os
import secrets
import sqlite3
from datetime import datetime
from typing import List, Optional

from argon2 import PasswordHasher
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import hmac
import hashlib
from starlette.templating import Jinja2Templates

# Determine base directory and ensure required subdirectories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTACHMENTS_DIR = os.path.join(BASE_DIR, "attachments")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Path to the SQLite database file
DATABASE_PATH = os.path.join(BASE_DIR, "database.db")

app = FastAPI()

# Secret key used to sign session cookies.  In a production system this
# should be a long random string stored securely (e.g. environment
# variable).  It is used to generate an HMAC signature for the session
# cookie.
SECRET_KEY = "change-this-secret-key"

# Mount attachments directory so that uploaded files can be served
app.mount("/attachments", StaticFiles(directory=ATTACHMENTS_DIR), name="attachments")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Argon2 password hasher for both user and note passwords
ph = PasswordHasher()


def get_db_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row factory configured."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialise the database schema and create a default admin user."""
    conn = get_db_connection()
    cur = conn.cursor()
    # Create tables if they don’t exist
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            folder_id INTEGER,
            title TEXT NOT NULL,
            content TEXT,
            password TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (folder_id) REFERENCES folders(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER NOT NULL,
            filename TEXT,
            filepath TEXT,
            is_audio INTEGER DEFAULT 0,
            FOREIGN KEY (note_id) REFERENCES notes(id)
        )
        """
    )
    conn.commit()
    # Create default admin user if it doesn’t exist
    admin = cur.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
    if not admin:
        hashed_pwd = ph.hash("admin123")
        cur.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)",
            ("admin", hashed_pwd),
        )
        conn.commit()
    conn.close()


# Initialise the database at import time
init_db()


def sign_value(value: str) -> str:
    """Return the value concatenated with its HMAC signature.

    We compute a SHA‑256 HMAC using the SECRET_KEY over the UTF‑8 encoded
    value.  The resulting string has the form ``value|signature``.
    """
    signature = hmac.new(SECRET_KEY.encode(), value.encode(), hashlib.sha256).hexdigest()
    return f"{value}|{signature}"


def verify_signed_value(signed: str) -> Optional[str]:
    """Verify an HMAC‑signed value and return the original value if valid.

    If the signature is invalid or the cookie is malformed, returns ``None``.
    """
    if not signed:
        return None
    if "|" not in signed:
        return None
    value, signature = signed.rsplit("|", 1)
    expected = hmac.new(SECRET_KEY.encode(), value.encode(), hashlib.sha256).hexdigest()
    if hmac.compare_digest(expected, signature):
        return value
    return None


def get_current_user(request: Request) -> Optional[sqlite3.Row]:
    """Retrieve the currently logged‑in user from the signed session cookie."""
    session_cookie = request.cookies.get("session")
    user_id_str = verify_signed_value(session_cookie) if session_cookie else None
    if not user_id_str:
        return None
    try:
        user_id = int(user_id_str)
    except ValueError:
        return None
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user


def require_login(request: Request):
    """Ensure that a user is logged in.  Returns (ok, value).

    If the user is logged in, returns (True, userRow).  Otherwise returns
    (False, RedirectResponse) so that callers can early return.
    """
    user = get_current_user(request)
    if user is None:
        return False, RedirectResponse("/login")
    return True, user


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect visitors to the notes dashboard or login page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse("/notes")
    return RedirectResponse("/login")


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    """Render the login form."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    """Process the login submission."""
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if user:
        try:
            ph.verify(user["password"], password)
            # Upon successful login, issue a signed session cookie containing
            # the user ID.  The cookie is HTTP‑only to mitigate XSS attacks.
            response = RedirectResponse("/notes", status_code=302)
            signed = sign_value(str(user["id"]))
            response.set_cookie(
                key="session",
                value=signed,
                httponly=True,
                samesite="lax",
            )
            return response
        except Exception:
            pass  # Verification failed
    # Invalid credentials
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Неверное имя пользователя или пароль"},
    )


@app.get("/logout")
async def logout(request: Request):
    """Log the user out by clearing the session."""
    # Remove the session cookie by setting an empty value and max_age=0
    response = RedirectResponse("/login")
    response.delete_cookie("session")
    return response


@app.get("/create_user", response_class=HTMLResponse)
async def create_user_get(request: Request):
    """Render the user creation form (admin only)."""
    ok, user = require_login(request)
    if not ok:
        return user
    if user["is_admin"] != 1:
        return RedirectResponse("/notes")
    # Pass user into template context so that navigation can reflect admin status
    return templates.TemplateResponse(
        "create_user.html",
        {"request": request, "user": user},
    )


@app.post("/create_user")
async def create_user_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    is_admin: Optional[str] = Form(None),
):
    """Create a new user (admin only)."""
    ok, user = require_login(request)
    if not ok:
        return user
    if user["is_admin"] != 1:
        return RedirectResponse("/notes")
    hashed = ph.hash(password)
    is_admin_flag = 1 if (is_admin == "on" or is_admin == "1") else 0
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            (username, hashed, is_admin_flag),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return templates.TemplateResponse(
            "create_user.html",
            {
                "request": request,
                "user": user,
                "error": "Пользователь с таким именем уже существует",
            },
        )
    conn.close()
    return RedirectResponse("/notes", status_code=302)


@app.get("/notes", response_class=HTMLResponse)
async def list_notes(request: Request):
    """Display the list of notes for the current user."""
    ok, user = require_login(request)
    if not ok:
        return user
    conn = get_db_connection()
    folders = conn.execute(
        "SELECT * FROM folders WHERE user_id = ? ORDER BY name COLLATE NOCASE",
        (user["id"],),
    ).fetchall()
    notes = conn.execute(
        "SELECT * FROM notes WHERE user_id = ? ORDER BY created_at DESC",
        (user["id"],),
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(
        "notes.html",
        {
            "request": request,
            "user": user,
            "folders": folders,
            "notes": notes,
        },
    )


@app.get("/folders/create", response_class=HTMLResponse)
async def create_folder_get(request: Request):
    """Render the create folder form."""
    ok, user = require_login(request)
    if not ok:
        return user
    # Include user in context for navigation
    return templates.TemplateResponse(
        "folder_form.html",
        {"request": request, "user": user},
    )


@app.post("/folders/create")
async def create_folder_post(request: Request, name: str = Form(...)):
    """Create a new folder for the current user."""
    ok, user = require_login(request)
    if not ok:
        return user
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO folders (user_id, name) VALUES (?, ?)", (user["id"], name)
    )
    conn.commit()
    conn.close()
    return RedirectResponse("/notes", status_code=302)


@app.post("/folders/delete/{folder_id}")
async def delete_folder(request: Request, folder_id: int):
    """Delete a folder and disassociate any notes within it."""
    ok, user = require_login(request)
    if not ok:
        return user
    conn = get_db_connection()
    # Set folder_id to NULL on notes belonging to this folder
    conn.execute(
        "UPDATE notes SET folder_id = NULL WHERE folder_id = ? AND user_id = ?",
        (folder_id, user["id"]),
    )
    conn.execute(
        "DELETE FROM folders WHERE id = ? AND user_id = ?", (folder_id, user["id"])
    )
    conn.commit()
    conn.close()
    return RedirectResponse("/notes", status_code=302)


@app.get("/notes/create", response_class=HTMLResponse)
async def create_note_get(request: Request):
    """Render the form to create a new note."""
    ok, user = require_login(request)
    if not ok:
        return user
    conn = get_db_connection()
    folders = conn.execute(
        "SELECT * FROM folders WHERE user_id = ? ORDER BY name COLLATE NOCASE",
        (user["id"],),
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(
        "note_form.html",
        {
            "request": request,
            "user": user,
            "folders": folders,
            "note": None,
        },
    )


@app.post("/notes/create")
async def create_note_post(
    request: Request,
    title: str = Form(...),
    folder_id: Optional[str] = Form(None),
    content_html: str = Form(""),
    note_password: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    """Handle creation of a new note, including attachments and optional password."""
    ok, user = require_login(request)
    if not ok:
        return user
    folder = None
    if folder_id and folder_id != "none":
        try:
            folder = int(folder_id)
        except ValueError:
            folder = None
    hashed_note_password: Optional[str] = None
    if note_password:
        hashed_note_password = ph.hash(note_password)
    conn = get_db_connection()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO notes (user_id, folder_id, title, content, password, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (user["id"], folder, title, content_html, hashed_note_password, now),
    )
    note_id = cur.lastrowid
    conn.commit()
    # Handle attachments
    if files:
        for upload in files:
            if not upload.filename:
                continue
            original_name = upload.filename
            # Prepend a random token to avoid name collisions
            unique_name = f"{secrets.token_hex(8)}_{original_name}"
            save_path = os.path.join(ATTACHMENTS_DIR, unique_name)
            # Save file to disk in chunks, mirroring Django’s recommendation
            # to avoid loading large files into memory【655965192673326†L134-L142】
            with open(save_path, "wb") as out_file:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
            ext = os.path.splitext(original_name)[1].lower()
            is_audio = 1 if ext in [".mp3", ".wav", ".ogg", ".m4a"] else 0
            conn.execute(
                "INSERT INTO attachments (note_id, filename, filepath, is_audio)"
                " VALUES (?, ?, ?, ?)",
                (note_id, original_name, unique_name, is_audio),
            )
    conn.commit()
    conn.close()
    return RedirectResponse(f"/notes/{note_id}", status_code=302)


@app.get("/notes/{note_id}", response_class=HTMLResponse)
async def view_note(request: Request, note_id: int):
    """Display a single note.  If a password is set, prompt for it before showing the content."""
    ok, user = require_login(request)
    if not ok:
        return user
    conn = get_db_connection()
    note = conn.execute(
        "SELECT * FROM notes WHERE id = ? AND user_id = ?", (note_id, user["id"])
    ).fetchone()
    if note is None:
        conn.close()
        return RedirectResponse("/notes")
    attachments = conn.execute(
        "SELECT * FROM attachments WHERE note_id = ?", (note_id,)
    ).fetchall()
    conn.close()
    # If the note has a password, prompt the user every time.  We do not
    # persist authorization across requests to avoid implementing session
    # storage.  The user must re‑enter the password on each visit.
    if note["password"]:
        return templates.TemplateResponse(
            "note_password.html",
            {"request": request, "user": user, "note_id": note_id},
        )
    return templates.TemplateResponse(
        "note_detail.html",
        {
            "request": request,
            "user": user,
            "note": note,
            "attachments": attachments,
        },
    )


@app.post("/notes/{note_id}/check_password")
async def check_note_password(
    request: Request, note_id: int, password: str = Form(...)
):
    """Verify the password for a password‑protected note."""
    ok, user = require_login(request)
    if not ok:
        return user
    conn = get_db_connection()
    note = conn.execute(
        "SELECT * FROM notes WHERE id = ? AND user_id = ?", (note_id, user["id"])
    ).fetchone()
    conn.close()
    if note is None:
        return RedirectResponse("/notes")
    try:
        ph.verify(note["password"], password)
        # Upon successful verification, render the note directly.  We do not
        # persist authorization beyond this request, so the password will be
        # required again the next time the note is accessed.
        # Fetch attachments again for rendering
        conn2 = get_db_connection()
        attachments = conn2.execute(
            "SELECT * FROM attachments WHERE note_id = ?", (note_id,)
        ).fetchall()
        conn2.close()
        return templates.TemplateResponse(
            "note_detail.html",
            {
                "request": request,
                "user": user,
                "note": note,
                "attachments": attachments,
            },
        )
    except Exception:
        return templates.TemplateResponse(
            "note_password.html",
            {
                "request": request,
                "user": user,
                "note_id": note_id,
                "error": "Неверный пароль",
            },
        )


@app.post("/notes/delete/{note_id}")
async def delete_note(request: Request, note_id: int):
    """Delete a note and its associated attachments."""
    ok, user = require_login(request)
    if not ok:
        return user
    conn = get_db_connection()
    # Fetch attachments to remove files from disk
    attachments = conn.execute(
        "SELECT * FROM attachments WHERE note_id = ?", (note_id,)
    ).fetchall()
    for att in attachments:
        file_path = os.path.join(ATTACHMENTS_DIR, att["filepath"])
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
    conn.execute("DELETE FROM attachments WHERE note_id = ?", (note_id,))
    conn.execute(
        "DELETE FROM notes WHERE id = ? AND user_id = ?", (note_id, user["id"])
    )
    conn.commit()
    conn.close()
    return RedirectResponse("/notes", status_code=302)


@app.get("/attachments/{filename}")
async def get_attachment(request: Request, filename: str):
    """Serve an attachment file if the current user owns the associated note."""
    ok, user = require_login(request)
    if not ok:
        return ok
    conn = get_db_connection()
    att = conn.execute(
        "SELECT * FROM attachments WHERE filepath = ?", (filename,)
    ).fetchone()
    if not att:
        conn.close()
        return Response(status_code=404)
    note = conn.execute(
        "SELECT * FROM notes WHERE id = ?", (att["note_id"],)
    ).fetchone()
    conn.close()
    if note is None or note["user_id"] != user["id"]:
        return Response(status_code=403)
    file_path = os.path.join(ATTACHMENTS_DIR, filename)
    if not os.path.exists(file_path):
        return Response(status_code=404)
    return FileResponse(file_path, filename=att["filename"])