"""Interfaz nativa de Windows para operar el sistema TTS."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import uuid
import winsound
from datetime import datetime
from pathlib import Path
from tkinter import END, BOTH, LEFT, WORD, StringVar, Tk, Toplevel
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from tts_project.config import RAW_DIR, ROOT_DIR
from tts_project.service import TTSService


class DesktopApp:
    """Ventana principal del sistema TTS para Windows."""

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("TTS Propio")
        self.root.geometry("960x700")
        self.service = TTSService()
        self.latest_audio: Path | None = None
        self.test_text = StringVar()
        self.log_window: Toplevel | None = None
        self.log_output: ScrolledText | None = None
        self._build_layout()

    def _build_layout(self) -> None:
        """Construye la interfaz con pestanas principales."""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=BOTH, expand=True, padx=12, pady=12)

        top_bar = ttk.Frame(self.root, padding=(12, 0, 12, 0))
        top_bar.pack(fill="x")
        ttk.Button(top_bar, text="Ver logs", command=self._open_log_window).pack(side=LEFT, pady=(0, 8))

        training_tab = ttk.Frame(notebook, padding=12)
        testing_tab = ttk.Frame(notebook, padding=12)
        notebook.add(training_tab, text="Entrenamiento")
        notebook.add(testing_tab, text="Probar modelo")

        self._build_training_tab(training_tab)
        self._build_testing_tab(testing_tab)

    def _build_training_tab(self, parent: ttk.Frame) -> None:
        """Crea el panel de entrenamiento."""
        paths_box = ScrolledText(parent, wrap=WORD, height=7)
        paths_box.pack(fill="x", pady=(0, 12))
        paths_box.insert(
            "1.0",
            "\n".join(
                [
                    "Archivos esperados:",
                    f"- Audio recomendado: {RAW_DIR / 'audio_entrenamiento.wav'}",
                    f"- Audio alterno: {RAW_DIR / 'audio_entrenamiento.mp3'}",
                    f"- Audio alterno: {RAW_DIR / 'audio_entrenamiento.flac'}",
                    f"- Texto: {RAW_DIR / 'texto_entrenamiento.txt'}",
                ]
            ),
        )
        paths_box.configure(state="disabled")

        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill="x", pady=(0, 12))
        actions = [
            ("Validar archivos", "scripts/preflight.py"),
            ("Preparar dataset", "scripts/prepare_dataset.py"),
            ("Entrenar modelo", "scripts/train.py"),
            ("Pipeline completo", "scripts/run_pipeline.py"),
        ]
        for label, script in actions:
            ttk.Button(
                buttons_frame,
                text=label,
                command=lambda script_path=script: self._run_script_async(script_path),
            ).pack(side=LEFT, padx=(0, 8))

        self.logs_box = ScrolledText(parent, wrap=WORD, height=24)
        self.logs_box.pack(fill=BOTH, expand=True)
        self._append_logs("La aplicacion esta lista. Usa los botones para entrenar o validar.")

    def _build_testing_tab(self, parent: ttk.Frame) -> None:
        """Crea el panel para generar y reproducir audio."""
        ttk.Label(parent, text="Texto para sintetizar").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.test_text).pack(fill="x", pady=(4, 12))

        actions_frame = ttk.Frame(parent)
        actions_frame.pack(fill="x", pady=(0, 12))
        ttk.Button(actions_frame, text="Generar audio", command=self._synthesize_async).pack(side=LEFT, padx=(0, 8))
        ttk.Button(actions_frame, text="Reproducir ultimo audio", command=self._play_latest_audio).pack(side=LEFT)

        ttk.Label(parent, text="Historial").pack(anchor="w")
        self.history_box = ScrolledText(parent, wrap=WORD, height=24)
        self.history_box.pack(fill=BOTH, expand=True)
        self.history_box.insert("1.0", "Escribe un texto y pulsa 'Generar audio'.\n")
        self.history_box.configure(state="disabled")

    def _run_script_async(self, script_path: str) -> None:
        """Ejecuta scripts del proyecto sin bloquear la ventana."""
        thread = threading.Thread(target=self._run_script, args=(script_path,), daemon=True)
        thread.start()

    def _run_script(self, script_path: str) -> None:
        """Lanza un script y escribe su salida en la caja de logs."""
        self._append_logs(f"\n>>> Ejecutando {script_path}\n")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            self._append_error(f"Estado: ERROR\n{output}\n")
            return
        self._append_logs(f"Estado: OK\n{output}\n")

    def _synthesize_async(self) -> None:
        """Genera audio sin congelar la interfaz."""
        thread = threading.Thread(target=self._synthesize, daemon=True)
        thread.start()

    def _synthesize(self) -> None:
        """Genera un WAV desde el texto actual."""
        text = self.test_text.get().strip()
        if not text:
            self._append_error("No se pudo sintetizar: el texto esta vacio.\n")
            self.root.after(0, lambda: messagebox.showwarning("Texto vacio", "Escribe un texto para sintetizar."))
            return
        try:
            output_path = Path(tempfile.gettempdir()) / f"tts_chat_{uuid.uuid4().hex}.wav"
            audio_path = self.service.synthesize_to_file(text, output_path)
            self.latest_audio = audio_path
            self._append_logs(f"Sintesis completada: {audio_path}\n")
            self._append_history(f"Tu: {text}\nSistema: Audio generado en {audio_path}\n")
        except Exception as error:  # noqa: BLE001
            self._append_error(f"Error durante la sintesis: {error}\n")
            self.root.after(0, lambda: messagebox.showerror("Error", str(error)))

    def _play_latest_audio(self) -> None:
        """Reproduce el ultimo archivo WAV generado."""
        if self.latest_audio is None or not self.latest_audio.exists():
            self._append_error("No hay un audio disponible para reproducir.\n")
            messagebox.showinfo("Sin audio", "Todavia no has generado un audio.")
            return
        winsound.PlaySound(str(self.latest_audio), winsound.SND_FILENAME | winsound.SND_ASYNC)
        self._append_logs(f"Reproduciendo audio: {self.latest_audio}\n")

    def _open_log_window(self) -> None:
        """Abre una ventanita de logs independiente."""
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.lift()
            self.log_window.focus_force()
            return

        self.log_window = Toplevel(self.root)
        self.log_window.title("Logs del sistema TTS")
        self.log_window.geometry("900x420")
        self.log_output = ScrolledText(self.log_window, wrap=WORD, height=24)
        self.log_output.pack(fill=BOTH, expand=True, padx=12, pady=12)
        self.log_output.insert("1.0", "Ventana de logs lista.\n")
        self.log_output.configure(state="disabled")

    def _append_logs(self, text: str) -> None:
        """Agrega texto a la caja de logs de forma segura."""
        def callback() -> None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            entry = f"[{timestamp}] {text}"
            self.logs_box.configure(state="normal")
            self.logs_box.insert(END, entry)
            self.logs_box.see(END)
            self.logs_box.configure(state="disabled")
            if self.log_output is not None and self.log_output.winfo_exists():
                self.log_output.configure(state="normal")
                self.log_output.insert(END, entry)
                self.log_output.see(END)
                self.log_output.configure(state="disabled")

        self.root.after(0, callback)

    def _append_error(self, text: str) -> None:
        """Agrega un error al log principal y a la ventanita de logs."""
        self._append_logs(f"[ERROR] {text}")
        self.root.after(0, self._open_log_window)

    def _append_history(self, text: str) -> None:
        """Agrega una entrada al historial de sintesis."""
        def callback() -> None:
            self.history_box.configure(state="normal")
            self.history_box.insert(END, text + "\n")
            self.history_box.see(END)
            self.history_box.configure(state="disabled")

        self.root.after(0, callback)

    def run(self) -> None:
        """Inicia el loop principal de la interfaz."""
        self.root.mainloop()
