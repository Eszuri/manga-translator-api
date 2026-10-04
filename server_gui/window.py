"""Desktop control panel; inference runs in an owned worker process."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtNetwork import QAbstractSocket, QNetworkInterface
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from server_gui.controller import ServerController
from server_gui.paths import icon_path, model_directory_status, normalize_model_dir, user_data_dir
from server_gui.preferences import Preferences, load_preferences, save_preferences
from server_gui.theme import STYLESHEET, ServerSpinBox, apply_theme


def label(text="", style="", wrap=False):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(wrap)
    if style:
        widget.setObjectName(style)
    return widget


def card(title, parent_layout):
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 18)
    layout.setSpacing(12)
    layout.addWidget(label(title, "sectionTitle"))
    parent_layout.addWidget(frame)
    return layout


def scroll_page(content):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(content)
    return scroll


class ServerWindow(QMainWindow):
    def __init__(self, settings_path: Path | None = None, auto_start: bool = False):
        super().__init__()
        apply_theme(QApplication.instance())
        self.settings_path = (settings_path or user_data_dir() / "settings.json").resolve()
        self._restart_requested = False
        self._close_requested = False
        self._active_preferences = None
        self._initial_error = ""
        self._notice_source = ""
        self._state = "Stopped"
        self._startup_message = "Preparing server process"
        self._startup_elapsed = QElapsedTimer()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self.refresh_startup_status)
        try:
            self.preferences = load_preferences(self.settings_path)
        except (OSError, ValueError, TypeError) as exc:
            self.preferences = Preferences()
            self._initial_error = f"Saved settings could not be loaded: {exc}. The file was not changed."
        self.controller = ServerController(self)
        self.setWindowTitle("Manga Translator Server")
        self.setWindowIcon(QIcon(str(icon_path())))
        # Qt screen geometry is already DPI-scaled; leave room for the title bar.
        available = self.screen().availableGeometry()
        width = max(680, available.width() - 40)
        height = max(460, available.height() - 60)
        self.setMinimumSize(min(760, width), min(600, height))
        self.resize(min(980, width), min(760, height))
        self.setStyleSheet(STYLESHEET)
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(14)
        heading = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(4)
        titles.addWidget(label("Manga Translator", "heading"))
        titles.addWidget(label("SERVER CONTROL PANEL  /  WINDOWS · DIRECTML", "eyebrow"))
        heading.addLayout(titles, 1)
        self.state_label = label("STOPPED", "status")
        heading.addWidget(self.state_label, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(heading)
        controls = QHBoxLayout()
        self.start_button = QPushButton("Start server")
        self.start_button.setObjectName("primary")
        self.stop_button = QPushButton("Stop")
        self.restart_button = QPushButton("Restart")
        self.start_button.clicked.connect(self.start_server)
        self.stop_button.clicked.connect(self.stop_server)
        self.restart_button.clicked.connect(self.restart_server)
        for button in (self.start_button, self.stop_button, self.restart_button):
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        self.state_detail = label("Server is not running.", "muted", True)
        layout.addWidget(self.state_detail)
        self.notice = QFrame()
        self.notice.setObjectName("notice")
        notice_layout = QHBoxLayout(self.notice)
        notice_layout.setContentsMargins(12, 8, 12, 8)
        self.error_label = label("", wrap=True)
        self.error_label.setMaximumHeight(42)
        notice_layout.addWidget(self.error_label, 1)
        details = QPushButton("View log")
        details.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        notice_layout.addWidget(details)
        layout.addWidget(self.notice)
        self.notice.hide()
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_overview()
        self._build_settings()
        self._build_logs()
        self.controller.state_changed.connect(self.update_state)
        self.controller.log_line.connect(self.append_log)
        self.controller.error.connect(lambda text: self.show_error(text, source="server"))
        self.controller.provider_changed.connect(self.update_provider)
        self.controller.progress_changed.connect(self.update_progress)
        for field in (self.model_input, self.url_input, self.llm_model_input, self.key_input):
            field.textChanged.connect(self.settings_edited)
        self.port_input.valueChanged.connect(self.settings_edited)
        self.timeout_input.valueChanged.connect(self.settings_edited)
        self.network_input.toggled.connect(self.settings_edited)
        self.translator_input.currentIndexChanged.connect(self.settings_edited)
        self.model_input.textChanged.connect(self.check_model_selection)
        self.update_state("Stopped")
        self.check_model_selection()
        if self._initial_error:
            self.show_error(self._initial_error)
            self.tabs.setCurrentIndex(1)
        elif auto_start and self._models_ready(self.preferences):
            QTimer.singleShot(250, self.start_server)

    def _build_overview(self):
        overview = QWidget()
        layout = QVBoxLayout(overview)
        layout.setContentsMargins(0, 16, 8, 0)
        layout.setSpacing(14)
        connection = card("Connect to the API", layout)
        connection.addWidget(label("Local address · this PC", "muted"))
        row = QHBoxLayout()
        self.address_input = QLineEdit()
        self.address_input.setReadOnly(True)
        row.addWidget(self.address_input, 1)
        copy = QPushButton("Copy")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.local_address()))
        row.addWidget(copy)
        self.docs_button = QPushButton("Open API docs")
        self.docs_button.clicked.connect(self.open_docs)
        row.addWidget(self.docs_button)
        connection.addLayout(row)
        self.network_hint = label("", "muted", True)
        connection.addWidget(self.network_hint)
        self.network_row = QWidget()
        row = QHBoxLayout(self.network_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.network_addresses = QComboBox()
        row.addWidget(self.network_addresses, 1)
        copy_network = QPushButton("Copy")
        copy_network.clicked.connect(lambda: QApplication.clipboard().setText(self.network_addresses.currentText()))
        row.addWidget(copy_network)
        connection.addWidget(self.network_row)
        runtime = card("Processing", layout)
        self.gpu_label = label("GPU not active", "value")
        runtime.addWidget(self.gpu_label)
        runtime.addWidget(label("GPU inference is required. CPU fallback is disabled.", "muted", True))
        self.translator_label = label("", wrap=True)
        runtime.addWidget(self.translator_label)
        models = card("Model files", layout)
        self.models_label = label("", wrap=True)
        models.addWidget(self.models_label)
        row = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setReadOnly(True)
        row.addWidget(self.model_path, 1)
        self.configure_models_button = QPushButton("Change folder")
        self.configure_models_button.clicked.connect(self.open_model_settings)
        row.addWidget(self.configure_models_button)
        models.addLayout(row)
        models.addWidget(label("Models stay in the selected folder. They are not copied or downloaded.", "muted", True))
        layout.addStretch()
        self.tabs.addTab(scroll_page(overview), "Overview")

    @staticmethod
    def _form(parent=None):
        form = QFormLayout(parent) if parent is not None else QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(14)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return form

    def _build_settings(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_panel = QWidget()
        layout = QVBoxLayout(self.settings_panel)
        layout.setContentsMargins(0, 16, 8, 4)
        layout.setSpacing(14)
        models = card("Model folder", layout)
        row = QHBoxLayout()
        self.model_input = QLineEdit(self.preferences.model_dir)
        row.addWidget(self.model_input, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.choose_models)
        row.addWidget(browse)
        models.addLayout(row)
        self.model_status = label("", "muted", True)
        models.addWidget(self.model_status)
        server = card("Server", layout)
        form = self._form()
        self.port_input = ServerSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(self.preferences.port)
        self.port_input.setMaximumWidth(180)
        form.addRow("API port", self.port_input)
        self.network_input = QCheckBox("Allow access from other devices")
        self.network_input.setChecked(self.preferences.allow_network)
        form.addRow("Network access", self.network_input)
        server.addLayout(form)
        server.addWidget(label("Use a trusted network or Tailscale. This API has no login; Windows Firewall rules are not changed.", "muted", True))
        translation = card("Translation", layout)
        form = self._form()
        self.translator_input = QComboBox()
        self.translator_input.addItem("Google Translate", "google")
        self.translator_input.addItem("LLM (OpenAI-compatible)", "llm")
        self.translator_input.setCurrentIndex(0 if self.preferences.translator == "google" else 1)
        form.addRow("Default translator", self.translator_input)
        translation.addLayout(form)
        self.translation_hint = label("", "muted", True)
        translation.addWidget(self.translation_hint)
        self.llm_toggle = QCheckBox("Configure LLM")
        self.llm_toggle.setChecked(self.preferences.translator == "llm" or bool(self.preferences.llm_model or self.preferences.api_key))
        translation.addWidget(self.llm_toggle)
        self.llm_panel = QWidget()
        form = self._form(self.llm_panel)
        form.setContentsMargins(0, 4, 0, 0)
        self.url_input = QLineEdit(self.preferences.llm_base_url)
        self.url_input.setPlaceholderText("http://127.0.0.1:11434/v1")
        form.addRow("Base URL", self.url_input)
        self.llm_model_input = QLineEdit(self.preferences.llm_model)
        self.llm_model_input.setPlaceholderText("Exact model ID on your LLM server")
        form.addRow("Model ID", self.llm_model_input)
        self.key_input = QLineEdit(self.preferences.api_key)
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("API key", self.key_input)
        self.reveal_key = QCheckBox("Show API key")
        self.reveal_key.toggled.connect(lambda show: self.key_input.setEchoMode(
            QLineEdit.EchoMode.Normal if show else QLineEdit.EchoMode.Password))
        form.addRow("", self.reveal_key)
        self.timeout_input = ServerSpinBox()
        self.timeout_input.setRange(1, 3600)
        self.timeout_input.setSuffix(" seconds")
        self.timeout_input.setValue(int(self.preferences.llm_timeout))
        form.addRow("Request timeout", self.timeout_input)
        form.addRow("", label("Use a base URL without /chat/completions. Local LLM servers can use an empty key.", "muted", True))
        translation.addWidget(self.llm_panel)
        self.llm_toggle.toggled.connect(self.llm_panel.setVisible)
        self.llm_panel.setVisible(self.llm_toggle.isChecked())
        layout.addWidget(label("API keys are encrypted for your Windows account. Backend .env files are not used by this GUI.", "muted", True))
        layout.addStretch()
        page_layout.addWidget(scroll_page(self.settings_panel), 1)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 10, 8, 4)
        self.save_hint = label("No unsaved changes", "muted")
        footer.addWidget(self.save_hint, 1)
        self.save_button = QPushButton("Save settings")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self.save_settings)
        footer.addWidget(self.save_button)
        page_layout.addLayout(footer)
        self.tabs.addTab(page, "Settings")

    def _build_logs(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 16, 0, 0)
        row = QHBoxLayout()
        self.session_label = label("No server session yet", "muted")
        row.addWidget(self.session_label, 1)
        copy = QPushButton("Copy log")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.log.toPlainText()))
        row.addWidget(copy)
        clear = QPushButton("Clear view")
        clear.clicked.connect(lambda: self.log.clear())
        row.addWidget(clear)
        folder = QPushButton("Open log folder")
        folder.clicked.connect(self.open_log_folder)
        row.addWidget(folder)
        layout.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1200)
        self.log.setPlaceholderText("Startup checks, requests, and errors appear here.")
        layout.addWidget(self.log, 1)
        layout.addWidget(label("This view resets on Start. Earlier worker logs remain in the log folder.", "muted", True))
        self.tabs.addTab(page, "Logs")

    def local_address(self):
        prefs = self._active_preferences or self.preferences
        return f"http://127.0.0.1:{prefs.port}"

    def open_docs(self):
        if self._state == "Running" and self.controller.active:
            QDesktopServices.openUrl(QUrl(self.local_address() + "/docs"))

    def open_log_folder(self):
        directory = user_data_dir() / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def open_model_settings(self):
        self.tabs.setCurrentIndex(1)
        self.model_input.setFocus()

    def read_settings(self):
        return replace(self.preferences, model_dir=self.model_input.text().strip(),
                       port=self.port_input.value(), allow_network=self.network_input.isChecked(),
                       translator=self.translator_input.currentData(), llm_base_url=self.url_input.text().strip(),
                       llm_model=self.llm_model_input.text().strip(), api_key=self.key_input.text().strip(),
                       llm_timeout=float(self.timeout_input.value()))

    def settings_edited(self):
        dirty = self.read_settings() != self.preferences
        busy = self._state in ("Starting", "Stopping") or self._close_requested
        self.save_hint.setText("Unsaved changes" if dirty else "No unsaved changes")
        self.save_button.setText("Save and restart" if self.controller.active else "Save settings")
        self.save_button.setEnabled(not busy and (dirty or not self.settings_path.is_file() or bool(self._initial_error)))
        llm = self.translator_input.currentData() == "llm"
        self.translation_hint.setText(
            "Enter your LLM connection below. The extension can override the default translator."
            if llm else "Google needs an internet connection, but no API key. The extension can select LLM separately.")
        if llm:
            self.llm_toggle.setChecked(True)
        self.llm_toggle.setEnabled(not llm)

    def _validated_settings(self, require_models=False):
        prefs = self.read_settings()
        prefs.validate()
        prefs.model_dir = str(normalize_model_dir(prefs.model_dir))
        if require_models:
            directory, missing = model_directory_status(prefs.model_dir)
            if missing:
                raise ValueError(f"Missing or empty model files in {directory}:\n" + "\n".join(missing))
        return prefs

    def _persist(self, prefs):
        if prefs != self.preferences or not self.settings_path.is_file() or self._initial_error:
            save_preferences(prefs, self.settings_path)
        self.preferences = prefs
        self._initial_error = ""
        if self._notice_source == "settings":
            self.notice.hide()
            self.error_label.clear()
        self.model_input.setText(prefs.model_dir)
        self.settings_edited()
        self.update_details()

    def save_settings(self):
        try:
            self._persist(self._validated_settings(require_models=self.controller.active))
        except (OSError, ValueError, TypeError) as exc:
            self.show_error(str(exc))
            return False
        if self.controller.active:
            self._restart_requested = True
            self.controller.stop()
        else:
            self.save_hint.setText("Settings saved")
        return True

    def choose_models(self):
        directory = QFileDialog.getExistingDirectory(self, "Select model folder", self.model_input.text())
        if directory:
            self.model_input.setText(directory)

    def check_model_selection(self):
        try:
            if not self.model_input.text().strip():
                raise ValueError("Choose the folder containing comic-text-detector.onnx and manga-ocr.")
            directory, missing = model_directory_status(self.model_input.text().strip())
            self.model_status.setText(
                f"{len(missing)} required files are missing or empty. Select a complete model folder."
                if missing else "All required files found. Models will be verified when the server starts.")
            self.model_status.setToolTip("\n".join(missing) if missing else str(directory))
        except (OSError, ValueError) as exc:
            self.model_status.setText(str(exc))
            self.model_status.setToolTip("")

    @staticmethod
    def _models_ready(prefs):
        try:
            return bool(prefs.model_dir.strip()) and not model_directory_status(prefs.model_dir)[1]
        except (OSError, ValueError):
            return False

    def update_details(self):
        prefs = self._active_preferences or self.preferences
        self.address_input.setText(self.local_address())
        self.network_addresses.clear()
        addresses = set()
        if prefs.allow_network:
            for interface in QNetworkInterface.allInterfaces():
                if not interface.flags() & QNetworkInterface.InterfaceFlag.IsUp:
                    continue
                for entry in interface.addressEntries():
                    address = entry.ip()
                    if (address.protocol() == QAbstractSocket.NetworkLayerProtocol.IPv4Protocol
                            and not address.isLoopback() and not address.isNull()
                            and not address.toString().startswith("169.254.")):
                        addresses.add(address.toString())
        self.network_addresses.addItems([f"http://{address}:{prefs.port}" for address in sorted(addresses)])
        self.network_row.setVisible(bool(addresses))
        self.network_hint.setText(
            "Other devices · use the address for your network or Tailscale."
            if addresses else "Network access is enabled, but no external IPv4 address was found."
            if prefs.allow_network else "Other devices: access disabled. Enable it in Settings → Server.")
        self.model_path.setText(prefs.model_dir)
        self.model_path.setToolTip(prefs.model_dir)
        self.model_path.setCursorPosition(0)
        ready = self._models_ready(prefs)
        self.models_label.setText("All required files found" if ready else "Setup needed — choose a folder containing all required models.")
        translator = "Google Translate" if prefs.translator == "google" else f"LLM · {prefs.llm_model}"
        self.translator_label.setText("Default translator: " + translator)
        if self._state == "Stopped":
            self.state_detail.setText("Ready to start." if ready else "Select your model folder to begin.")

    def _begin_session(self):
        self.log.clear()
        self.notice.hide()
        self.error_label.clear()
        self.session_label.setText("Session · " + datetime.now().strftime("%H:%M:%S"))
        self.append_log("Starting server. Checking settings and model files…")

    def start_server(self):
        if self.controller.active or self._close_requested:
            return
        self._begin_session()
        try:
            prefs = self._validated_settings(require_models=True)
            self._persist(prefs)
        except (OSError, ValueError, TypeError) as exc:
            self.show_error(str(exc))
            self.tabs.setCurrentIndex(1)
            return
        self._active_preferences = replace(prefs)
        self.controller.start(prefs, self.settings_path)

    def stop_server(self):
        self._restart_requested = False
        self.controller.stop()

    def update_progress(self, message):
        self._startup_message = message
        self.refresh_startup_status()

    def refresh_startup_status(self):
        if self._state == "Starting" and self._startup_elapsed.isValid():
            elapsed = self._startup_elapsed.elapsed() // 1000
            self.state_detail.setText(f"{self._startup_message} · {elapsed}s elapsed · Stop cancels startup")

    def restart_server(self):
        if not self.controller.active or self._state in ("Starting", "Stopping"):
            return
        try:
            self._persist(self._validated_settings(require_models=True))
        except (OSError, ValueError, TypeError) as exc:
            self.show_error(str(exc))
            self.tabs.setCurrentIndex(1)
            return
        self._restart_requested = True
        self.controller.stop()

    def append_log(self, text):
        bar = self.log.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        position = bar.value()
        self.log.appendPlainText(text)
        if not at_bottom:
            bar.setValue(position)

    def show_error(self, text, source="settings"):
        # Keep the header bounded; the Logs tab contains the full diagnostic.
        self._notice_source = source
        summary = text.splitlines()[0] if text else "An unknown error occurred."
        self.error_label.setText(summary[:180] + ("…" if len(summary) > 180 else ""))
        self.notice.show()
        self.append_log("ERROR: " + text)

    def update_provider(self, value):
        self.gpu_label.setText("DirectML · GPU active" if value == "DmlExecutionProvider" and self._state == "Running"
                               else "Checking GPU models…" if self._state == "Starting"
                               else "GPU status unavailable" if self._state == "Unresponsive" else "GPU not active")

    def update_state(self, state):
        self._state = state
        if state == "Starting":
            self._startup_message = "Preparing server process"
            self._startup_elapsed.start()
            self._status_timer.start()
        else:
            self._status_timer.stop()
        self.state_label.setText(state.upper())
        tone = "good" if state == "Running" else "bad" if state == "Failed" else "busy" if state in ("Starting", "Stopping", "Unresponsive") else "neutral"
        self.state_label.setProperty("tone", tone)
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        self.state_detail.setText({"Stopped": "Server is not running.", "Starting": "Loading GPU models…",
            "Running": "Ready for requests.", "Stopping": "Waiting for the server to stop…",
            "Unresponsive": "Health check delayed. See Logs.", "Failed": "Server stopped with an error."}.get(state, state))
        active = self.controller.active or state in ("Starting", "Stopping")
        self.start_button.setEnabled(not active and not self._close_requested)
        self.stop_button.setEnabled(active and not self._close_requested)
        self.stop_button.setText("Force stop" if state == "Stopping" else "Stop")
        self.restart_button.setEnabled(active and state in ("Running", "Unresponsive"))
        self.docs_button.setEnabled(state == "Running")
        self.settings_panel.setEnabled(state not in ("Starting", "Stopping") and not self._close_requested)
        self.configure_models_button.setEnabled(state not in ("Starting", "Stopping"))
        if state == "Running":
            self.notice.hide()
            self.error_label.clear()
            self.gpu_label.setText("DirectML · GPU active")
        else:
            self.update_provider("")
        if state in ("Stopped", "Failed") and not self.controller.active:
            self._active_preferences = None
            if self._close_requested:
                QTimer.singleShot(0, self.close)
            elif self._restart_requested:
                self._restart_requested = False
                QTimer.singleShot(0, self.start_server)
        self.settings_edited()
        self.update_details()
        self.refresh_startup_status()

    def closeEvent(self, event):
        if not self._close_requested and self.read_settings() != self.preferences:
            choice = QMessageBox.question(self, "Unsaved settings", "Save your settings before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if choice == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.StandardButton.Save:
                try:
                    self._persist(self._validated_settings())
                except (OSError, ValueError, TypeError) as exc:
                    self.show_error(str(exc))
                    event.ignore()
                    return
        self._close_requested = True
        self._restart_requested = False
        if self.controller.active:
            self.controller.stop()
            event.ignore()
        else:
            event.accept()
