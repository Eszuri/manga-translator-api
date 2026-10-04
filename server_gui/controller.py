"""Non-blocking ownership, startup checks, and shutdown of the desktop worker."""
import codecs
from dataclasses import replace
import json
from pathlib import Path
import secrets
import sys

from PySide6.QtCore import QElapsedTimer, QObject, QProcess, QProcessEnvironment, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkProxy, QNetworkReply, QNetworkRequest

from app.core.paths import missing_model_files
from server_gui.paths import application_dir, normalize_model_dir
from server_gui.preferences import Preferences, save_preferences


class ServerController(QObject):
    state_changed = Signal(str)
    log_line = Signal(str)
    error = Signal(str)
    provider_changed = Signal(str)
    progress_changed = Signal(str)
    STARTUP_TIMEOUT_MS = 180000
    HEALTH_STALE_MS = 30000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = "Stopped"
        self.process = None
        self.instance = ""
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._generation = 0
        self._stopping = False
        self._failed_shutdown = False
        self._last_error = ""
        self._provider = ""
        self._reply = None
        self._elapsed = QElapsedTimer()
        self._last_healthy = QElapsedTimer()
        self._network = QNetworkAccessManager(self)
        # This private loopback endpoint must not trigger system proxy discovery.
        self._network.setProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))
        self._timer = QTimer(self)
        self._timer.setInterval(750)
        self._timer.timeout.connect(self._check_health)
        self._startup_timer = QTimer(self)
        self._startup_timer.setSingleShot(True)
        self._startup_timer.timeout.connect(self._startup_timeout)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_stop)

    @property
    def active(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def _set_state(self, state: str):
        if self.state == state:
            return
        self.state = state
        self.state_changed.emit(state)

    def start(self, prefs: Preferences, settings_path: Path):
        if self.active or self.state in ("Starting", "Stopping"):
            return
        self._generation += 1
        generation = self._generation
        self._cancel_health()
        self._timer.stop()
        self._startup_timer.stop()
        self._kill_timer.stop()
        self._provider = ""
        self.provider_changed.emit("Not ready")
        try:
            prefs.validate()
            prefs = replace(prefs, model_dir=str(normalize_model_dir(prefs.model_dir)))
            missing = missing_model_files(Path(prefs.model_dir))
            if missing:
                raise FileNotFoundError("Missing or empty model files:\n" + "\n".join(missing))
            save_preferences(prefs, settings_path)
        except (OSError, ValueError, TypeError) as exc:
            self._set_state("Failed")
            self.error.emit(str(exc))
            return
        self.port = prefs.port
        self.instance = secrets.token_hex(24)
        self._stopping = False
        self._failed_shutdown = False
        self._last_error = ""
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._last_healthy.invalidate()
        if self.process:
            self.process.deleteLater()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("MANGA_GUI_INSTANCE", self.instance)
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        work_directory = application_dir() if getattr(sys, "frozen", False) else application_dir() / "api"
        self.process.setWorkingDirectory(str(work_directory))
        arguments = ["--server-worker", "--settings-file", str(settings_path.resolve())]
        if not getattr(sys, "frozen", False):
            arguments = ["-u", "-m", "tools.run_gui", *arguments]
        self.process.setProgram(sys.executable)
        self.process.setArguments(arguments)
        process = self.process
        process.started.connect(lambda: self._started(process, generation))
        process.readyReadStandardOutput.connect(lambda: self._read_output(process, generation))
        process.finished.connect(lambda code, status: self._finished(process, generation, code, status))
        process.errorOccurred.connect(lambda error: self._process_error(process, generation, error))
        self._set_state("Starting")
        self.provider_changed.emit("Checking DirectML...")
        self.progress_changed.emit("Checking DirectML support...")
        self._elapsed.start()
        self._startup_timer.start(self.STARTUP_TIMEOUT_MS)
        self._timer.start()
        process.start()

    def _started(self, process, generation):
        if process is self.process and generation == self._generation and self._stopping:
            # Stop may arrive while QProcess is still creating the child.
            process.kill()

    def _read_output(self, process, generation, final=False):
        if process is not self.process or generation != self._generation:
            return
        self._buffer += self._decoder.decode(bytes(process.readAllStandardOutput()), final=final)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))
        if final and self._buffer:
            self._handle_line(self._buffer.rstrip("\r"))
            self._buffer = ""

    def _handle_line(self, line):
        if line.startswith("GUI_EVENT "):
            try:
                event = json.loads(line[len("GUI_EVENT "):])
                if not isinstance(event, dict):
                    raise ValueError("Invalid worker event")
                # Provider availability alone does not prove that both models loaded.
                if self.state == "Starting" and ("gpu_provider" in event or event.get("phase") == "loading_models"):
                    self.provider_changed.emit("Validating GPU models...")
                message = event.get("message")
                if self.state == "Starting":
                    if isinstance(message, str) and message.strip():
                        self.progress_changed.emit(message.strip())
                    elif event.get("phase") == "loading_models":
                        self.progress_changed.emit("Loading GPU models...")
                if isinstance(event.get("error"), str) and event["error"]:
                    self._last_error = event["error"]
                    self.error.emit(self._last_error)
            except (ValueError, TypeError):
                self.log_line.emit(line)
        elif line:
            self.log_line.emit(line)

    def _process_error(self, process, generation, error):
        if process is not self.process or generation != self._generation:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self._timer.stop()
            self._startup_timer.stop()
            self._kill_timer.stop()
            self._cancel_health()
            self._last_error = "Could not launch the server process: " + process.errorString()
            self.process = None
            process.deleteLater()
            self.provider_changed.emit("Not ready")
            self._set_state("Failed")
            self.error.emit(self._last_error)

    def _finished(self, process, generation, code, exit_status):
        if process is not self.process or generation != self._generation:
            return
        self._timer.stop()
        self._startup_timer.stop()
        self._kill_timer.stop()
        self._cancel_health()
        self._read_output(process, generation, final=True)
        self.process = None
        process.deleteLater()
        self._provider = ""
        self.provider_changed.emit("Not running")
        if self._stopping and not self._failed_shutdown:
            self._set_state("Stopped")
        else:
            self._set_state("Failed")
            if not self._last_error:
                reason = "crashed" if exit_status == QProcess.ExitStatus.CrashExit else "stopped unexpectedly"
                self._last_error = f"Server process {reason} (exit code {code}). Open the log folder for details."
                self.error.emit(self._last_error)

    def _cancel_health(self):
        reply, self._reply = self._reply, None
        if reply is not None:
            # Detach first: abort() can synchronously emit finished().
            reply.abort()

    def _check_health(self):
        if not self.active or self._stopping:
            return
        if (self.state == "Running" and self._last_healthy.isValid()
                and self._last_healthy.elapsed() >= self.HEALTH_STALE_MS):
            self._set_state("Unresponsive")
            self.log_line.emit("The server has not answered health checks for 30 seconds. Waiting for it to respond.")
        if self._reply is not None:
            return
        request = QNetworkRequest(QUrl(f"http://127.0.0.1:{self.port}/internal/gui/health"))
        request.setRawHeader(b"X-Server-Instance", self.instance.encode())
        request.setTransferTimeout(5000)
        reply = self._network.get(request)
        self._reply = reply
        generation = self._generation
        instance = self.instance
        reply.finished.connect(lambda: self._health_result(reply, generation, instance))

    def _startup_timeout(self):
        if self.state != "Starting" or not self.active or self._stopping:
            return
        remaining = self.STARTUP_TIMEOUT_MS - self._elapsed.elapsed()
        if remaining > 0:
            # Coarse Qt timers may fire early; preserve the full startup allowance.
            self._startup_timer.start(remaining)
            return
        self._last_error = (
            f"Server did not become ready within {self.STARTUP_TIMEOUT_MS // 1000} seconds. "
            "Check the model/GPU startup log."
        )
        self.error.emit(self._last_error)
        self._failed_shutdown = True
        self.stop()

    def _health_result(self, reply, generation, instance):
        try:
            if (generation != self._generation or self._reply is not reply
                    or self._stopping or not self.active or instance != self.instance):
                return
            healthy = False
            if reply.error() == QNetworkReply.NetworkError.NoError:
                try:
                    result = json.loads(bytes(reply.readAll()).decode())
                    healthy = (isinstance(result, dict) and result.get("instance") == instance
                               and result.get("gpu_provider") == "DmlExecutionProvider")
                except (ValueError, UnicodeDecodeError):
                    pass
                if healthy:
                    self._startup_timer.stop()
                    self._last_healthy.start()
                    if self._provider != result["gpu_provider"]:
                        self._provider = result["gpu_provider"]
                        self.provider_changed.emit(self._provider)
                    self._set_state("Running")
        finally:
            if self._reply is reply:
                self._reply = None
            reply.deleteLater()

    def stop(self):
        if not self.active:
            return
        if self._stopping:
            self.force_stop()
            return
        starting = self.state == "Starting"
        self._stopping = True
        self._timer.stop()
        self._startup_timer.stop()
        self._cancel_health()
        self._set_state("Stopping")
        if starting:
            # Native model initialization cannot observe the stdin stop command.
            self.log_line.emit("Cancelling startup; stopping the owned worker process.")
            self.process.kill()
            return
        self.process.write(b"STOP\n")
        self.process.closeWriteChannel()
        self._kill_timer.start(15000)

    def force_stop(self):
        if self.active and self._stopping:
            self._kill_timer.stop()
            self.log_line.emit("Force stop requested; stopping the owned worker process.")
            self.process.kill()

    def _force_stop(self):
        if self.active and self._stopping:
            self.log_line.emit("Graceful shutdown exceeded 15 seconds; stopping the owned worker process.")
            self.process.kill()
