"""Colors and scalable controls for the desktop panel."""

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QSpinBox, QStyle, QStyleOptionSpinBox


def apply_theme(app: QApplication) -> None:
    """Use one palette for native controls, popups, and custom-styled panels."""
    if app.property("mangaServerThemeApplied"):
        return
    app.setStyle("Fusion")
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#0d1422",
        QPalette.ColorRole.WindowText: "#e5ebf5",
        QPalette.ColorRole.Base: "#0c1525",
        QPalette.ColorRole.AlternateBase: "#131e30",
        QPalette.ColorRole.Text: "#e5ebf5",
        QPalette.ColorRole.Button: "#25334a",
        QPalette.ColorRole.ButtonText: "#e5ebf5",
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Highlight: "#2b64d8",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.ToolTipBase: "#20314d",
        QPalette.ColorRole.ToolTipText: "#e5ebf5",
        QPalette.ColorRole.PlaceholderText: "#8293ad",
        QPalette.ColorRole.Light: "#526b92",
        QPalette.ColorRole.Midlight: "#3b4b65",
        QPalette.ColorRole.Mid: "#2f405a",
        QPalette.ColorRole.Dark: "#0c1525",
        QPalette.ColorRole.Shadow: "#080d16",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#6f809b"))
    app.setPalette(palette)
    app.setProperty("mangaServerThemeApplied", True)


class ServerSpinBox(QSpinBox):
    """Keep native spin-button behavior with visible vector chevrons in dark mode."""

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons:
            return
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for control, step, direction in (
            (QStyle.SubControl.SC_SpinBoxUp, QAbstractSpinBox.StepEnabledFlag.StepUpEnabled, -1),
            (QStyle.SubControl.SC_SpinBoxDown, QAbstractSpinBox.StepEnabledFlag.StepDownEnabled, 1),
        ):
            rectangle = self.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, control, self)
            center = rectangle.center()
            enabled = self.isEnabled() and bool(option.stepEnabled & step)
            color = QColor("#c6d7ef" if enabled else "#53637b")
            painter.setPen(QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPolyline(QPolygonF([
                QPointF(center.x() - 3, center.y() - direction * 1.5),
                QPointF(center.x(), center.y() + direction * 1.5),
                QPointF(center.x() + 3, center.y() - direction * 1.5),
            ]))
        painter.end()


STYLESHEET = """
QMainWindow { background: #0d1422; }
QWidget { color: #e5ebf5; font-family: 'Segoe UI'; font-size: 13px; }
QLabel { background: transparent; }
QLabel#heading { font-size: 26px; font-weight: 700; }
QLabel#eyebrow { font-size: 10px; color: #8293ad; }
QLabel#muted { color: #9eacc2; }
QLabel#sectionTitle { font-size: 15px; font-weight: 600; }
QLabel#value { font-size: 18px; font-weight: 600; }
QLabel#status { border-radius: 6px; padding: 8px 12px; font-size: 11px; font-weight: 700; background: #23314a; color: #afbed5; }
QLabel#status[tone='good'] { background: #113b35; color: #72dfb9; }
QLabel#status[tone='busy'] { background: #3b3120; color: #f6cd7b; }
QLabel#status[tone='bad'] { background: #432734; color: #ffafb6; }
QFrame#card { background: #131e30; border: 1px solid #27354c; border-radius: 9px; }
QFrame#notice { background: #3a2531; border: 1px solid #74404d; border-radius: 6px; }
QFrame#notice QLabel { color: #ffc6cb; }
QPushButton { background: #25334a; border: 1px solid #3b4b65; border-radius: 6px; padding: 8px 14px; min-height: 20px; }
QPushButton:hover { background: #30415d; border-color: #5b739a; }
QPushButton:pressed { background: #1c2940; }
QPushButton:focus { border-color: #82b5ff; }
QPushButton#primary { background: #2b64d8; border-color: #4280ed; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #3574ec; }
QPushButton:disabled, QPushButton#primary:disabled { color: #6f809b; background: #172238; border-color: #2a3850; }
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit { background: #0c1525; border: 1px solid #3a4b65; border-radius: 5px; padding: 8px 10px; selection-background-color: #285cba; selection-color: white; }
QLineEdit, QSpinBox, QComboBox { min-height: 20px; }
QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: #6ca5ff; }
QLineEdit:read-only { color: #b7c8e3; border-color: #2f405a; }
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled, QCheckBox:disabled { color: #6f809b; }
QSpinBox { padding-right: 26px; }
QComboBox { padding-right: 30px; }
QComboBox QAbstractItemView { background: #17253d; color: #e5ebf5; selection-background-color: #2b64d8; border: 1px solid #526b92; }
QCheckBox { spacing: 10px; min-height: 28px; background: transparent; }
QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #7991b5; border-radius: 4px; background: #0c1525; }
QCheckBox::indicator:checked { background: #69a2ff; border: 4px solid #254e8d; }
QCheckBox::indicator:disabled { border-color: #43536d; }
QPlainTextEdit { font-family: Consolas, monospace; font-size: 12px; padding: 12px; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }
QTabWidget::pane { border: none; border-top: 1px solid #2b3b54; }
QTabBar::tab { background: transparent; color: #96a8c3; padding: 12px 20px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #b5d2ff; border-bottom: 2px solid #6a9ff2; }
QTabBar::tab:hover { color: #e5ebf5; }
QScrollBar:vertical { background: #0d1422; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3b4b65; min-height: 30px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QToolTip { background: #20314d; color: #e5ebf5; border: 1px solid #526b92; padding: 6px; }
QMessageBox { background: #131e30; }
"""
