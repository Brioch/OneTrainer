# Force matplotlib and pydantic into sys.modules before PySide6/shiboken installs
# its import hooks, see scripts/train_ui_qt.py for details.
import matplotlib.pyplot  # noqa: F401, ICN001
import pydantic._internal._validators  # noqa: F401
from util.import_util import script_imports

script_imports()

from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.PySide6CaptionUIView import PySide6CaptionUIView
from modules.util.args.CaptionUIArgs import CaptionUIArgs
from modules.util.ui.pyside6_util import create_application


def main():
    args = CaptionUIArgs.parse_args()

    app = create_application()  # noqa: F841 - must stay alive while the window runs
    ui = CaptionUIController(args.dir, args.include_subdirectories).create_window(None, PySide6CaptionUIView)
    ui.exec()


if __name__ == '__main__':
    main()
