from pathlib import Path
from lumachords.profile_store import ProfileStore
import platformdirs


class FileDialogUtils:
    FILTER_VIDEO_FILES = {"Video files (*.mp4, *.mkv, *.webm, *.avi, *.mov)": ["*.mp4", "*.mkv", "*.webm", "*.avi", "*.mov"], "All files (*.*)": "*.*"}

    @staticmethod
    def show_open_dialog(title: str, filter:str, start_dir: str=None) -> None:
        import crossfiledialog as cfd
        from crossfiledialog.exceptions import FileDialogException
        if not start_dir:
            if (profile_dir := ProfileStore.get_last_open_dir()) and (Path(profile_dir).is_dir()):
                start_dir = profile_dir
        if not start_dir:
            try:
                docs_dir = platformdirs.user_documents_dir()
                if docs_dir:
                    start_dir = docs_dir
            except:
                pass
        if not start_dir:
            start_dir = "."
        try:
            result = cfd.open_file(title=title, start_dir=start_dir, filter=filter)
            if result:
                ProfileStore.set_last_open_dir(str(Path(result).parent))
            return result
        except FileDialogException:
            return None
        except Exception as e:
            raise e
        
    @staticmethod
    def show_save_dialog(title: str, default_ext: str, start_dir: str=None) -> None:
        import crossfiledialog as cfd
        from crossfiledialog.exceptions import FileDialogException

        if not start_dir:
            if (profile_dir := ProfileStore.get_last_save_dir()) and (Path(profile_dir).is_dir()):
                start_dir = profile_dir
        if not start_dir:
            try:
                docs_dir = platformdirs.user_documents_dir()
                if docs_dir:
                    start_dir = docs_dir
            except:
                pass
        if not start_dir:
            start_dir = "."
        try:
            result = cfd.save_file(title=title, start_dir=start_dir)
            ext = Path(result).suffix
            if not ext:
                result += "." + default_ext
            ProfileStore.set_last_save_dir(str(Path(result).parent))
            return result
        except FileDialogException:
            return None
        except Exception as e:
            raise e
