import glob
from pathlib import Path

from atm.utils import ROOT


def check_suffix(file, suffix):
    if all([file, suffix]):
        if isinstance(file, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()
            if s:
                assert s in suffix, f"Acceptable suffix is {suffix}, not {s}"        


def check_file(file: str, suffix: str | list[str] | tuple[str]):
    check_suffix(file, suffix)
    file = file.strip()

    files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))    # search and find file
    if not files:
        raise FileNotFoundError(f"{file} doesn't exists.")
    elif len(files) > 1:
        raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
    
    return files[0] if len(files) else []  # return file


def check_yaml(file, suffix=('yaml', 'yml')):
    return check_file(file, suffix)