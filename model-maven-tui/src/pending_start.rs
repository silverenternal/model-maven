//! Durable, project-scoped markers for training requests without acknowledged receipts.
use std::{
    fs, io,
    path::{Path, PathBuf},
};

fn valid_id(id: &str) -> bool {
    id.starts_with("tui-")
        && (16..=64).contains(&id.len())
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

fn directory(project: &Path, create: bool) -> io::Result<PathBuf> {
    let metadata = project.join(".model-maven");
    let root = metadata.join("tui-pending-starts");
    for path in [&metadata, &root] {
        match fs::symlink_metadata(path) {
            Ok(value) if value.is_dir() && !value.file_type().is_symlink() => (),
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "pending request path is not a real directory",
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound && create && path == &root => {
                let mut builder = fs::DirBuilder::new();
                #[cfg(unix)]
                {
                    use std::os::unix::fs::DirBuilderExt;
                    builder.mode(0o700);
                }
                match builder.create(path) {
                    Ok(()) => sync_directory(&metadata)?,
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        let metadata = fs::symlink_metadata(path)?;
                        if !metadata.is_dir() || metadata.file_type().is_symlink() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "pending request path is not a real directory",
                            ));
                        }
                    }
                    Err(error) => return Err(error),
                }
            }
            Err(error) => return Err(error),
        }
    }
    Ok(root)
}

pub(crate) fn create(project: &Path, id: &str) -> io::Result<()> {
    if !valid_id(id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid request identity",
        ));
    }
    let root = directory(project, true)?;
    let path = root.join(id);
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(&path)?;
    if let Err(error) = file.sync_all().and_then(|_| sync_directory(&root)) {
        let _ = fs::remove_file(path);
        return Err(error);
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> io::Result<()> {
    fs::File::open(path)?.sync_all()
}
#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

pub(crate) fn list(project: &Path) -> io::Result<Vec<String>> {
    let root = match directory(project, false) {
        Ok(root) => root,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut result = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str().filter(|name| valid_id(name)) {
            let kind = entry.file_type()?;
            if !kind.is_file() || kind.is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "pending request marker is not a regular file",
                ));
            }
            result.push(name.to_owned());
            if result.len() > 256 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "too many pending training requests",
                ));
            }
        }
    }
    result.sort();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_are_project_scoped_and_require_valid_ids() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        fs::create_dir(a.path().join(".model-maven")).unwrap();
        fs::create_dir(b.path().join(".model-maven")).unwrap();
        let id = "tui-1234567890abcdef";
        create(a.path(), id).unwrap();
        assert_eq!(list(a.path()).unwrap(), vec![id]);
        assert!(list(b.path()).unwrap().is_empty());
        assert!(create(a.path(), id).is_err());
        assert!(clear(a.path(), "../other").is_err());
        clear(a.path(), id).unwrap();
        assert!(list(a.path()).unwrap().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_marker_cannot_hide_an_unresolved_start() {
        use std::os::unix::fs::symlink;
        let project = tempfile::tempdir().unwrap();
        fs::create_dir(project.path().join(".model-maven")).unwrap();
        let id = "tui-1234567890abcdef";
        create(project.path(), id).unwrap();
        let marker = project
            .path()
            .join(".model-maven/tui-pending-starts")
            .join(id);
        fs::remove_file(&marker).unwrap();
        symlink("missing", marker).unwrap();
        assert!(list(project.path()).is_err());
    }
}

pub(crate) fn clear(project: &Path, id: &str) -> io::Result<()> {
    if !valid_id(id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid request identity",
        ));
    }
    let root = directory(project, false)?;
    match fs::remove_file(root.join(id)) {
        Ok(()) => sync_directory(&root),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}
