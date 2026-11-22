//! Project-local recovery for the custom starter form. A nonblocking advisory
//! lock keeps two cooperating TUI processes from replacing each other's drafts.
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CustomScaffoldDraft {
    pub specs: String,
    pub output: String,
    #[serde(skip)]
    pub unsaved: bool,
}

#[cfg(unix)]
pub(crate) struct DraftStore {
    directory: PathBuf,
    _lock: fs::File,
}

#[cfg(unix)]
impl DraftStore {
    pub fn open(project: &Path) -> io::Result<Self> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;
        let directory = project.join(".model-maven");
        match fs::symlink_metadata(&directory) {
            Ok(metadata) if !metadata.is_dir() || metadata.file_type().is_symlink() => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "project metadata directory is not a real directory",
                ));
            }
            Ok(_) => (),
            Err(error) if error.kind() == io::ErrorKind::NotFound => fs::create_dir(&directory)?,
            Err(error) => return Err(error),
        }
        let lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(directory.join("scaffold-draft.lock"))?;
        // SAFETY: the file descriptor remains open for the lifetime of DraftStore.
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            directory,
            _lock: lock,
        })
    }

    pub fn load(&self) -> io::Result<Option<CustomScaffoldDraft>> {
        let path = self.directory.join("scaffold-draft.json");
        let metadata = match fs::symlink_metadata(&path) {
            Ok(value) => value,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 65536 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "draft is not a regular file or exceeds 64 KiB",
            ));
        }
        let bytes = fs::read(path)?;
        let draft = serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(Some(draft))
    }

    pub fn save(&self, draft: &CustomScaffoldDraft) -> io::Result<()> {
        let bytes = serde_json::to_vec(draft)?;
        if bytes.len() > 65536 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "draft exceeds 64 KiB",
            ));
        }
        let mut temp = tempfile::NamedTempFile::new_in(&self.directory)?;
        temp.write_all(&bytes)?;
        temp.as_file().sync_all()?;
        temp.persist(self.directory.join("scaffold-draft.json"))
            .map_err(|error| error.error)?;
        Ok(())
    }

    pub fn clear(&self) -> io::Result<()> {
        match fs::remove_file(self.directory.join("scaffold-draft.json")) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

#[cfg(not(unix))]
pub(crate) struct DraftStore;
#[cfg(not(unix))]
impl DraftStore {
    pub fn open(_: &Path) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "persistent scaffold drafts are unavailable on this platform",
        ))
    }
    pub fn load(&self) -> io::Result<Option<CustomScaffoldDraft>> {
        unreachable!()
    }
    pub fn save(&self, _: &CustomScaffoldDraft) -> io::Result<()> {
        unreachable!()
    }
    pub fn clear(&self) -> io::Result<()> {
        unreachable!()
    }
}
